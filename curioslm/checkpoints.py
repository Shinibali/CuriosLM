"""
Checkpoint format: save, load, rebuild a model from one, and stage it to a private Hugging
Face Hub repo so an ephemeral GPU node's local disk is never the only copy.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import datetime, timedelta, timezone

import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

from .config import GPTConfig
from .model import GPT

# Without a TTY (piped to a log file) every progress-bar refresh becomes a new appended line --
# hundreds of thousands of them for one multi-GB transfer.
disable_progress_bars()

CHECKPOINT_FILENAME = 'ckpt.pt'

# Saves land every few minutes, so this keeps at most a handful of recent versions per run while
# still far exceeding the seconds between an LFS upload and its commit.
PRUNE_MIN_AGE = timedelta(minutes=10)

# max_workers=1 serializes uploads: a save arriving before the previous upload finishes queues
# behind it rather than racing it for the same (repeatedly overwritten) remote path.
_upload_pool = ThreadPoolExecutor(max_workers=1)

# Derived from the dataclass, so a new GPTConfig knob propagates here automatically.
CONFIG_KEYS = tuple(f.name for f in fields(GPTConfig))

# Architecture must match the checkpoint exactly or the weights won't load. dropout is excluded:
# it owns no parameters and is legitimately changed when resuming.
ARCH_KEYS = tuple(k for k in CONFIG_KEYS if k != 'dropout')

_COMPILE_PREFIX = '_orig_mod.'


def strip_compile_prefix(state_dict):
    """Strip the '_orig_mod.' prefix torch.compile adds, so weights load into an
    uncompiled model."""
    for k in list(state_dict):
        if k.startswith(_COMPILE_PREFIX):
            state_dict[k[len(_COMPILE_PREFIX):]] = state_dict.pop(k)
    return state_dict


def save(path, model, optimizer, model_args, iter_num, best_val_loss, config,  train_loader_state=None):
    """Write the checkpoint atomically: temp file then rename, so a crash mid-write can't
    leave a truncated file where the only good checkpoint used to be."""
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
        "train_loader_state": train_loader_state, # None for datasets sampled at random
    }, f'{path}.tmp')
    os.replace(f'{path}.tmp', path)


def load(path, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


def check_no_overwrite(path, wandb_run_name):
    """Refuse to clobber a different run's checkpoint at this path, comparing the wandb_run_name
    each checkpoint already carries. For init_from='scratch' only; 'resume' overwrites on purpose."""
    if not os.path.exists(path):
        return
    existing_name = load(path, map_location='cpu').get('config', {}).get('wandb_run_name')
    if existing_name != wandb_run_name:
        raise RuntimeError(
            f"{path} already holds a checkpoint for run {existing_name!r}, not "
            f"{wandb_run_name!r} -- refusing to overwrite it. Use a different out_dir, or "
            "init_from='resume' if this really is meant to continue the same run."
        )


def upload_checkpoint(out_dir, repo_id):
    """Push out_dir/ckpt.pt to the staging repo at a path mirroring out_dir, so the repo's file
    listing itself says which run each checkpoint belongs to. Creates the repo on first use."""
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    path_in_repo = f"{out_dir}/{CHECKPOINT_FILENAME}"
    api.upload_file(
        path_or_fileobj=os.path.join(out_dir, CHECKPOINT_FILENAME),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"uploaded checkpoint to {repo_id}:{path_in_repo}")
    prune_old_versions(repo_id)


def prune_old_versions(repo_id):
    """Overwriting ckpt.pt keeps every old version, both in git history and as a stored LFS object,
    and all of it counts against the private-storage quota.
    Squash history to one commit, then permanently delete LFS objects no current file references.
    Separate from the upload so a failed prune is never reported as a failed upload."""
    try:
        api = HfApi()
        api.super_squash_history(repo_id=repo_id, repo_type="model")
        current = {s.lfs.sha256 for s in api.repo_info(repo_id, repo_type="model", files_metadata=True).siblings if s.lfs}
        stale = stale_orphans(api.list_lfs_files(repo_id, repo_type="model"), current,
                              older_than=datetime.now(timezone.utc) - PRUNE_MIN_AGE)
        if stale:
            api.permanently_delete_lfs_files(repo_id, stale, repo_type="model")
            print(f"pruned {len(stale)} old checkpoint version(s) from {repo_id}")
    except Exception as e:
        print(f"checkpoint prune failed (upload itself succeeded): {e!r}", flush=True)


def stale_orphans(lfs_files, current_oids, older_than):
    """LFS objects no current file points to, pushed before `older_than`. The age cut-off spares
    another run's upload that is stored but not yet committed -- a window of seconds."""
    return [f for f in lfs_files if f.file_oid not in current_oids and f.pushed_at < older_than]


def upload_checkpoint_async(out_dir, repo_id):
    """Non-blocking upload_checkpoint: returns a Future immediately so the training loop never
    stalls on the transfer. concurrent.futures' atexit hook joins the pool, so a run's final
    upload still finishes even though nothing here waits on it."""
    future = _upload_pool.submit(upload_checkpoint, out_dir, repo_id)
    future.add_done_callback(_report_upload_failure)
    return future


def _report_upload_failure(future):
    # Nothing ever reads the Future's result, so without this a failed upload would vanish silently.
    if future.exception() is not None:
        print(f"checkpoint upload FAILED: {future.exception()!r}", flush=True)


def download_checkpoint(out_dir, repo_id):
    """Pull <out_dir>/ckpt.pt from the staging repo into out_dir, to resume on a node that
    doesn't have it locally. local_dir='.' plus an out_dir-prefixed filename lands it at the
    same relative path it was uploaded from."""
    os.makedirs(out_dir, exist_ok=True)
    path_in_repo = f"{out_dir}/{CHECKPOINT_FILENAME}"
    hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type="model", local_dir=".")
    print(f"downloaded checkpoint from {repo_id}:{path_in_repo}")


def build_model(checkpoint, overrides=None):
    """Rebuild the model a checkpoint describes and load its weights. Architecture comes from
    the checkpoint; `overrides` changes non-architectural settings only (in practice: dropout).
    Returns (model, model_args) so the caller can carry model_args into later checkpoints."""
    saved = checkpoint['model_args']
    model_args = {k: saved[k] for k in CONFIG_KEYS if k in saved}
    for k, v in (overrides or {}).items():
        if k in CONFIG_KEYS and k not in ARCH_KEYS:
            model_args[k] = v

    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(strip_compile_prefix(checkpoint['model']))
    return model, model_args
