"""
Upload/download tokenized shards to/from a private Hugging Face Hub dataset repo -- the
staging mechanism between where shards get tokenized (prepare.py, run once, CPU-only, on
your Mac) and wherever GPU training actually runs (a RunPod node, which never runs
prepare.py's raw-text download/tokenization itself).


Every node downloads the *full* shard set (not a disjoint subset per node) -- see
prepare.py's docstring for why that's the right call at this project's scale.
"""
import os

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

from .dataset import data_dir

# dataset_name and repo_id are always passed in explicitly, sourced from DataConfig


def upload_shards(dataset_name, repo_id):
    """Push every .npy shard in data/<dataset_name>/ to the private HF repo, creating it
    (private) first if it doesn't exist yet."""
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    directory = data_dir(dataset_name)
    shard_files = sorted(f for f in os.listdir(directory) if f.endswith(".npy"))
    assert shard_files, f"no .npy shards found in {directory} to upload"

    for filename in shard_files:
        api.upload_file(
            path_or_fileobj=os.path.join(directory, filename),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"uploaded {filename}")

    return shard_files


def download_shards(local_dir, repo_id):
    """Pull every .npy shard from the private HF repo down into local_dir. Caller decides
    the destination explicitly (rather than defaulting to data_dir(dataset_name)) so a
    fresh/empty directory can be used to genuinely test a from-scratch node, not silently
    read shards that already happen to exist locally."""
    os.makedirs(local_dir, exist_ok=True)

    shard_files = sorted(f for f in list_repo_files(repo_id, repo_type="dataset") if f.endswith(".npy"))
    assert shard_files, f"no .npy shards found in {repo_id}"

    for filename in shard_files:
        hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=local_dir)
        print(f"downloaded {filename}")

    return shard_files


def cleanup_local_shards(dataset_name, repo_id):
    """Delete local shards, but only after confirming each one is safely on the remote repo
    (same filename AND matching file size). Never called automatically by upload_shards --
    deliberately a separate, explicit step: a mistaken deletion here means re-running the
    whole (tens-of-minutes) prepare.py pass to regenerate what was lost, so this only removes
    what it has actually verified, and leaves anything unconfirmed untouched."""
    directory = data_dir(dataset_name)
    local_files = {f: os.path.getsize(os.path.join(directory, f))
                   for f in os.listdir(directory) if f.endswith(".npy")}

    info = HfApi().repo_info(repo_id, repo_type="dataset", files_metadata=True)
    remote_sizes = {s.rfilename: s.size for s in info.siblings}

    deleted, skipped = [], []
    for filename, local_size in local_files.items():
        if remote_sizes.get(filename) == local_size:
            os.remove(os.path.join(directory, filename))
            deleted.append(filename)
        else:
            skipped.append(filename)

    if skipped:
        print(f"NOT deleted (not confirmed on {repo_id} with a matching size): {skipped}")
    print(f"deleted {len(deleted)} local shard(s), kept {len(skipped)}")
    return deleted, skipped


if __name__ == "__main__":
    import sys
    from ..config import DataConfig

    cfg = DataConfig()
    command = sys.argv[1] if len(sys.argv) > 1 else "upload"

    if command == "upload":
        upload_shards(cfg.dataset, cfg.hf_repo_id)
    elif command == "download":
        download_shards(data_dir(cfg.dataset), cfg.hf_repo_id)
    elif command == "cleanup":
        cleanup_local_shards(cfg.dataset, cfg.hf_repo_id)
    else:
        print(f"unknown command {command!r} -- expected one of: upload, download, cleanup")
        sys.exit(1)
