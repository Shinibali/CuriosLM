"""
Upload/download tokenized shards to/from a private Hugging Face Hub dataset repo -- staging
between where shards get tokenized (prepare.py, CPU-only, run once) and wherever GPU training
runs. Every node downloads the full shard set, not a disjoint subset; see prepare.py.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

from .dataset import data_dir

# Transfers are network I/O-bound, not GIL-bound, so a thread pool gives real speedup.
MAX_WORKERS = 8


def upload_shards(dataset_name, repo_id, max_workers=MAX_WORKERS):
    """Push every .npy shard in data/<dataset_name>/ to the private HF repo, creating the repo
    first if needed. Uploads up to max_workers shards concurrently."""
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    directory = data_dir(dataset_name)
    shard_files = sorted(f for f in os.listdir(directory) if f.endswith(".npy"))
    assert shard_files, f"no .npy shards found in {directory} to upload"

    def _upload_one(filename):
        api.upload_file(
            path_or_fileobj=os.path.join(directory, filename),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
        return filename

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_upload_one, filename) for filename in shard_files]
        for future in as_completed(futures):
            print(f"uploaded {future.result()}")

    return shard_files


def download_shards(local_dir, repo_id, max_workers=MAX_WORKERS):
    """Pull every .npy shard from the private HF repo into local_dir, up to max_workers at a
    time. The destination is explicit so a fresh directory can genuinely test a from-scratch
    node. Skips files that already exist locally -- existence only, no size check, so delete a
    suspected-truncated copy by hand first."""
    os.makedirs(local_dir, exist_ok=True)

    shard_files = sorted(f for f in list_repo_files(repo_id, repo_type="dataset") if f.endswith(".npy"))
    assert shard_files, f"no .npy shards found in {repo_id}"

    to_download = [f for f in shard_files if not os.path.exists(os.path.join(local_dir, f))]
    skipped = [f for f in shard_files if f not in to_download]

    def _download_one(filename):
        hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=local_dir)
        return filename

    downloaded = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_download_one, filename) for filename in to_download]
        for future in as_completed(futures):
            filename = future.result()
            print(f"downloaded {filename}")
            downloaded.append(filename)

    print(f"downloaded {len(downloaded)} shard(s), skipped {len(skipped)} already present")
    return shard_files


def cleanup_local_shards(dataset_name, repo_id):
    """Delete local shards, but only ones confirmed on the remote repo by matching filename and
    size; anything unconfirmed is left untouched. Deliberately a separate step from
    upload_shards, since a mistaken deletion costs a full prepare.py re-run."""
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
        upload_shards(cfg.dataset, cfg.hf_dataset_repo_id)
    elif command == "download":
        download_shards(data_dir(cfg.dataset), cfg.hf_dataset_repo_id)
    elif command == "cleanup":
        cleanup_local_shards(cfg.dataset, cfg.hf_dataset_repo_id)
    else:
        print(f"unknown command {command!r} -- expected one of: upload, download, cleanup")
        sys.exit(1)
