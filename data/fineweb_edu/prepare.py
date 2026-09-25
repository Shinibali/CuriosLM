"""
Tokenize HuggingFaceFW/fineweb-edu (sample-10BT) into fixed-size shards.

Adapted from Karpathy's build-nanogpt fineweb.py, with two deviations:
  - Shards are read back with np.load(), never np.memmap(): .npy embeds its own dtype/shape
    header, which memmap would silently treat as the first few "tokens". Safe only because each
    shard fits in RAM. Every node pulls the full shard set and partitions at the token level
    within it, matching build-nanogpt's DataLoaderLite -- cheap at ~20GB and a couple of nodes;
    revisit for a much larger corpus or many more nodes.
  - Output goes in data/fineweb_edu/, matching this project's data/<name>/prepare.py layout.

Run: python data/fineweb_edu/prepare.py
"""
import json
import os
import sys
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DATA_DIR))  # data/fineweb_edu -> data -> root
sys.path.insert(0, PROJECT_ROOT)  # this script lives outside the curioslm package, so the root
                                   # isn't on sys.path when it runs as a plain script
from curioslm.config import DataConfig  # noqa: E402 -- must follow the sys.path fix above

_data_cfg = DataConfig()
DATASET_NAME = _data_cfg.dataset  # single source of truth
SHARD_SIZE = _data_cfg.shard_size  # NOT independent of world_size*batch_size*block_size 

HF_SOURCE_DATASET = "HuggingFaceFW/fineweb-edu"  # prep-only, nothing downstream reads this
REMOTE_NAME = "sample-10BT"                      # prep-only, nothing downstream reads this
APPROX_TOTAL_TOKENS = 10_000_000_000  # FineWeb-Edu's own dataset card figure for sample-10BT 
CHECKPOINT_PATH = os.path.join(DATA_DIR, ".prepare_checkpoint.json")

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256, GPT-2's <|endoftext|>


def tokenize(doc):
    # EOT separates documents, so one web page doesn't read as continuous text into the next.
    # encode_ordinary keeps a literal "<|endoftext|>" in the text as plain tokens.
    tokens = [EOT] + enc.encode_ordinary(doc["text"])
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token id doesn't fit in uint16"
    return tokens_np.astype(np.uint16)


def shard_tokens(token_arrays, shard_size, start_shard_index=0, start_docs_read=0):
    """Accumulate an iterable of uint16 token arrays into fixed-size shards.

    Every token lands in exactly one shard, in order, split only at shard_size boundaries; an
    array larger than shard_size splits across several. Yields (shard_index, tokens_np,
    docs_read). docs_read is a conservative resume point: the current document may be only
    partly in this shard, so skipping docs_read documents on restart never re-includes written
    data, at the cost of at most one document.
    """
    shard_index = start_shard_index
    buffer = np.empty((shard_size,), dtype=np.uint16)
    count = 0
    docs_read = start_docs_read

    for tokens in token_arrays:
        docs_read += 1
        while count + len(tokens) >= shard_size:
            remainder = shard_size - count
            buffer[count:count + remainder] = tokens[:remainder]
            yield shard_index, buffer.copy(), docs_read
            shard_index += 1
            tokens = tokens[remainder:]
            count = 0
        buffer[count:count + len(tokens)] = tokens
        count += len(tokens)

    if count > 0:
        yield shard_index, buffer[:count].copy(), docs_read


def write_shard(shard_index, tokens_np):
    split = "val" if shard_index == 0 else "train"  # first shard held out for validation
    path = os.path.join(DATA_DIR, f"{DATASET_NAME}_{split}_{shard_index:06d}.npy")
    np.save(path, tokens_np)


def save_checkpoint(next_shard_index, docs_read):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"shard_size": SHARD_SIZE, "next_shard_index": next_shard_index, "docs_read": docs_read}, f)


def load_checkpoint():
    """None if starting fresh. Fails loudly if SHARD_SIZE changed since the checkpoint."""
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    with open(CHECKPOINT_PATH) as f:
        state = json.load(f)
    assert state["shard_size"] == SHARD_SIZE, (
        f"checkpoint was made with shard_size={state['shard_size']}, but SHARD_SIZE is now "
        f"{SHARD_SIZE} -- resuming with a different shard size would produce inconsistent shards"
    )
    return state


def main():
    # streaming=True: a non-streaming load at this size resolves the whole parquet listing first.
    # Streaming yields documents immediately and never caches the raw ~30-50GB text to disk.
    fw = load_dataset(HF_SOURCE_DATASET, name=REMOTE_NAME, split="train", streaming=True)

    checkpoint = load_checkpoint()
    if checkpoint is not None:
        print(f"resuming: skipping {checkpoint['docs_read']:,} already-processed documents, "
              f"starting at shard {checkpoint['next_shard_index']}")
        fw = fw.skip(checkpoint["docs_read"])
        start_shard_index = checkpoint["next_shard_index"]
        start_docs_read = checkpoint["docs_read"]
    else:
        start_shard_index = 0
        start_docs_read = 0

    nprocs = max(1, (os.cpu_count() or 2) // 2)
    total_shards = APPROX_TOTAL_TOKENS // SHARD_SIZE  # approximate -- see the constant's comment
    with mp.Pool(nprocs) as pool:
        token_stream = pool.imap(tokenize, fw, chunksize=16)
        progress = tqdm(desc="sharding", unit="shard", initial=start_shard_index, total=total_shards)
        for shard_index, shard, docs_read in shard_tokens(token_stream, SHARD_SIZE, start_shard_index, start_docs_read):
            write_shard(shard_index, shard)
            save_checkpoint(shard_index + 1, docs_read)
            progress.update(1)

    if os.path.exists(CHECKPOINT_PATH):  # ran to completion -- don't leave a stale checkpoint around
        os.remove(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()