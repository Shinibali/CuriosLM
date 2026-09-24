"""
Tokenize HuggingFaceFW/fineweb-edu (sample-10BT) into fixed-size shards.

Adapted from Karpathy's build-nanogpt fineweb.py, with two deliberate deviations from the reference script
  - Shards are read back with np.load(), never np.memmap() -- .npy embeds its own dtype/shape, which is exactly what memmap can't parse (it would silently treat the header bytes as the first few "tokens"). This is only safe because every shard here is small enough (shard_size below) to load fully into RAM per shard. Multi-node data access: every node pulls the *full* shard set (not a disjoint subset per node) and partitions at the token level within it, matching build-nanogpt's own DataLoaderLite -- deliberately, after reconsidering a per-node shard-assignment scheme. This is a scale-calibrated choice, not a general one: at ~20GB tokenized (sample-10BT) and 2 nodes, N-way redundant downloads are cheap and the proven reference pattern carries less risk than a bespoke partitioning scheme. Revisit if either axis grows meaningfully -- a much larger corpus (per-node full copies stop being cheap) or many more nodes (N-way redundant pulls start stressing the source, i.e. the private HF repo).
  - Output goes in data/fineweb_edu/, matching this project's existing
    data/<name>/prepare.py layout (see data/shakespeare/prepare.py) rather than
    build-nanogpt's own directory convention.

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
sys.path.insert(0, PROJECT_ROOT)  # prepare.py lives outside the curioslm package (not on
                                   # sys.path by default when run as `python data/.../prepare.py`
                                   # -- verified empirically, see git history for the check)
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
    # Prepend EOT as a document separator -- every document in the source dataset is a separate web page, and without this the model would see the end of one page flow directly into the start of the next as if it were continuous text. encode_ordinary (not encode) is used so literal "<|endoftext|>" substrings inside a document's text are tokenized as plain text, not misread as the real special token.
    tokens = [EOT] + enc.encode_ordinary(doc["text"])
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token id doesn't fit in uint16"
    return tokens_np.astype(np.uint16)


def shard_tokens(token_arrays, shard_size, start_shard_index=0, start_docs_read=0):
    """Accumulate an iterable of uint16 token arrays into fixed-size shards.

    Every input token ends up in exactly one output shard, in original order, split only at shard_size boundaries -- no token is ever dropped or duplicated. 

    A token array bigger than shard_size correctly splits across multiple shards. Yields (shard_index, tokens_np, docs_read) triples. docs_read is a running count of documents pulled from token_arrays -- the *current* document may only be partially in this shard (its tail can spill into the next one), so docs_read is a safe, slightly conservative resume point: on restart, skipping this many documents on the source stream never re-includes anything already written to disk, at the cost of losing at most one document's worth of tokens -- whatever was only ever sitting in the in-progress buffer, never written, when a crash happened. start_shard_index/start_docs_read let a resumed run continue numbering from where a previous run left off.
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
    """None if starting fresh. Guards against resuming with a changed SHARD_SIZE, which
    would silently produce inconsistently-sized shards -- fail loudly instead."""
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
    # streaming=True: a non-streaming load_dataset(split="train") on a dataset this size has to
    # resolve the full parquet shard listing before anything can be read. Streaming reads shard-by-shard over
    # HTTP instead, starts yielding documents immediately, and never caches the raw ~30-50GB
    # text to local disk -- only the tokenized shards this script writes persist.
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