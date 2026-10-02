"""
Shared by every data/<name>/prepare.py: stream a Hugging Face dataset, tokenize it with GPT-2 BPE and write
fixed-size uint16 shards named <name>_<split>_<index>.npy (shard 0 = val, the rest = train). Crash-resumable.

Adapted from Karpathy's build-nanogpt fineweb.py. Shards are read back with np.load(), never np.memmap(): .npy
embeds its own dtype/shape header, which memmap would silently treat as the first few "tokens". Safe only because
each shard fits in RAM. Every node pulls the full shard set and partitions at the token level within it, matching
build-nanogpt's DataLoaderLite -- cheap at ~20GB and a couple of nodes; revisit for a much larger corpus or many
more nodes.
"""
import json
import multiprocessing as mp
import os
import threading

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

CHECKPOINT_NAME = ".prepare_checkpoint.json"

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256, GPT-2's <|endoftext|>


def tokenize(doc):
    # EOT separates documents, so one web page doesn't read as continuous text into the next.
    # encode_ordinary keeps a literal "<|endoftext|>" in the text as plain tokens.
    tokens = [EOT] + enc.encode_ordinary(doc["text"])
    tokens_np = np.array(tokens)  # int64 first, so an out-of-range id is caught below, not wrapped
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token id doesn't fit in uint16"
    return tokens_np.astype(np.uint16)  # vocab 50,257 < 2**16: half the disk of int32


def shard_tokens(token_arrays, shard_size, start_shard_index=0, start_docs_read=0):
    """Accumulate an iterable of uint16 token arrays into fixed-size shards.

    Every token lands in exactly one shard, in order, split only at shard_size boundaries; an
    array larger than shard_size splits across several. Yields (shard_index, tokens_np,
    docs_read). docs_read is a conservative resume point: the current document may be only
    partly in this shard, so skipping docs_read documents on restart never re-includes written
    data, at the cost of at most one document.
    """
    shard_index = start_shard_index
    buffer = np.empty((shard_size,), dtype=np.uint16)  # one shard's worth, reused across shards
    count = 0  # tokens filled in the current shard
    docs_read = start_docs_read

    for tokens in token_arrays:
        docs_read += 1
        while count + len(tokens) >= shard_size:  # this document fills (or overflows) the shard
            remainder = shard_size - count  # room left in the shard
            buffer[count:count + remainder] = tokens[:remainder]
            yield shard_index, buffer.copy(), docs_read  # copy: the buffer is reused for the next shard
            shard_index += 1
            tokens = tokens[remainder:]  # the overflow starts the next shard
            count = 0
        buffer[count:count + len(tokens)] = tokens  # what's left fits: append it
        count += len(tokens)

    if count > 0:  # stream ended mid-shard: emit the short final shard
        yield shard_index, buffer[:count].copy(), docs_read


def write_shard(out_dir, name, shard_index, tokens_np):
    split = "val" if shard_index == 0 else "train"  # first shard held out for validation
    np.save(os.path.join(out_dir, f"{name}_{split}_{shard_index:06d}.npy"), tokens_np)


def save_checkpoint(path, shard_size, next_shard_index, docs_read, seed=None):
    with open(path, "w") as f:
        json.dump({"shard_size": shard_size, "seed": seed, "next_shard_index": next_shard_index,
                   "docs_read": docs_read}, f)


def load_checkpoint(path, shard_size, seed=None):
    """None if starting fresh. Fails loudly if the shard size or shuffle seed changed since the checkpoint:
    resuming with different settings would produce inconsistent shards."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    assert state["shard_size"] == shard_size and state.get("seed") == seed, (
        f"checkpoint was made with shard_size={state['shard_size']}, seed={state.get('seed')}, but this run "
        f"uses shard_size={shard_size}, seed={seed}")
    return state


def prepare(name, out_dir, hf_source, hf_config, shard_size, *, max_shards=None, shuffle_seed=None,
            shuffle_buffer=10_000, approx_total_tokens=None):
    """Stream hf_source/hf_config's train split into shards in out_dir.

    max_shards: stop after this many full shards (for taking a slice of a larger corpus); never writes a short
        shard. Without it, the whole stream is sharded and the last shard may be short.
    shuffle_seed: shuffle the stream first (shard order + a document buffer), so a slice isn't just whatever the
        first files hold. The fixed seed keeps resume's skip() replaying the same order.
    approx_total_tokens: only sizes the progress bar when max_shards isn't given.
    """
    # streaming=True: a non-streaming load at this size resolves the whole parquet listing first.
    # Streaming yields documents immediately and never caches the raw text to disk.
    ds = load_dataset(hf_source, name=hf_config, split="train", streaming=True)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)

    checkpoint_path = os.path.join(out_dir, CHECKPOINT_NAME)
    state = load_checkpoint(checkpoint_path, shard_size, shuffle_seed)
    start_shard_index, start_docs_read = 0, 0
    if state is not None:
        print(f"resuming: skipping {state['docs_read']:,} already-processed documents, "
              f"starting at shard {state['next_shard_index']}")
        ds = ds.skip(state["docs_read"])  # may drop the one document split across the last shard boundary
        start_shard_index, start_docs_read = state["next_shard_index"], state["docs_read"]
    if max_shards is not None and start_shard_index >= max_shards:
        print("all shards already written")
        return

    # imap's feeder thread reads the stream as fast as it can, with no back-pressure. Stopping early must also stop
    # that thread, or shutting the pool down waits for it to download the rest of the corpus.
    stop = threading.Event()

    def documents():
        for doc in ds:
            if stop.is_set():  # main loop finished early: stop pulling from the stream
                return
            yield doc

    total = max_shards if max_shards is not None else (approx_total_tokens or 0) // shard_size or None  # progress bar
    nprocs = max(1, (os.cpu_count() or 2) // 2)  # half the cores tokenize; the rest stream, shard and write
    complete = max_shards is None  # an uncapped run is complete once the stream is exhausted
    with mp.Pool(nprocs) as pool:
        try:
            token_stream = pool.imap(tokenize, documents(), chunksize=16)  # imap keeps document order
            progress = tqdm(desc="sharding", unit="shard", initial=start_shard_index, total=total)
            for shard_index, shard, docs_read in shard_tokens(token_stream, shard_size, start_shard_index,
                                                              start_docs_read):
                if max_shards is not None and len(shard) < shard_size:  # stream ended before the cap
                    break
                write_shard(out_dir, name, shard_index, shard)
                save_checkpoint(checkpoint_path, shard_size, shard_index + 1, docs_read, shuffle_seed)  # after write
                progress.update(1)
                if max_shards is not None and shard_index + 1 >= max_shards:
                    complete = True
                    break
        finally:
            stop.set()

    if complete and os.path.exists(checkpoint_path):  # don't leave a stale checkpoint around
        os.remove(checkpoint_path)
