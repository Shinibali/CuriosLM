"""Shared by this study's scripts: where the step-320 page lives, the checkpoint that scores it, output paths."""
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(ANALYSIS)
for path in (ROOT, ANALYSIS):  # curioslm, and analysis/common
    if path not in sys.path:
        sys.path.insert(0, path)
from curioslm.loaders.shard_loader import ShardedDataLoader  # noqa: E402

DATA = os.path.join(HERE, "data")
FIGURES = os.path.join(HERE, "figures")
CONFIGS_JSON = os.path.join(DATA, "run_configs.json")      # written by 01_loss_spike.py
SPIKE_JSON = os.path.join(DATA, "spike.json")              # written by 01_loss_spike.py
LOCATION_JSON = os.path.join(DATA, "page_location.json")   # written by 02_unmangle.py
CLEAN_TXT = os.path.join(DATA, "page_clean.txt")           # written by 02_unmangle.py
CHANGES_CSV = os.path.join(DATA, "unmangle_changes.csv")   # written by 02_unmangle.py

RUN = "curioslm-ladder/kpqh0522"  # the 1.5B-token Gemma-style run: its batches are replayed, its checkpoint scores them
CKPT = os.path.join(ROOT, "out-ladder-fineweb_edu-a4", "ckpt.pt")
WORD = re.compile(r"[^\W_]+")  # a word: letters and digits, so "20f" is one word


def logged_batch(run=RUN):
    """The micro-batch whose loss `run` logged at the spike step, found by replaying GPU 0's training loader
    with the run's config: (shard path, token offset in it, tokens of shape (batch, block_size + 1))."""
    with open(SPIKE_JSON) as f:
        step = json.load(f)["step"]
    with open(CONFIGS_JSON) as f:
        cfg = json.load(f)[run]
    world = cfg["world_size"]
    per_gpu = cfg["gradient_accumulation_steps"] // world  # grad accum is global; train.py divides it by world size
    loader = ShardedDataLoader(cfg["dataset"], "train", cfg["block_size"], cfg["batch_size"], rank=0,
                               world_size=world, device="cpu", device_type="cpu")
    # train.py fetches one batch before the loop, then one per micro-step, and logs the last micro-step's loss:
    # step s's logged batch is fetch number s * per_gpu + per_gpu - 1, counting from 0.
    for _ in range(step * per_gpu + per_gpu - 1):  # every fetch before the logged one
        loader.next_batch()
    shard, offset = loader.shards[loader.current_shard_index], loader.current_position  # where the next fetch reads
    # The loader numbers the shards it finds locally; the replay is only right if none before this one is missing.
    expected = f"{cfg['dataset']}_train_{loader.current_shard_index + 1:06d}.npy"  # train shards start at 000001
    assert os.path.basename(shard) == expected, f"replay needs every train shard up to {expected} locally"
    x, y = loader.next_batch()
    return shard, offset, torch.cat([x, y[:, -1:]], dim=1)  # y is x shifted by one: its last column completes each row


def load_sequence():
    """The page's tokens exactly as the loader served them (block_size inputs plus the last target), from the
    location 02_unmangle.py recorded."""
    with open(LOCATION_JSON) as f:
        loc = json.load(f)
    start = loc["offset"] + loc["sequence"] * loc["block_size"]  # rows are consecutive block_size slices
    tokens = np.load(os.path.join(ROOT, loc["shard"]), mmap_mode="r")  # memory-mapped: reads only the slice
    return tokens[start:start + loc["block_size"] + 1].astype(np.int64).tolist()  # +1: the last target
