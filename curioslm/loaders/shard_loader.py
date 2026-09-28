"""
DDP-aware sequential shard-cycling data loader for the large multi-shard runs, following
Karpathy's build-nanogpt DataLoaderLite.

Unlike dataset.py's make_get_batch (random-sampling, stateless per call), this is sequential
and stateful: every rank marches through the *same* shard sequence in lockstep, each reading
a distinct non-overlapping window of the currently active shard.
"""
import os

import numpy as np
import torch

from .dataset import data_dir


class ShardedDataLoader:
    """Sequential, rank-partitioned loader over a split's tokenized .npy shards. rank/world_size
    are cluster-global, not per-node -- e.g. ranks 0-7 on node A, 8-15 on node B, world_size=16."""

    def __init__(self, dataset_name, split, block_size, batch_size, rank, world_size, device, device_type):
        assert split in {"train", "val"}
        self.block_size = block_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.device_type = device_type

        dataset_dir = data_dir(dataset_name)
        # Exact prepare.py naming, not a substring match: `split in filename` once let a stray
        # "fineweb_val_000000.npy" in as a second val shard.
        prefix = f"{dataset_name}_{split}_"
        self.shards = sorted([os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir)
                              if filename.startswith(prefix) and filename.endswith(".npy")])

        assert len(self.shards) > 0, f"no shards found for split {split!r} in {data_dir(dataset_name)}"

        self.reset()

    def reset(self):
        """(Re)start at the beginning of the shard sequence, at this rank's offset. Called from
        __init__, and callable again to start a fresh epoch on purpose."""
        self.current_shard_index = 0  
        self.tokens = self._load_shard(self.current_shard_index)               
        self.current_position = self.rank * self.batch_size * self.block_size  # R * B * T      

    def _load_shard(self, shard_index):
        """Load one shard's tokens fully into memory (np.load only -- never np.memmap."""

        return np.load(self.shards[shard_index])

    def state_dict(self):
        """Where this loader is, rank-independent, so rank 0's copy can restore every rank."""

        offset = self.rank * self.batch_size * self.block_size
        return {
            "shard_index": self.current_shard_index,
            "position": self.current_position - offset, # offset such that when resumed, it starts from tokens in the interrupted batch
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "world_size": self.world_size
        }

    def load_state_dict(self, state):
        """Resume exactly where state_dict() left off, at this rank's own offset."""

        assert self.batch_size == state["batch_size"] and self.world_size == state["world_size"] and self.block_size == state['block_size'], f"batch_size, block_size, world_size ({state["batch_size"], state["block_size"], state["world_size"]}) in ShardLoader `state` differs from parameters initialized in this run ({self.batch_size, self.block_size, self.world_size})"

        self.current_shard_index = state["shard_index"]
        self.tokens = self._load_shard(self.current_shard_index)           
        self.current_position = state["position"] + self.rank * self.batch_size * self.block_size  # R * B * T      

    def next_batch(self):
        """This rank's next (x, y) batch, each (batch_size, block_size); y is x shifted by one."""

        B_times_T = self.batch_size * self.block_size

        chunk = self.tokens[self.current_position: self.current_position + B_times_T + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.int64).view(self.batch_size, self.block_size)
        y = torch.tensor(chunk[1:], dtype=torch.int64).view(self.batch_size, self.block_size)

        self.current_position += self.world_size * B_times_T

        # Look ahead: if the *next* call wouldn't have a full window left, advance now rather
        # than running out mid-batch.
        if self.current_position + B_times_T * self.world_size + 1 > len(self.tokens):
            self.current_shard_index = (self.current_shard_index + 1) % len(self.shards)  # wrap to shard 0 after the last one
            self.tokens = self._load_shard(self.current_shard_index)
            self.current_position = self.rank * B_times_T  # same starting-offset formula as reset()

        if self.device_type == 'cuda':
            # pinned memory can be copied to the GPU asynchronously
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)

        return x,y
