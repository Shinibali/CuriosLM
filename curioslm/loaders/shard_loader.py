"""
DDP-aware sequential shard-cycling data loader, for the large-scale FineWeb-Edu runs.

Distinct from dataset.py's make_get_batch, which is random-sampling and stateless per call
(fine for the small single-file Shakespeare smoke tests). This loader is fundamentally
different: SEQUENTIAL and STATEFUL. Every rank marches forward through the *same* shard
sequence in lockstep with every other rank -- rank r always reads a distinct, non-overlapping window of tokens within whatever shard is currently "active" for the whole cluster, and every rank advances together. This is Karpathy's build-nanogpt DataLoaderLite pattern.
"""
import os

import numpy as np
import torch

from .dataset import data_dir


class ShardedDataLoader:
    """
    Sequential, rank-partitioned loader over a split's tokenized .npy shards.

    rank/world_size identify THIS process among all GPUs across the whole cluster (not just
    within one node) -- e.g. rank 0-7 on node A, rank 8-15 on node B, world_size=16.
    """

    def __init__(self, dataset_name, split, block_size, batch_size, rank, world_size):
        assert split in {"train", "val"}
        self.block_size = block_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        dataset_dir = data_dir(dataset_name)
        self.shards = sorted([os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir) if split in filename])

        assert len(self.shards) > 0, f"no shards found for split {split!r} in {data_dir(dataset_name)}"

        self.reset()

    def reset(self):
        """(Re)start this loader at the beginning of the shard sequence, at this rank's
        starting offset. Called once from __init__; you may also want it callable again
        later (e.g. to start a fresh epoch on purpose), so keep it self-contained."""
        self.current_shard_index = 0  
        self.tokens = self._load_shard(self.current_shard_index)               
        self.current_position = self.rank * self.batch_size * self.block_size  # R * B * T      

    def _load_shard(self, shard_index):
        """Load one shard's tokens fully into memory (np.load only -- never np.memmap."""

        return np.load(self.shards[shard_index])


    def next_batch(self):
        """
        Return this rank's next (x, y) batch as torch tensors, each shape
        (batch_size, block_size) -- y is x shifted by one position.
        """

        B_times_T = self.batch_size * self.block_size

        chunk = self.tokens[self.current_position: self.current_position + B_times_T + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.int64).view(self.batch_size, self.block_size)
        y = torch.tensor(chunk[1:], dtype=torch.int64).view(self.batch_size, self.block_size)

        self.current_position += self.world_size * B_times_T

        # look-ahead check: is there still a full batch_size*block_size+1 window left in the
        # current shard for the *next* call? if not, move on before we run out mid-batch.
        if self.current_position + B_times_T * self.world_size + 1 > len(self.tokens):
            self.current_shard_index = (self.current_shard_index + 1) % len(self.shards)  # wrap to shard 0 after the last one
            self.tokens = self._load_shard(self.current_shard_index)
            self.current_position = self.rank * B_times_T  # same starting-offset formula as reset()

        return x,y
