"""
Loader of single long pages, for long-context fine-tuning (YaRN).

The sharded stream packs documents back to back, so a 4,096-token window mostly joins several short pages and the
model rarely has a reason to look far back. Here every sequence is one page: the first block_size + 1 tokens of a
page at least block_size tokens long, starting at the EOT that opens it (as every page starts in training), so no
sequence spans two documents. Pages are found in the same prepare.py shards the other loaders read; nothing new is
prepared or uploaded. Same interface as ShardedDataLoader: sequential, rank-partitioned, exactly resumable.
"""
import os

import numpy as np
import torch

from .dataset import EOT, data_dir


class LongPageLoader:
    """Sequential, rank-partitioned loader over a split's long pages. Each rank reads its own batch_size pages per
    call, all ranks marching through the same shard sequence in lockstep."""

    def __init__(self, dataset_name, split, block_size, batch_size, rank, world_size, device, device_type):
        assert split in {"train", "val"}
        self.block_size = block_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.device_type = device_type
        prefix = f"{dataset_name}_{split}_"  # exact prepare.py naming, as in ShardedDataLoader
        dataset_dir = data_dir(dataset_name)
        self.shards = sorted([os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir)
                              if filename.startswith(prefix) and filename.endswith(".npy")])
        assert len(self.shards) > 0, f"no shards found for split {split!r} in {dataset_dir}"
        self.reset()

    def _load_shard(self, shard_index):
        """(pages, block_size + 1) uint16: every page in the shard with at least block_size tokens, cut at that
        length plus one target. The shard's last page is skipped: it may continue into the next shard."""
        # e.g. tokens = [EOT a b EOT c d e f EOT g ...] with block_size = 4: only the page "c d e f" is long enough
        tokens = np.load(self.shards[shard_index])  # the whole shard, (~100M,) token ids
        eots = np.flatnonzero(tokens == EOT)  # (num_pages,) index of the EOT opening each page, e.g. [0, 3, 8]
        page_lengths = np.diff(eots) - 1  # (num_pages - 1,) tokens between consecutive EOTs, e.g. [2, 4]
        # eots[:-1] pairs each page's start with its length; the last page has no closing EOT here, so it's dropped
        starts = eots[:-1][page_lengths >= self.block_size]  # (num_long,) EOT index of each long page, e.g. [3]
        # broadcast (num_long, 1) + (block_size + 1,) -> (num_long, block_size + 1) indices:
        pages = tokens[starts[:, None] + np.arange(self.block_size + 1)]  # rows = pages, columns = EOT + block_size
        # one next_batch() call hands every rank batch_size rows, so a shard must hold at least world_size * batch_size of them; 
        assert len(pages) >= self.world_size * self.batch_size, \
            f"{self.shards[shard_index]} has {len(pages)} pages of {self.block_size}+ tokens, fewer than one step"
        return pages  # (num_long, block_size + 1) uint16

    def reset(self):
        """(Re)start at the first shard, at this rank's offset."""
        self.current_shard_index = 0
        self.pages = self._load_shard(self.current_shard_index)
        self.current_row = self.rank * self.batch_size

    def state_dict(self):
        """Where this loader is, rank-independent, so rank 0's copy can restore every rank."""
        return {"shard_index": self.current_shard_index, "row": self.current_row - self.rank * self.batch_size,
                "batch_size": self.batch_size, "block_size": self.block_size, "world_size": self.world_size}

    def load_state_dict(self, state):
        """Resume exactly where state_dict() left off, at this rank's own offset."""
        saved = (state["batch_size"], state["block_size"], state["world_size"])
        assert saved == (self.batch_size, self.block_size, self.world_size), \
            f"loader state (batch, block, world) {saved} differs from this run's"
        self.current_shard_index = state["shard_index"]
        self.pages = self._load_shard(self.current_shard_index)
        self.current_row = state["row"] + self.rank * self.batch_size

    def next_batch(self):
        """This rank's next (x, y) batch, each (batch_size, block_size); y is x shifted by one, within each page."""
        # same lockstep as ShardedDataLoader, counted in pages (rows) instead of tokens: rank r reads rows
        # [row + r*B, row + (r+1)*B), e.g. 4 ranks, B = 15: rank 0 rows 0-14, rank 1 15-29, ..., rank 3 45-59
        rows = torch.from_numpy(self.pages[self.current_row:self.current_row + self.batch_size].astype(np.int64))
        x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()  # (B, block_size) each
        self.current_row += self.world_size * self.batch_size  # every rank skips past all ranks' rows: next call 60-119
        if self.current_row + self.world_size * self.batch_size > len(self.pages):  # next call wouldn't fit
            self.current_shard_index = (self.current_shard_index + 1) % len(self.shards)
            self.pages = self._load_shard(self.current_shard_index)
            self.current_row = self.rank * self.batch_size
        if self.device_type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)