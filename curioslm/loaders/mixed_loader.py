"""Training micro-batches drawn from several datasets at fixed weights, e.g. 50:50 FineWeb-Edu : Cosmopedia-v2.

The blend is decided over the *global* sequence of micro-batches — one per rank per call, numbered call × world_size
+ rank — with Megatron-LM's deterministic rule (`build_blending_indices`): each global micro-batch goes to the dataset
furthest below its target share so far. So the mix is exact over every window of global micro-batches, and in
particular every optimizer step is the right mix on any GPU count (50:50 alternates F, C, F, C, … across ranks and
micro-steps). Ranks may read different datasets on the same call.

Each rank keeps a full-sequence view of every dataset (a rank-0, world-size-1 ShardedDataLoader on CPU), advances it
by every chunk the round assigned to that dataset, and keeps only its own: each dataset is consumed in order, with
nothing skipped or read twice across ranks, and the state is identical on every rank, so a resume replays exactly.
"""
from .shard_loader import ShardedDataLoader


class MixedDataLoader:

    def __init__(self, weights, split, block_size, batch_size, rank, world_size, device, device_type):
        assert weights and all(w > 0 for w in weights.values()), f"weights must be positive: {weights}"
        self.weights = dict(weights)  # e.g. {"fineweb_edu": 0.5, "cosmopedia_v2": 0.5}
        self.names = list(weights)  # fixed order: every rank sees the same list, so ties break the same way
        self.rank, self.world_size = rank, world_size
        self.device, self.device_type = device, device_type
        # full-sequence view of each dataset; batches stay on CPU until this rank's own one is chosen
        self.loaders = {name: ShardedDataLoader(name, split, block_size, batch_size, rank=0, world_size=1,
                                                device="cpu", device_type="cpu") for name in self.names}
        total = sum(self.weights.values())
        self.probs = [self.weights[n] / total for n in self.names]  # weights normalized to sum to 1, in names order
        self.counts = {name: 0 for name in self.names}  # global micro-batches assigned to each dataset so far

    def reset(self):
        """Back to the start of every stream, as in ShardedDataLoader.reset (used for fixed evaluation batches)."""
        self.counts = {name: 0 for name in self.names}
        for loader in self.loaders.values():
            loader.reset()

    def _pick(self):
        """The dataset furthest below its target share so far (largest target − actual count); counts it."""
        n = max(sum(self.counts.values()), 1)  # global picks so far; 1 on the first call so the errors aren't all 0
        errors = [p * n - self.counts[name] for name, p in zip(self.names, self.probs)]  # target minus actual
        name = self.names[errors.index(max(errors))]  # most underrepresented; ties go to the earliest name
        self.counts[name] += 1
        return name

    def next_batch(self):
        """This rank's next (x, y) micro-batch, each (batch_size, block_size). Every rank runs the same round:
        assign the next world_size global micro-batches, read them in order, keep the one at this rank's position."""
        mine = None
        for position in range(self.world_size):  # global micro-batches call × world_size + position
            batch = self.loaders[self._pick()].next_batch()  # advance that dataset's stream by one chunk
            if position == self.rank:
                mine = batch
        x, y = mine
        if self.device_type == 'cuda':  # pinned memory copies to the GPU asynchronously
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def state_dict(self):
        """Everything needed to resume exactly where this loader is (identical on every rank)."""
        return {
            "weights": self.weights,
            "world_size": self.world_size,  # a round reads world_size chunks: resuming on another count shifts data
            "counts": dict(self.counts),  # enough to replay every future pick: _pick depends only on the counts
            "loaders": {name: loader.state_dict() for name, loader in self.loaders.items()},  # each stream's position
        }

    def fractions(self):
        """Share of global micro-batches assigned to each dataset so far (for logging)."""
        n = max(sum(self.counts.values()), 1)
        return {name: count / n for name, count in self.counts.items()}

    def load_state_dict(self, state):
        """Resume exactly where state_dict() left off. Refuse a state saved with different weights or GPU count."""
        assert state["weights"] == self.weights, f"saved weights {state['weights']} differ from {self.weights}"
        assert state["world_size"] == self.world_size, (
            f"saved on {state['world_size']} ranks, resuming on {self.world_size}: the data order would change")
        self.counts = dict(state["counts"])
        for name, loader in self.loaders.items():
            loader.load_state_dict(state["loaders"][name])  # each dataset resumes its own stream
