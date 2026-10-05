"""Training micro-batches drawn from several datasets at fixed weights, e.g. 50:50 FineWeb-Edu : Cosmopedia-v2.

One ShardedDataLoader per dataset does all the shard reading, per-rank offsets and resume for its own stream; this
class only decides which dataset each micro-batch comes from. The choice is deterministic, as in Megatron-LM's
blended datasets (`build_blending_indices`): each pick goes to the dataset furthest below its target share so far,
so the running mix never drifts more than one micro-batch from the weights, every rank makes the same picks, and a
resume replays them exactly.
"""
from .shard_loader import ShardedDataLoader


class MixedDataLoader:

    def __init__(self, weights, split, block_size, batch_size, rank, world_size, device, device_type):
        assert weights and all(w > 0 for w in weights.values()), f"weights must be positive: {weights}"
        self.weights = dict(weights)  # e.g. {"fineweb_edu": 0.5, "cosmopedia_v2": 0.5}
        self.names = list(weights)  # fixed order: every rank sees the same list, so ties break the same way
        self.loaders = {name: ShardedDataLoader(name, split, block_size, batch_size, rank, world_size, device,
                                                device_type) for name in self.names}  # one sequential stream each
        total = sum(self.weights.values())
        self.probs = [self.weights[n] / total for n in self.names]  # weights normalized to sum to 1, in names order
        self.counts = {name: 0 for name in self.names}  # micro-batches taken from each dataset so far

    def reset(self):
        """Back to the start of every stream, as in ShardedDataLoader.reset (used for fixed evaluation batches)."""
        self.counts = {name: 0 for name in self.names}
        for loader in self.loaders.values():
            loader.reset()

    def _pick(self):
        """The dataset furthest below its target share so far (largest target − actual count)."""
        n = max(sum(self.counts.values()), 1)  # picks made so far; 1 on the first call so the errors aren't all 0
        errors = [p * n - self.counts[name] for name, p in zip(self.names, self.probs)]  # target minus actual
        return self.names[errors.index(max(errors))]  # most underrepresented; ties go to the earliest name

    def next_batch(self):
        """This rank's next (x, y) micro-batch, each (batch_size, block_size), from one dataset."""
        name = self._pick()
        self.counts[name] += 1
        return self.loaders[name].next_batch()

    def state_dict(self):
        """Everything needed to resume exactly where this loader is."""
        return {
            "weights": self.weights,
            "counts": dict(self.counts),  # enough to replay every future pick: _pick depends only on the counts
            "loaders": {name: loader.state_dict() for name, loader in self.loaders.items()},  # each stream's position
        }

    def fractions(self):
        """Share of micro-batches taken from each dataset so far (for logging)."""
        n = max(sum(self.counts.values()), 1)
        return {name: count / n for name, count in self.counts.items()}

    def load_state_dict(self, state):
        """Resume exactly where state_dict() left off. Refuse a state saved with different weights."""
        assert state["weights"] == self.weights, f"saved weights {state['weights']} differ from {self.weights}"
        self.counts = dict(state["counts"])
        for name, loader in self.loaders.items():
            loader.load_state_dict(state["loaders"][name])  # each dataset resumes its own stream
