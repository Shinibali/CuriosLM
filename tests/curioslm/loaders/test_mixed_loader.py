"""MixedDataLoader: training batches drawn from two (or more) datasets at fixed weights, built on ShardedDataLoader.

Interface under test (everything behind it is the implementer's choice):
    loader = MixedDataLoader({"fineweb_edu": 0.5, "cosmopedia_v2": 0.5}, "train", block_size, batch_size,
                             rank, world_size, device, device_type)
    x, y = loader.next_batch()         # this rank's micro-batch, from one source, (batch_size, block_size) each
    loader.state_dict() / loader.load_state_dict(state)   # exact resume
"""
import os

import numpy as np
import pytest

import curioslm.loaders.shard_loader as shard_loader_module
from curioslm.loaders.mixed_loader import MixedDataLoader

B, T = 2, 8  # tiny batches: 16 tokens per micro-batch
STREAMS = {"alpha": 1000, "beta": 5000}  # each source's tokens live in its own id range, so batches are identifiable
SHARD_TOKENS = 1000


@pytest.fixture
def two_sources(tmp_path, monkeypatch):
    """Two fake datasets in their own folders, 3 train shards each; returns each source's full token stream."""
    monkeypatch.setattr(shard_loader_module, "data_dir", lambda name: str(tmp_path / name))
    streams = {}
    for name, base in STREAMS.items():
        (tmp_path / name).mkdir()
        tokens = (base + np.arange(3 * SHARD_TOKENS) % 4000).astype(np.uint16)
        for i in range(3):
            np.save(os.path.join(tmp_path, name, f"{name}_train_{i + 1:06d}.npy"),
                    tokens[i * SHARD_TOKENS:(i + 1) * SHARD_TOKENS])
        streams[name] = tokens
    return streams


def make(weights, rank=0, world_size=1):
    return MixedDataLoader(weights, "train", T, B, rank, world_size, device="cpu", device_type="cpu")


def source_of(x):
    """Which fake dataset a batch came from, by its token id range."""
    lo = int(x.min())
    return "beta" if lo >= STREAMS["beta"] else "alpha"


def test_batches_follow_the_weights_at_every_point(two_sources):
    """Deterministic blending: after every micro-batch, each source's count is within one of its target share
    (here 75:25), not just on average."""
    loader = make({"alpha": 0.75, "beta": 0.25})
    taken = {"alpha": 0, "beta": 0}
    for n in range(1, 401):
        taken[source_of(loader.next_batch()[0])] += 1
        assert abs(taken["alpha"] - 0.75 * n) <= 1, f"after {n} batches: {taken}"


def test_equal_weights_alternate(two_sources):
    """50:50 gives exactly one batch of each in every consecutive pair, so every optimizer step is an even mix."""
    loader = make({"alpha": 0.5, "beta": 0.5})
    picks = [source_of(loader.next_batch()[0]) for _ in range(40)]
    assert all({picks[i], picks[i + 1]} == {"alpha", "beta"} for i in range(0, 40, 2))


def test_each_source_reads_its_stream_in_order(two_sources):
    """Mixing must not skip or repeat data within a source: a source's k-th batch is the k-th consecutive slice of
    its stream (rank 0 of 1)."""
    loader = make({"alpha": 0.5, "beta": 0.5})
    seen = {name: 0 for name in STREAMS}
    for _ in range(60):
        x, y = loader.next_batch()
        name = source_of(x)
        k = seen[name]
        expected = two_sources[name][k * B * T:(k + 1) * B * T + 1].astype(np.int64)  # B*T inputs + 1 target
        assert np.array_equal(x.flatten().numpy(), expected[:-1]), f"{name} batch {k} out of order"
        assert np.array_equal(y.flatten().numpy(), expected[1:])
        seen[name] += 1


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_every_optimizer_step_is_an_exact_mix_on_any_gpu_count(two_sources, world_size):
    """A step is 8 global micro-batches (grad accum 8, split over the ranks); at 50:50 every step must hold exactly
    4 of each, however many GPUs share it."""
    ranks = [make({"alpha": 0.5, "beta": 0.5}, rank=r, world_size=world_size) for r in range(world_size)]
    for step in range(10):
        picks = [source_of(loader.next_batch()[0]) for _ in range(8 // world_size) for loader in ranks]
        assert picks.count("alpha") == 4, f"step {step} on {world_size} GPUs: {picks}"


@pytest.mark.parametrize("world_size", [2, 4])
def test_ranks_read_each_stream_in_order_without_gaps_or_repeats(two_sources, world_size):
    """Across all ranks, each source's micro-batches, taken in global order (call, then rank), are consecutive slices
    of its stream: nothing skipped, nothing read twice."""
    ranks = [make({"alpha": 0.5, "beta": 0.5}, rank=r, world_size=world_size) for r in range(world_size)]
    seen = {name: 0 for name in STREAMS}
    for _ in range(20):
        for loader in ranks:
            x, y = loader.next_batch()
            name = source_of(x)
            k = seen[name]
            expected = two_sources[name][k * B * T:(k + 1) * B * T + 1].astype(np.int64)
            assert np.array_equal(x.flatten().numpy(), expected[:-1]), f"{name} chunk {k} out of order"
            seen[name] += 1


def test_resume_is_exact(two_sources):
    """state_dict() mid-stream, restored into a fresh loader, gives exactly the batches the original would have."""
    loader = make({"alpha": 0.5, "beta": 0.5})
    for _ in range(37):
        loader.next_batch()
    state = loader.state_dict()
    expected = [loader.next_batch()[0] for _ in range(20)]
    fresh = make({"alpha": 0.5, "beta": 0.5})
    fresh.load_state_dict(state)
    for i, want in enumerate(expected):
        assert np.array_equal(fresh.next_batch()[0].numpy(), want.numpy()), f"batch {i} after resume differs"


def test_resume_rejects_a_different_gpu_count(two_sources):
    """A round reads world_size chunks, so resuming on another GPU count would change the data order: fail loudly."""
    state = make({"alpha": 0.5, "beta": 0.5}, rank=0, world_size=2).state_dict()
    with pytest.raises(AssertionError):
        make({"alpha": 0.5, "beta": 0.5}, rank=0, world_size=4).load_state_dict(state)


def test_resume_rejects_different_weights(two_sources):
    """Resuming with other weights would silently change the run: fail loudly."""
    state = make({"alpha": 0.5, "beta": 0.5}).state_dict()
    with pytest.raises(AssertionError):
        make({"alpha": 0.9, "beta": 0.1}).load_state_dict(state)
