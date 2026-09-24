import os

import numpy as np
import pytest
import torch

import curioslm.loaders.shard_loader as shard_loader_module
from curioslm.loaders.shard_loader import ShardedDataLoader


@pytest.fixture
def fake_data_dir(tmp_path, monkeypatch):
    """Points data_dir() at a temp directory for the duration of a test, and returns it."""
    monkeypatch.setattr(shard_loader_module, "data_dir", lambda name: str(tmp_path))
    return tmp_path


def write_shard(directory, split, index, tokens):
    np.save(os.path.join(directory, f"fineweb_{split}_{index:06d}.npy"), np.asarray(tokens, dtype=np.uint16))


def test_shard_discovery_filters_by_split_and_sorts_numerically(fake_data_dir):
    """Both splits' shards live in the same directory -- only this split's files should be
    picked up, and in correct numeric order regardless of filesystem listing order."""
    write_shard(fake_data_dir, "train", 3, np.arange(4))
    write_shard(fake_data_dir, "val", 0, np.arange(4))
    write_shard(fake_data_dir, "train", 1, np.arange(4))
    write_shard(fake_data_dir, "train", 2, np.arange(4))
    (fake_data_dir / ".prepare_checkpoint.json").write_text("{}")  # leftover checkpoint file

    loader = ShardedDataLoader("fineweb_edu", "train", block_size=2, batch_size=1, rank=0, world_size=1)

    names = [os.path.basename(p) for p in loader.shards]
    assert names == ["fineweb_train_000001.npy", "fineweb_train_000002.npy", "fineweb_train_000003.npy"]


def test_reset_sets_shard_zero_and_rank_offset_position(fake_data_dir):
    write_shard(fake_data_dir, "train", 1, np.arange(100, 140))

    loader_rank0 = ShardedDataLoader("fineweb_edu", "train", block_size=4, batch_size=2, rank=0, world_size=3)
    loader_rank2 = ShardedDataLoader("fineweb_edu", "train", block_size=4, batch_size=2, rank=2, world_size=3)

    assert loader_rank0.current_shard_index == 0
    assert loader_rank0.current_position == 0            # rank 0 * B*T
    assert loader_rank2.current_position == 2 * 2 * 4     # rank 2 * B*T
    assert np.array_equal(loader_rank0.tokens, np.arange(100, 140))


def test_load_shard_returns_the_correct_shard_by_index(fake_data_dir):
    write_shard(fake_data_dir, "train", 1, [1, 2, 3])
    write_shard(fake_data_dir, "train", 2, [4, 5, 6])
    loader = ShardedDataLoader("fineweb_edu", "train", block_size=1, batch_size=1, rank=0, world_size=1)

    assert np.array_equal(loader._load_shard(0), [1, 2, 3])
    assert np.array_equal(loader._load_shard(1), [4, 5, 6])


def test_next_batch_shape_dtype_and_y_is_x_shifted_by_one(fake_data_dir):
    write_shard(fake_data_dir, "train", 1, np.arange(0, 200))
    loader = ShardedDataLoader("fineweb_edu", "train", block_size=4, batch_size=2, rank=0, world_size=1)

    x, y = loader.next_batch()

    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert x.shape == (2, 4) and y.shape == (2, 4)
    assert torch.equal(x.flatten(), torch.arange(0, 8))
    assert torch.equal(y.flatten(), torch.arange(1, 9))  # y is x shifted by exactly one position


def test_next_batch_advances_position_by_world_size_batch_block(fake_data_dir):
    write_shard(fake_data_dir, "train", 1, np.arange(0, 1000))  # long enough, no rollover in play
    world_size = 4
    loader = ShardedDataLoader("fineweb_edu", "train", block_size=8, batch_size=2, rank=0, world_size=world_size)

    before = loader.current_position
    loader.next_batch()
    after = loader.current_position

    assert after - before == world_size * 2 * 8  # world_size * batch_size * block_size


def test_ranks_never_read_overlapping_token_values_at_the_same_step(fake_data_dir):
    write_shard(fake_data_dir, "train", 1, np.arange(0, 400))
    write_shard(fake_data_dir, "train", 2, np.arange(1000, 1400))
    world_size = 3
    loaders = [
        ShardedDataLoader("fineweb_edu", "train", block_size=4, batch_size=2, rank=r, world_size=world_size)
        for r in range(world_size)
    ]

    for _ in range(15):  # enough steps to cross a shard rollover
        seen_this_step = []
        for loader in loaders:
            x, _ = loader.next_batch()
            seen_this_step.append(set(x.flatten().tolist()))
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert not (seen_this_step[i] & seen_this_step[j])


def test_shard_rollover_advances_index_and_resets_position(fake_data_dir):
    # shard sized so exactly one next_batch() call exhausts it (forces an immediate rollover)
    write_shard(fake_data_dir, "train", 1, np.arange(0, 9))     # block*batch+1 = 9, nothing left after
    write_shard(fake_data_dir, "train", 2, np.arange(1000, 1009))
    loader = ShardedDataLoader("fineweb_edu", "train", block_size=4, batch_size=2, rank=0, world_size=1)

    loader.next_batch()

    assert loader.current_shard_index == 1
    assert loader.current_position == 0  # rank 0's starting offset in the new shard
    assert np.array_equal(loader.tokens, np.arange(1000, 1009))


def test_shard_index_wraps_around_after_last_shard(fake_data_dir):
    write_shard(fake_data_dir, "train", 1, np.arange(0, 9))
    write_shard(fake_data_dir, "train", 2, np.arange(1000, 1009))
    loader = ShardedDataLoader("fineweb_edu", "train", block_size=4, batch_size=2, rank=0, world_size=1)

    loader.next_batch()  # shard 0 -> 1
    assert loader.current_shard_index == 1
    loader.next_batch()  # shard 1 -> wraps back to 0
    assert loader.current_shard_index == 0
    assert np.array_equal(loader.tokens, np.arange(0, 9))  # really did reload shard 0, not just the index


def test_no_duplication_across_ranks_within_a_single_shard(fake_data_dir):
    """
    No-duplication property in normal operation (no rollover in play): shard sized well
    beyond what a few steps could consume, so this isolates "do ranks ever double-read
    the same tokens" from rollover/epoch-boundary behavior (covered by the rollover tests
    below -- see their comments for why small shards can make ranks drift onto different
    shards independently, which is expected, not a bug, and irrelevant at real shard sizes).
    """
    world_size, block_size, batch_size = 2, 4, 2
    steps = 3
    max_tokens_needed = steps * world_size * batch_size * block_size + world_size * batch_size * block_size
    shard_size = max_tokens_needed * 4  # comfortable margin -- no rollover should trigger at all

    write_shard(fake_data_dir, "train", 1, np.arange(0, shard_size))
    loaders = [
        ShardedDataLoader("fineweb_edu", "train", block_size=block_size, batch_size=batch_size,
                           rank=r, world_size=world_size)
        for r in range(world_size)
    ]

    seen = []
    for _ in range(steps):
        for loader in loaders:
            x, _ = loader.next_batch()
            seen.extend(x.flatten().tolist())
            assert loader.current_shard_index == 0, "shouldn't have rolled over with this much margin"

    assert len(seen) == len(set(seen)), "some token value was read by more than one rank/call"


def test_split_val_and_train_use_disjoint_shard_files(fake_data_dir):
    write_shard(fake_data_dir, "train", 1, np.arange(0, 20))
    write_shard(fake_data_dir, "val", 0, np.arange(500, 520))

    train_loader = ShardedDataLoader("fineweb_edu", "train", block_size=4, batch_size=2, rank=0, world_size=1)
    val_loader = ShardedDataLoader("fineweb_edu", "val", block_size=4, batch_size=2, rank=0, world_size=1)

    assert np.array_equal(train_loader.tokens, np.arange(0, 20))
    assert np.array_equal(val_loader.tokens, np.arange(500, 520))


def test_missing_shards_raises_clear_error(fake_data_dir):
    with pytest.raises(AssertionError, match="no shards found"):
        ShardedDataLoader("fineweb_edu", "train", block_size=4, batch_size=2, rank=0, world_size=1)
