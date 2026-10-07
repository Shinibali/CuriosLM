import os

import numpy as np
import pytest

import curioslm.loaders.long_page_loader as long_page_module
from curioslm.loaders.long_page_loader import EOT, LongPageLoader


@pytest.fixture
def fake_data_dir(tmp_path, monkeypatch):
    """Points data_dir() at a temp directory for one test, and returns it."""
    monkeypatch.setattr(long_page_module, "data_dir", lambda name: str(tmp_path))
    return tmp_path


def write_shard(directory, split, index, pages):
    """A prepare.py-style shard: each page is EOT followed by its tokens; page p's tokens are all 100 + p."""
    tokens = np.concatenate([[EOT] + [100 + page] * length for page, length in enumerate(pages)])
    np.save(os.path.join(directory, f"fineweb_edu_{split}_{index:06d}.npy"), tokens.astype(np.uint16))


def make_loader(**kwargs):
    settings = dict(block_size=4, batch_size=1, rank=0, world_size=1, device="cpu", device_type="cpu")
    settings.update(kwargs)
    return LongPageLoader("fineweb_edu", "train", **settings)


def test_keeps_only_pages_long_enough_cut_from_their_eot(fake_data_dir):
    """Pages of 2 and 3 tokens are dropped; pages of 4+ give EOT plus their first 4 tokens. The shard's last page
    is skipped (it may continue into the next shard)."""
    write_shard(fake_data_dir, "train", 1, [2, 6, 3, 4, 9])
    loader = make_loader()
    assert loader.pages.tolist() == [[EOT, 101, 101, 101, 101], [EOT, 103, 103, 103, 103]]


def test_batch_is_one_page_shifted_by_one(fake_data_dir):
    """x starts at the page's EOT, y is x shifted by one, and nothing crosses into another page."""
    write_shard(fake_data_dir, "train", 1, [5, 6, 7, 1])
    x, y = make_loader().next_batch()
    assert x.tolist() == [[EOT, 100, 100, 100]]
    assert y.tolist() == [[100, 100, 100, 100]]


def test_ranks_read_distinct_pages_and_move_to_the_next_shard(fake_data_dir):
    """Two ranks take alternate pages; when a shard can't fill another step, both move to the next one."""
    write_shard(fake_data_dir, "train", 1, [4, 4, 4, 1])  # three usable pages: one step for two ranks
    write_shard(fake_data_dir, "train", 2, [5, 5, 1])
    first = [make_loader(rank=rank, world_size=2) for rank in range(2)]
    assert [loader.next_batch()[1][0, 0].item() for loader in first] == [100, 101]
    assert [loader.next_batch()[1][0, 0].item() for loader in first] == [100, 101]  # shard 2's pages 0 and 1


def test_resume_continues_exactly(fake_data_dir):
    """A loader restored from state_dict() yields the same next batch as the one it was saved from."""
    write_shard(fake_data_dir, "train", 1, [4, 5, 6, 7, 8, 1])
    loader = make_loader(batch_size=2)
    loader.next_batch()
    restored = make_loader(batch_size=2)
    restored.load_state_dict(loader.state_dict())
    assert restored.next_batch()[0].tolist() == loader.next_batch()[0].tolist()
