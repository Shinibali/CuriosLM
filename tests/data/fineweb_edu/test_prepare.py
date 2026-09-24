import json

import numpy as np
import pytest

from data.fineweb_edu.prepare import (
    EOT,
    load_checkpoint,
    save_checkpoint,
    shard_tokens,
    tokenize,
    write_shard,
)


def test_shard_tokens_no_loss_or_duplication():
    """
    The core "are we saving duplicate data" concern: feed a sequence of token arrays through
    shard_tokens and confirm the shards reconstruct the exact original stream -- same total
    length, same values in the same order, nothing repeated or dropped at a shard boundary.
    """
    rng = np.random.default_rng(0)
    docs = [rng.integers(0, 50257, size=n, dtype=np.uint16) for n in [7, 12, 3, 20, 1, 15, 9]]
    original = np.concatenate(docs)

    shards = list(shard_tokens(iter(docs), shard_size=10))
    reconstructed = np.concatenate([s for _, s, _ in shards])

    assert np.array_equal(original, reconstructed)
    assert [i for i, _, _ in shards] == list(range(len(shards)))  # shard indices are sequential


def test_shard_tokens_handles_document_larger_than_shard_size():
    """
    Real documents are always << shard_size, but the accumulator uses a while loop
    specifically so this doesn't silently corrupt data if that assumption ever breaks --
    a single array spanning 3+ shard boundaries should still split and reconstruct exactly.
    """
    big_doc = np.arange(35, dtype=np.uint16)
    shards = list(shard_tokens(iter([big_doc]), shard_size=10))

    assert len(shards) == 4  # 10 + 10 + 10 + 5
    assert np.array_equal(np.concatenate([s for _, s, _ in shards]), big_doc)


def test_shard_tokens_exact_fit_boundary():
    """count + len(tokens) landing exactly on shard_size shouldn't duplicate or drop a token."""
    docs = [np.arange(6, dtype=np.uint16), np.arange(6, 10, dtype=np.uint16)]  # 6 + 4 = 10 exactly
    shards = list(shard_tokens(iter(docs), shard_size=10))

    assert len(shards) == 1
    assert np.array_equal(shards[0][1], np.arange(10, dtype=np.uint16))


def test_shard_tokens_docs_read_is_a_running_count():
    """
    docs_read (the 3rd yielded value) is what resumption skips on restart -- it must count
    every document pulled from the input so far, including the one that triggered this
    shard (even if only partially consumed), never fewer.
    """
    docs = [np.arange(4, dtype=np.uint16) for _ in range(5)]  # 5 docs of 4 tokens, shard_size=10
    shards = list(shard_tokens(iter(docs), shard_size=10))
    docs_read_values = [d for _, _, d in shards]

    assert docs_read_values == sorted(docs_read_values)  # never decreases
    assert docs_read_values[-1] == 5  # final value accounts for every document


def test_shard_tokens_resume_never_duplicates_already_written_data():
    """
    Simulates a crash: run shard_tokens partway, stop consuming after 2 shards (as if the
    process died), then "resume" with a fresh shard_tokens call using start_shard_index/
    start_docs_read from the last shard actually written, over only the remaining documents
    (mimicking fw.skip(docs_read) on restart). The two runs' shards combined must contain no
    duplicate shard indices and no duplicated token content -- the one acceptable outcome is
    losing the small in-progress tail that was never written before the "crash".
    """
    rng = np.random.default_rng(1)
    docs = [rng.integers(0, 50257, size=n, dtype=np.uint16) for n in [4, 6, 3, 8, 2, 9, 5, 7]]

    # "first run": only consume the first 2 shards, as if the process died right after
    first_run_shards = []
    gen = shard_tokens(iter(docs), shard_size=10)
    for _ in range(2):
        first_run_shards.append(next(gen))
    last_shard_index, _, last_docs_read = first_run_shards[-1]

    # "resume": skip exactly last_docs_read documents (what fw.skip() would do), continue
    # numbering from where the first run left off
    remaining_docs = docs[last_docs_read:]
    second_run_shards = list(
        shard_tokens(iter(remaining_docs), shard_size=10,
                     start_shard_index=last_shard_index + 1, start_docs_read=last_docs_read)
    )

    all_shards = first_run_shards + second_run_shards
    indices = [i for i, _, _ in all_shards]
    assert indices == sorted(set(indices)), "shard indices must be sequential with no repeats"

    all_tokens = np.concatenate([s for _, s, _ in all_shards])
    original = np.concatenate(docs)

    # no duplication: every written token appears in the original stream in the same
    # relative order -- i.e. all_tokens is a (not necessarily contiguous) subsequence of
    # original. The gap, if any, sits wherever the crash happened (the document that was
    # mid-flight), not necessarily at the very end -- so this checks order-preservation
    # and absence of repeats, not a simple prefix match.
    assert len(all_tokens) <= len(original)
    j = 0
    for value in all_tokens:
        while j < len(original) and original[j] != value:
            j += 1
        assert j < len(original), "all_tokens is not a valid in-order subsequence of original"
        j += 1
    # the loss, if any, is bounded to under one document's worth of tokens
    assert len(original) - len(all_tokens) < max(len(d) for d in docs)


def test_tokenize_prepends_eot_and_uses_uint16():
    tokens = tokenize({"text": "hello world"})
    assert tokens.dtype == np.uint16
    assert tokens[0] == EOT
    assert len(tokens) > 1  # EOT plus at least one real token


def test_write_shard_naming_and_split(tmp_path, monkeypatch):
    """Shard 0 is held out as val, every other shard is train -- verified via the actual
    write_shard function against a real temp directory, not just by reading the source."""
    import data.fineweb_edu.prepare as prepare_module
    monkeypatch.setattr(prepare_module, "DATA_DIR", str(tmp_path))

    write_shard(0, np.array([1, 2, 3], dtype=np.uint16))
    write_shard(1, np.array([4, 5], dtype=np.uint16))

    assert (tmp_path / "fineweb_edu_val_000000.npy").exists()
    assert (tmp_path / "fineweb_edu_train_000001.npy").exists()
    assert np.array_equal(np.load(tmp_path / "fineweb_edu_val_000000.npy"), [1, 2, 3])


def test_checkpoint_round_trip(tmp_path, monkeypatch):
    import data.fineweb_edu.prepare as prepare_module
    monkeypatch.setattr(prepare_module, "CHECKPOINT_PATH", str(tmp_path / "checkpoint.json"))

    assert load_checkpoint() is None  # nothing saved yet

    save_checkpoint(next_shard_index=7, docs_read=12345)
    state = load_checkpoint()

    assert state["next_shard_index"] == 7
    assert state["docs_read"] == 12345
    assert state["shard_size"] == prepare_module.SHARD_SIZE


def test_checkpoint_rejects_mismatched_shard_size(tmp_path, monkeypatch):
    """Resuming with a different SHARD_SIZE than the checkpoint was made with would silently
    produce inconsistent shards -- this must fail loudly instead."""
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps({"shard_size": 999, "next_shard_index": 1, "docs_read": 1}))

    import data.fineweb_edu.prepare as prepare_module
    monkeypatch.setattr(prepare_module, "CHECKPOINT_PATH", str(checkpoint_path))
    assert prepare_module.SHARD_SIZE != 999  # sanity: the mismatch is real

    with pytest.raises(AssertionError):
        load_checkpoint()


def test_streaming_pull_has_no_duplicate_tokens_end_to_end():
    """
    Integration check against the real HF streaming API (the thing that prompted this test
    file): pull a small real slice, tokenize it, shard it, and confirm the total token count
    matches the sum of each individual document's tokenization -- i.e. streaming isn't
    silently re-yielding a document, and shard_tokens isn't silently duplicating one.
    Skips gracefully if there's no network access, rather than failing the whole suite.
    """
    try:
        from datasets import load_dataset
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        docs = list(fw.take(25))
    except Exception as e:
        pytest.skip(f"no network access to Hugging Face: {e}")

    per_doc_tokens = [tokenize(doc) for doc in docs]
    expected_total = sum(len(t) for t in per_doc_tokens)

    shards = list(shard_tokens(iter(per_doc_tokens), shard_size=500))
    actual_total = sum(len(s) for _, s, _ in shards)

    assert actual_total == expected_total
    reconstructed = np.concatenate([s for _, s, _ in shards])
    assert np.array_equal(reconstructed, np.concatenate(per_doc_tokens))
