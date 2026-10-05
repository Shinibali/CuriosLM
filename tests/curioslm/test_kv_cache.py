"""Incremental KV cache: generating with the cache must give the same next-token logits as recomputing the whole
sequence without it, for every rung of the ladder, and local layers must keep no more than their window.

Interface under test (everything behind it is the implementer's choice):
    cache = model.new_kv_cache(batch_size, device)   # empty cache for this model
    logits, _ = model(idx, kv_cache=cache)            # idx: only the NEW tokens (B, t_new); positions continue
                                                      # from what the cache holds; returns (B, 1, V) like no-cache
    cache.seq_lens()                                  # list[int]: positions each layer currently stores
"""
import pytest
import torch

from curioslm.config import GPTConfig
from curioslm.model import GPT

BLOCK, VOCAB, WINDOW = 32, 97, 8
BASE = dict(n_embd=32, n_head=4, n_layer=4, block_size=BLOCK, vocab_size=VOCAB, dropout=0.0)
RUNGS = {  # the ladder's architecture deltas, at toy size; a4's window is small so eviction gets exercised
    "a0": {},
    "a1": dict(position_encoding="rope"),
    "a2": dict(position_encoding="rope", n_kv_head=1),
    "a3": dict(position_encoding="rope", n_kv_head=1, mlp_type="geglu", norm_placement="sandwich"),
    "a4": dict(position_encoding="rope", n_kv_head=1, mlp_type="geglu", norm_placement="sandwich",
               attention_window=WINDOW, local_global_ratio=2),
}


def make(rung):
    torch.manual_seed(0)
    return GPT(GPTConfig(**BASE, **RUNGS[rung])).eval()


@torch.no_grad()
def full_recompute(model, seq):
    """Next-token logits from the whole sequence, no cache: the definition of correct."""
    logits, _ = model(seq)  # (B,1,V)
    return logits


@pytest.mark.parametrize("rung", RUNGS)
@torch.no_grad()
def test_prefill_matches_no_cache(rung):
    model = make(rung)
    seq = torch.randint(0, VOCAB, (2, 11))
    cached, _ = model(seq, kv_cache=model.new_kv_cache(2, "cpu"))
    assert torch.allclose(cached, full_recompute(model, seq), atol=1e-4)


@pytest.mark.parametrize("rung", RUNGS)
@torch.no_grad()
def test_one_token_at_a_time_matches_full_recompute(rung):
    """Prefill 5 tokens, then 20 single-token steps: 25 positions, well past a4's window of 8."""
    model = make(rung)
    seq = torch.randint(0, VOCAB, (2, 25))
    cache = model.new_kv_cache(2, "cpu")
    model(seq[:, :5], kv_cache=cache)
    for t in range(5, 25):
        cached, _ = model(seq[:, t:t + 1], kv_cache=cache)
        assert torch.allclose(cached, full_recompute(model, seq[:, :t + 1]), atol=1e-4), f"step {t}"


@pytest.mark.parametrize("rung", RUNGS)
@torch.no_grad()
def test_multi_token_chunks_match_full_recompute(rung):
    """Several new tokens at once on top of a non-empty cache: the attention mask must line up the new queries
    with their true positions among the cached keys."""
    model = make(rung)
    seq = torch.randint(0, VOCAB, (2, 24))
    cache = model.new_kv_cache(2, "cpu")
    for start, end in [(0, 6), (6, 13), (13, 24)]:
        cached, _ = model(seq[:, start:end], kv_cache=cache)
        assert torch.allclose(cached, full_recompute(model, seq[:, :end]), atol=1e-4), f"chunk ending at {end}"


@torch.no_grad()
def test_local_layers_keep_at_most_window_positions():
    """The memory payoff of local attention: after 25 positions, local layers hold <= WINDOW, global ones all 25."""
    model = make("a4")
    seq = torch.randint(0, VOCAB, (2, 25))
    cache = model.new_kv_cache(2, "cpu")
    model(seq[:, :5], kv_cache=cache)
    for t in range(5, 25):
        model(seq[:, t:t + 1], kv_cache=cache)
    for layer, n in enumerate(cache.seq_lens()):
        if model.config.is_global_layer(layer):
            assert n == 25, f"global layer {layer} should keep every position"
        else:
            assert n <= WINDOW, f"local layer {layer} keeps {n} positions, more than its window of {WINDOW}"


@pytest.mark.parametrize("rung", RUNGS)
def test_generate_with_cache_matches_without(rung):
    """Greedy generation (top_k=1) must produce the same tokens with and without the cache, including past
    block_size, where generation falls back to recomputing the last block_size tokens."""
    model = make(rung)
    prompt = torch.randint(0, VOCAB, (2, 5))
    cached = model.generate(prompt, BLOCK + 10, top_k=1, use_cache=True)
    uncached = model.generate(prompt, BLOCK + 10, top_k=1, use_cache=False)
    assert torch.equal(cached, uncached)
