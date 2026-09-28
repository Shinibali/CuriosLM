import torch

from curioslm.config import GPTConfig
from curioslm.model import GPT, CausalSelfAttention


def test_is_global_layer_assignment_matches_ratio():
    """ratio=3 over 6 layers makes layers 0 and 3 global; only local layers carry a window_mask,
    since global layers take SDPA's is_causal path with no mask tensor."""
    config = GPTConfig(
        n_embd=32, n_head=4, n_layer=6, block_size=16, vocab_size=65,
        attention_window=2, local_global_ratio=3,
    )
    model = GPT(config)

    expected_global = {0, 3}
    for i, block in enumerate(model.transformer.h):
        assert block.attn.is_global_layer == (i in expected_global), f"layer {i} global flag wrong"
        assert hasattr(block.attn, "window_mask") == (i not in expected_global), (
            f"layer {i} should{'not' if i in expected_global else ''} have a window_mask buffer"
        )


def test_attention_window_none_makes_every_layer_global():
    """attention_window=None (the default) means full attention everywhere -- no local layers at all."""
    config = GPTConfig(n_embd=32, n_head=4, n_layer=4, block_size=16, vocab_size=65, attention_window=None)
    model = GPT(config)

    for block in model.transformer.h:
        assert block.attn.is_global_layer is True
        assert not hasattr(block.attn, "window_mask")


def test_window_mask_matches_expected_causal_and_window_pattern():
    """Mask content, not just that it constructs: window=2 means each query sees itself and
    exactly one position back."""
    config = GPTConfig(
        n_embd=32, n_head=4, n_layer=1, block_size=6, vocab_size=65,
        attention_window=2, local_global_ratio=5,
    )
    attn = CausalSelfAttention(config, layer_idx=1)  # 1 % 5 != 0 -> local layer
    assert not attn.is_global_layer

    expected = torch.tensor([
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(attn.window_mask, expected)


def test_local_global_model_forward_and_backward():
    """Forward and backward both complete, and a T shorter than block_size (generation) works.

    The backward half matters because flex_attention has no CPU/MPS backward -- the reason this
    path uses SDPA with a dense mask instead.
    """
    config = GPTConfig(
        n_embd=32, n_head=4, n_kv_head=2, n_layer=6, block_size=16, vocab_size=65,
        attention_window=2, local_global_ratio=3,
    )
    model = GPT(config)

    x = torch.randint(0, 65, (2, 10))
    y = torch.randint(0, 65, (2, 10))
    logits, loss = model(x, y)
    assert logits.shape == (2, 10, 65)

    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    x_short = torch.randint(0, 65, (1, 3))
    logits_short, _ = model(x_short)
    assert logits_short.shape == (1, 1, 65)


def test_flex_block_mask_gives_the_same_attention_as_the_dense_mask():
    """The GPU path (FlexAttention + block mask) and the Mac path (SDPA + dense mask) are built from the
    same local_causal rule and must agree. FlexAttention can't run backward off CUDA, so this checks the
    forward pass only, with gradients off."""
    from torch.nn.attention.flex_attention import flex_attention
    import torch.nn.functional as F

    config = GPTConfig(n_embd=64, n_head=4, n_kv_head=1, n_layer=2, block_size=64, vocab_size=65,
                       position_encoding="rope", attention_window=16, local_global_ratio=2)
    attn = GPT(config).transformer.h[1].attn
    assert not attn.is_global_layer

    T = 48  # shorter than block_size, as in generation
    with torch.no_grad():
        q, k, v = torch.randn(2, 4, T, 16), torch.randn(2, 1, T, 16), torch.randn(2, 1, T, 16)
        flex = flex_attention(q, k, v, block_mask=attn._flex_block_mask(T, "cpu"), enable_gqa=True)
        dense = F.scaled_dot_product_attention(q, k, v, attn_mask=attn.window_mask[:T, :T], enable_gqa=True)

    assert torch.allclose(flex, dense, atol=1e-5)


def test_flex_block_mask_is_built_once_per_length():
    """The mask depends only on positions, so it's cached per (T, device) rather than rebuilt every step."""
    config = GPTConfig(n_embd=32, n_head=4, n_layer=2, block_size=32, vocab_size=65,
                       attention_window=8, local_global_ratio=2)
    attn = GPT(config).transformer.h[1].attn

    assert attn._flex_block_mask(32, "cpu") is attn._flex_block_mask(32, "cpu")
    assert attn._flex_block_mask(16, "cpu") is not attn._flex_block_mask(32, "cpu")
