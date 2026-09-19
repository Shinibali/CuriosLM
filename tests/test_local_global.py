import torch

from curioslm.config import GPTConfig
from curioslm.model import GPT, CausalSelfAttention


def test_is_global_layer_assignment_matches_ratio():
    """
    local_global_ratio=3 with 6 layers should make layers 0 and 3 global (index % ratio == 0),
    the rest local -- and only local layers should carry a window_mask buffer at all, since
    global layers use SDPA's is_causal=True path with no mask tensor.
    """
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
    """
    Directly check the mask content, not just that it constructs: for window=2, each query
    position should see itself and exactly 1 position back, and nothing before or after that.
    """
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
    """
    Full integration check on the thing that actually broke earlier: flex_attention has no
    backward support on CPU/MPS, which is why this uses SDPA + a dense mask instead. Confirm
    both a forward pass and a backward pass (loss.backward()) complete without error, and that
    a shorter T than block_size (the generation-time case) also works.
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
