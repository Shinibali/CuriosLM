import torch
import torch.nn.functional as F

from curioslm.config import GPTConfig
from curioslm.model import CausalSelfAttention


def test_enable_gqa_matches_manual_head_broadcast():
    """PyTorch's enable_gqa=True groups consecutive query heads per kv head, not some other
    order -- pinned against an explicit repeat_interleave broadcast rather than trusted."""
    torch.manual_seed(0)
    B, n_head, n_kv_head, T, hs = 2, 4, 2, 6, 8
    q = torch.randn(B, n_head, T, hs)
    k = torch.randn(B, n_kv_head, T, hs)
    v = torch.randn(B, n_kv_head, T, hs)

    out_native = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    group_size = n_head // n_kv_head
    k_expanded = k.repeat_interleave(group_size, dim=1)
    v_expanded = v.repeat_interleave(group_size, dim=1)
    out_manual = F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=True)

    assert torch.allclose(out_native, out_manual, atol=1e-5), (
        "enable_gqa's head grouping does not match the expected repeat_interleave order"
    )


def test_causal_self_attention_gqa_shapes():
    """n_kv_head < n_head: c_attn projects to the reduced size, not a plain MHA's 3*n_embd."""
    n_head, n_kv_head, n_embd, head_size = 4, 2, 32, 8
    config = GPTConfig(
        n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd,
        n_layer=1, block_size=16, vocab_size=65,
    )
    attn = CausalSelfAttention(config, layer_idx=0)

    expected_c_attn_out = n_embd + 2 * (n_kv_head * head_size)
    assert attn.c_attn.out_features == expected_c_attn_out, (
        f"expected c_attn to project to {expected_c_attn_out} (n_embd + 2*kv_size), "
        f"got {attn.c_attn.out_features}"
    )

    x = torch.randn(2, 10, n_embd)
    y = attn(x)
    assert y.shape == (2, 10, n_embd)


def test_causal_self_attention_mha_equivalent_when_kv_heads_match():
    """n_kv_head=None resolves to n_head, giving plain MHA: full 3*n_embd, enable_gqa off."""
    n_head, n_embd = 4, 32
    config = GPTConfig(n_head=n_head, n_kv_head=None, n_embd=n_embd, n_layer=1, block_size=16, vocab_size=65)
    attn = CausalSelfAttention(config, layer_idx=0)

    assert attn.enable_gqa is False
    assert attn.c_attn.out_features == 3 * n_embd

    x = torch.randn(2, 10, n_embd)
    y = attn(x)
    assert y.shape == (2, 10, n_embd)
