import warnings

import torch

from curioslm.config import GPTConfig
from curioslm.model import GPT, Block, GeGLU, MLP, RMSNormFP32


def test_geglu_matches_mlp_param_count():
    """The 2/3*4x hidden-dim convention keeps GeGLU (3 matrices) within ~1% of plain MLP
    (2 matrices) at the same n_embd, so swapping one for the other is size-neutral."""
    config = GPTConfig(n_embd=128, n_head=4, n_kv_head=2, n_layer=1, block_size=16, vocab_size=65)
    geglu_params = sum(p.numel() for p in GeGLU(config).parameters())
    mlp_params = sum(p.numel() for p in MLP(config).parameters())
    assert abs(geglu_params - mlp_params) / mlp_params < 0.01, (
        f"GeGLU ({geglu_params:,}) and MLP ({mlp_params:,}) params differ by more than 1%"
    )


def test_rmsnorm_fp32_computes_in_fp32_and_preserves_input_dtype():
    """RMSNormFP32 avoids nn.RMSNorm's dtype-mismatch fallback under autocast: warning-free
    on bf16 input, and hands back the dtype it was given."""
    norm = RMSNormFP32(64)
    x = torch.randn(2, 8, 64, dtype=torch.bfloat16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = norm(x)

    assert len(caught) == 0, f"expected no warnings, got: {[str(w.message) for w in caught]}"
    assert out.dtype == torch.bfloat16


def test_block_sandwich_norm_has_four_norms_pre_has_two():
    """Sandwich norm wraps both sub-layers with pre- and post-norm (4 modules); plain pre-norm
    normalizes only each sub-layer's input (2)."""
    base_kwargs = dict(n_embd=128, n_head=4, n_kv_head=2, n_layer=1, block_size=16, vocab_size=65)

    sandwich_block = Block(GPTConfig(norm_placement="sandwich", **base_kwargs), layer_idx=0)
    assert all(isinstance(getattr(sandwich_block, f"ln_{i}"), RMSNormFP32) for i in (1, 2, 3, 4))

    pre_block = Block(GPTConfig(norm_placement="pre", **base_kwargs), layer_idx=0)
    assert isinstance(pre_block.ln_1, torch.nn.LayerNorm)
    assert isinstance(pre_block.ln_2, torch.nn.LayerNorm)
    assert not hasattr(pre_block, "ln_3")
    assert not hasattr(pre_block, "ln_4")


def test_full_model_b_recipe_end_to_end():
    """RoPE + GQA + GeGLU + sandwich norm all at once: constructs, forwards warning-free under
    bf16 autocast, and produces a real loss."""
    config = GPTConfig(
        position_encoding="rope", n_kv_head=2, mlp_type="geglu", norm_placement="sandwich",
        n_embd=128, n_head=4, n_layer=2, block_size=16, vocab_size=65,
    )
    model = GPT(config)
    x = torch.randint(0, 65, (2, 10))
    y = torch.randint(0, 65, (2, 10))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, loss = model(x, y)

    assert len(caught) == 0, f"expected no warnings, got: {[str(w.message) for w in caught]}"
    assert logits.shape == (2, 10, 65)
    assert loss.item() > 0
