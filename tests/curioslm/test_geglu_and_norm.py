import warnings

import torch

from curioslm.config import GPTConfig
from curioslm.model import GPT, Block, GeGLU, MLP, RMSNormFP32


def test_geglu_matches_mlp_param_count():
    """
    The whole point of the 2/3*4x hidden-dim convention: GeGLU (3 matrices) should land
    within a few params of plain MLP (2 matrices) at the same n_embd, not meaningfully
    more or less -- otherwise Model A vs Model B stops being a fair size comparison.
    """
    config = GPTConfig(n_embd=128, n_head=4, n_kv_head=2, n_layer=1, block_size=16, vocab_size=65)
    geglu_params = sum(p.numel() for p in GeGLU(config).parameters())
    mlp_params = sum(p.numel() for p in MLP(config).parameters())
    assert abs(geglu_params - mlp_params) / mlp_params < 0.01, (
        f"GeGLU ({geglu_params:,}) and MLP ({mlp_params:,}) params differ by more than 1%"
    )


def test_rmsnorm_fp32_computes_in_fp32_and_preserves_input_dtype():
    """
    RMSNormFP32 exists specifically to avoid nn.RMSNorm's dtype-mismatch fallback under
    autocast (bf16 input, fp32 weight) -- verify it both runs warning-free under bf16 and
    still hands back the same dtype it was given, so callers don't need to care.
    """
    norm = RMSNormFP32(64)
    x = torch.randn(2, 8, 64, dtype=torch.bfloat16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = norm(x)

    assert len(caught) == 0, f"expected no warnings, got: {[str(w.message) for w in caught]}"
    assert out.dtype == torch.bfloat16


def test_block_sandwich_norm_has_four_norms_pre_has_two():
    """
    Sandwich norm wraps both sub-layers (attention, MLP) with pre- and post-norm each --
    4 norm modules. Plain pre-norm only normalizes the input to each sub-layer -- 2.
    """
    base_kwargs = dict(n_embd=128, n_head=4, n_kv_head=2, n_layer=1, block_size=16, vocab_size=65)

    sandwich_block = Block(GPTConfig(norm_placement="sandwich", **base_kwargs), layer_idx=0)
    assert all(isinstance(getattr(sandwich_block, f"ln_{i}"), RMSNormFP32) for i in (1, 2, 3, 4))

    pre_block = Block(GPTConfig(norm_placement="pre", **base_kwargs), layer_idx=0)
    assert isinstance(pre_block.ln_1, torch.nn.LayerNorm)
    assert isinstance(pre_block.ln_2, torch.nn.LayerNorm)
    assert not hasattr(pre_block, "ln_3")
    assert not hasattr(pre_block, "ln_4")


def test_full_model_b_recipe_end_to_end():
    """
    RoPE + GQA + GeGLU + sandwich norm combined -- the actual full "Model B" recipe,
    exercised together for the first time. Just needs to construct, forward cleanly
    under bf16 autocast with no warnings, and produce a real loss.
    """
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
