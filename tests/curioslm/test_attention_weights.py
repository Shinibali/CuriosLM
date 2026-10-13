"""The explicit attention path (CausalSelfAttention._attention_with_weights): the contract her implementation must meet.
Checked from the outside: the same output as the fused kernels, and probabilities with the shape, sums and zeros the
masks imply, for the vanilla GPT-2 layout and the Gemma-style one (RoPE, GQA, local and global layers)."""
import pytest
import torch

from curioslm.config import GPTConfig
from curioslm.model import GPT

WINDOW = 4  # small enough that a 12-token input has positions outside every local window
LAYOUTS = {
    "gpt2": dict(),  # learned positions, full multi-head attention, every layer global
    "gemma": dict(position_encoding="rope", n_kv_head=2, mlp_type="geglu", norm_placement="sandwich",
                  attention_window=WINDOW, local_global_ratio=2),  # layers 0, 2 global; 1, 3 local
}


def tiny_model(layout):
    torch.manual_seed(0)
    config = GPTConfig(n_embd=32, n_head=4, n_layer=4, block_size=16, vocab_size=65, **LAYOUTS[layout])
    return GPT(config).eval()


def tokens():
    return torch.randint(0, 65, (2, 12), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_explicit_path_gives_the_same_output_as_the_fused_kernels(layout):
    """Logits at every position match the normal forward: the explicit path computes the same attention."""
    model, idx = tiny_model(layout), tokens()
    with torch.no_grad():
        reference, _ = model(idx, idx)  # targets given: logits at every position
        for block in model.transformer.h:
            block.attn.capture_attention = True
        explicit, _ = model(idx, idx)
    torch.testing.assert_close(explicit, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_probabilities_are_distributions_over_the_allowed_keys(layout):
    """One (B, n_head, T, T) tensor per layer; each row sums to 1; no weight on future keys, nor, in a local layer, on
    keys further back than the window."""
    model, idx = tiny_model(layout), tokens()
    weights = model.attention_weights(idx)
    batch, length = idx.shape
    assert len(weights) == model.config.n_layer
    positions = torch.arange(length)
    distance = positions[:, None] - positions[None, :]  # query position − key position
    for layer, probabilities in enumerate(weights):
        assert probabilities.shape == (batch, model.config.n_head, length, length)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(batch, model.config.n_head, length))
        assert (probabilities[..., distance < 0] == 0).all(), f"layer {layer} attends to future keys"
        if not model.config.is_global_layer(layer):
            assert (probabilities[..., distance >= WINDOW] == 0).all(), f"layer {layer} looks past its window"


def test_switch_is_turned_off_afterwards():
    """attention_weights leaves every layer back on the fused path, holding no probabilities."""
    model = tiny_model("gemma")
    model.attention_weights(tokens())
    for block in model.transformer.h:
        assert block.attn.capture_attention is False and block.attn.last_attention is None
