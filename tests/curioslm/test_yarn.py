"""YaRN context extension (Peng et al. 2023, https://arxiv.org/abs/2309.00071): the interface her model.py change
must meet. Checks behaviour from the outside (rotation tables, norms, masks, checkpoint loading), not the formula."""
import math

import torch

from curioslm import checkpoints
from curioslm.config import GPTConfig
from curioslm.model import GPT

ORIGINAL_CONTEXT, FACTOR = 1024, 4.0  # Model B's training length; the 4× target decided for Part 2 §6
ALPHA, BETA = 1, 32  # the paper's band thresholds, in turns within the original context (its Section 3.2)


def tiny_config(**overrides):
    """One 64-wide head: 32 bands at base 10,000 cover YaRN's fast, middle and slow groups at a 1,024 context."""
    settings = dict(n_embd=64, n_head=1, n_layer=1, vocab_size=65, position_encoding="rope",
                    block_size=int(ORIGINAL_CONTEXT * FACTOR), yarn_factor=FACTOR,
                    yarn_original_context=ORIGINAL_CONTEXT)
    settings.update(overrides)
    return GPTConfig(**settings)


def rope_tables(config):
    """(cosines, sines) buffers of layer 0, each (block_size, head_size)."""
    rope = GPT(config).transformer.h[0].attn.rope
    return rope.rope_cosines, rope.rope_sines


def band_frequencies(config):
    """Each band's angle per position, read back from the tables at position 1 (every angle there is ≤ 1 rad)."""
    cosines, sines = rope_tables(config)
    bands = cosines.size(1) // 2  # first half and second half repeat the same angles
    return torch.atan2(sines[1, :bands], cosines[1, :bands]).double()


def test_factor_one_leaves_rope_unchanged():
    """yarn_factor = 1 must reproduce plain RoPE exactly, so every existing checkpoint behaves as before."""
    plain = GPTConfig(n_embd=64, n_head=1, n_layer=1, vocab_size=65, position_encoding="rope",
                      block_size=ORIGINAL_CONTEXT)
    for scaled_table, plain_table in zip(rope_tables(tiny_config(yarn_factor=1.0, block_size=ORIGINAL_CONTEXT)),
                                         rope_tables(plain)):
        torch.testing.assert_close(scaled_table, plain_table)


def test_bands_follow_yarns_three_groups():
    """Fast bands (≥ β turns in the original context) keep their angle; slow bands (< α turns) are divided by the
    factor; bands in between land strictly between the two, slower bands moving further."""
    plain = band_frequencies(tiny_config(yarn_factor=1.0))
    scaled = band_frequencies(tiny_config())
    turns = ORIGINAL_CONTEXT * plain / (2 * math.pi)  # full turns each band makes within the original context
    fast, slow = turns >= BETA, turns < ALPHA
    middle = ~fast & ~slow
    assert fast.any() and middle.any() and slow.any(), "test config should cover all three groups"
    torch.testing.assert_close(scaled[fast], plain[fast])
    torch.testing.assert_close(scaled[slow], plain[slow] / FACTOR)
    assert ((scaled[middle] < plain[middle]) & (scaled[middle] > plain[middle] / FACTOR)).all()
    kept_share = (scaled[middle] - plain[middle] / FACTOR) / (plain[middle] - plain[middle] / FACTOR)
    assert (kept_share.diff() <= 0).all(), "slower middle bands should be interpolated more"


def test_temperature_scales_rotated_vectors():
    """The attention temperature rides on the tables: rotation keeps a vector's length, so the rotated query and key
    each grow by exactly √(1/t) = 0.1·ln(s) + 1 (YaRN Eq. 22), and q·k by its square."""
    torch.manual_seed(0)
    rope = GPT(tiny_config()).transformer.h[0].attn.rope
    q, k = torch.randn(2, 1, 1, 16, 64).unbind(0)  # (B, nh, T, hs) each
    q_rotated, k_rotated = rope(q, k)
    expected = 0.1 * math.log(FACTOR) + 1
    torch.testing.assert_close(q_rotated.norm(dim=-1) / q.norm(dim=-1), torch.full((1, 1, 16), expected))
    torch.testing.assert_close(k_rotated.norm(dim=-1) / k.norm(dim=-1), torch.full((1, 1, 16), expected))


def test_full_extended_sequence_runs_and_local_window_holds():
    """A 4,096-token input runs end to end, and a local layer still sees exactly its window at the last position."""
    window = 8
    config = tiny_config(n_layer=2, attention_window=window, local_global_ratio=2)  # layer 1 is local
    model = GPT(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, config.block_size))
    with torch.no_grad():
        logits, _ = model(tokens, tokens)
    assert logits.shape == (1, config.block_size, config.vocab_size)
    local = model.transformer.h[1].attn
    assert local.window_mask.shape == (config.block_size, config.block_size)
    assert local.window_mask[-1].sum().item() == window


def test_trained_checkpoint_loads_with_yarn_overrides():
    """A 1,024-token checkpoint saved before YaRN existed (no yarn keys) loads at 4,096 with YaRN switched on: these
    overrides change the position tables and masks, never the weights."""
    trained = GPT(tiny_config(yarn_factor=1.0, block_size=ORIGINAL_CONTEXT))
    saved_args = {key: value for key, value in vars(trained.config).items() if not key.startswith("yarn")}
    checkpoint = {"model_args": saved_args, "model": trained.state_dict()}
    model, model_args = checkpoints.build_model(checkpoint, overrides={
        "block_size": int(ORIGINAL_CONTEXT * FACTOR), "yarn_factor": FACTOR,
        "yarn_original_context": ORIGINAL_CONTEXT})
    assert model.config.block_size == ORIGINAL_CONTEXT * FACTOR and model.config.yarn_factor == FACTOR
    torch.testing.assert_close(band_frequencies(model.config), band_frequencies(tiny_config()))
