"""RoPE at longer contexts, by arithmetic alone: how RoPE's bands relate to the context as it grows, the groundwork
for YaRN, and what a learned position table would cost instead.

1. For a 7B-class head (head size 128), how many bands stay slower than a 128k context, for the RoPE bases in use.
2. For Model B, how many bands first complete a turn when its context is stretched 2×, 4× and 8×.
3. YaRN's split of Model B's bands (left alone / blended / rescaled), from the same speed groups as the real-heads
   figure (positions.speed_groups).
4. The size of a learned position table for a 128k context at 7B-class width.

    uv run --extra analysis python analysis/positions/05_rope_scaling.py
Needs Model B's checkpoint. Writes data/rope_scaling.csv.
"""
import csv
import math
import os

import positions  # this study's shared choices; also puts analysis/ on sys.path
from common.reference_models import LONG_CONTEXT, SEVEN_B  # noqa: E402

OUT_CSV = os.path.join(positions.DATA, "rope_scaling.csv")
# Choices: the RoPE bases compared, and the context stretches for the YaRN groundwork
ROPE_BASES = (10_000, 500_000, 1_000_000)  # ours (and Gemma 3's local layers); Llama 3; Gemma 3's global layers
STRETCHES = (2, 4, 8)


def wavelengths_for(head_size, base):
    """Positions per full turn of each band, from the model's own rotation table for this head size and base."""
    freqs, _ = positions.rope_frequencies({"n_embd": head_size, "n_head": 1, "rope_theta": base})
    return (2 * math.pi / freqs).tolist()


def main():
    facts = []
    head_size = SEVEN_B["n_embd"] // SEVEN_B["n_head"]
    for base in ROPE_BASES:
        wavelengths = wavelengths_for(head_size, base)
        facts.append((f"bands slower than a {LONG_CONTEXT:,}-token context, base {base:,}, head size {head_size} "
                      f"({len(wavelengths)} bands)", sum(wavelength > LONG_CONTEXT for wavelength in wavelengths)))
    model_b_args = positions.model_args(positions.out_dir(positions.ROPE))
    context = model_b_args["block_size"]
    model_b_wavelengths = (2 * math.pi / positions.rope_frequencies(model_b_args)[0]).tolist()
    for stretch in STRETCHES:
        newly_turning = sum(context < wavelength <= context * stretch for wavelength in model_b_wavelengths)
        facts.append((f"Model B bands that first complete a turn between {context:,} and {context * stretch:,} tokens",
                      newly_turning))
    groups = positions.speed_groups(model_b_wavelengths, context)
    facts += [("YaRN: Model B bands left alone (fast)", len(groups["fast"])),
              ("YaRN: Model B bands blended (middle)", len(groups["middle"])),
              ("YaRN: Model B bands fully rescaled (slow)", len(groups["slow"]))]
    facts.append((f"learned position table for {LONG_CONTEXT:,} positions at width {SEVEN_B['n_embd']:,} (millions "
                  "of params)", LONG_CONTEXT * SEVEN_B["n_embd"] / 1e6))
    for name, value in facts:
        print(f"  {name}: {value:,.3f}")
    os.makedirs(positions.DATA, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fact", "value"])
        writer.writerows(facts)


if __name__ == "__main__":
    main()
