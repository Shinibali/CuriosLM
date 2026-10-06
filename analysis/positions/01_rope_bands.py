"""RoPE's frequency ladder against the training context: which rotation bands complete a full cycle within it.

RoPE rotates each pair of query/key dimensions by an angle proportional to position, at its own frequency. Fast
bands turn many times within the context; slow bands never complete one turn, so the model only ever sees part of
their cycle. Those slow bands are the ones YaRN rescales to extend the context. Frequencies are read from the
model's own rotation table (the angle at position 1), not recomputed from a formula.

    uv run --extra analysis python analysis/positions/01_rope_bands.py
Needs Model B's checkpoint (downloaded if missing). Writes data/rope_bands.csv and figures/rope_bands.png.
"""
import csv
import math
import os

import matplotlib.pyplot as plt

import positions  # this study's shared choices; also puts analysis/ on sys.path
from common import style  # noqa: E402

OUT_CSV = os.path.join(positions.DATA, "rope_bands.csv")
OUT_PNG = os.path.join(positions.FIGURES, "rope_bands.png")


def main():
    model_args = positions.model_args(positions.out_dir(positions.ROPE))
    context = model_args["block_size"]
    freqs = positions.rope_frequencies(model_args)[0].tolist()  # angle turned per position, per band
    wavelengths = [2 * math.pi / freq for freq in freqs]  # positions per full turn
    num_turning = sum(wavelength <= context for wavelength in wavelengths)  # bands completing a turn in the context
    num_bands = len(freqs)

    os.makedirs(positions.DATA, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["band", "radians_per_position", "wavelength", "turns_in_context"])
        writer.writerows([band, freq, wavelength, context / wavelength]
                         for band, (freq, wavelength) in enumerate(zip(freqs, wavelengths)))
    print(f"{num_bands} bands; {num_turning} complete a turn within {context} positions, "
          f"{num_bands - num_turning} never do; "
          f"longest wavelength {wavelengths[-1]:,.0f} positions")

    style.apply()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = [style.ACCENT if wavelength <= context else style.INK_SECONDARY for wavelength in wavelengths]
    ax.scatter(range(len(wavelengths)), wavelengths, color=colors, s=28, zorder=3)
    ax.axhline(context, color=style.INK, linewidth=1, linestyle="--", zorder=2)
    ax.text(len(wavelengths) - 0.5, context * 1.25, f"training context: {context:,} tokens", ha="right",
            fontsize=9, color=style.INK)
    ax.text(0, wavelengths[-1] / 2, f"{num_turning} bands complete a turn\nwithin the context", fontsize=9,
            color=style.ACCENT, va="center")
    ax.text(num_turning + 1, wavelengths[-1], f"{num_bands - num_turning} bands never do", fontsize=9,
            color=style.INK_SECONDARY, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("RoPE band (pair of dimensions), fastest to slowest")
    ax.set_ylabel("Wavelength (positions per turn)")
    ax.grid(axis="x", visible=False)
    fig.text(0.02, 0.98, f"{num_bands - num_turning} of RoPE's {num_bands} bands never finish a turn in training",
             fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.92, f"Model B: {num_bands} bands per head, wavelengths from {wavelengths[0]:.0f} to "
             f"{wavelengths[-1]:,.0f} positions.", fontsize=9.5, color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.14)
    os.makedirs(positions.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_CSV}, {OUT_PNG}")


if __name__ == "__main__":
    main()
