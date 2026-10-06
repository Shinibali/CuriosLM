"""What YaRN does to Model B's RoPE bands: each band's wavelength before and after stretching the context.

YaRN sorts bands by full turns within the training context (thresholds from model.py): fast bands keep their
wavelength, slow bands get theirs multiplied by the stretch factor, and the bands in between are blended. Both
wavelength sets are read from the rotation tables model.py builds, with and without YaRN, so the figure shows the
implementation the evaluation runs, not a re-derivation.

    uv run --extra analysis python analysis/positions/06_yarn_bands.py
Needs Model B's checkpoint (downloaded if missing). Writes data/yarn_bands.csv and figures/yarn_bands.png.
"""
import csv
import math
import os

import matplotlib.pyplot as plt

import positions  # this study's shared choices; also puts analysis/ on sys.path
from common import style  # noqa: E402

OUT_CSV = os.path.join(positions.DATA, "yarn_bands.csv")
OUT_PNG = os.path.join(positions.FIGURES, "yarn_bands.png")


def main():
    model_args = positions.model_args(positions.out_dir(positions.ROPE))
    trained_context = model_args["block_size"]
    target_context = positions.YARN_FACTOR * trained_context
    before = [2 * math.pi / freq for freq in positions.rope_frequencies(model_args)[0].tolist()]  # positions per turn
    after = [2 * math.pi / freq for freq in positions.rope_frequencies(model_args, positions.YARN_FACTOR)[0].tolist()]
    groups = positions.speed_groups(before, trained_context)  # YaRN's split, by turns within the training context
    group_of = {band: name for name, bands in groups.items() for band in bands}

    os.makedirs(positions.DATA, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["band", "group", "wavelength", "wavelength_with_yarn", "stretch"])
        writer.writerows([band, group_of[band], before[band], after[band], after[band] / before[band]]
                         for band in range(len(before)))
    turning_trained = sum(wavelength <= trained_context for wavelength in before)
    turning_plain = sum(wavelength <= target_context for wavelength in before)  # no YaRN, run at the target
    turning_yarn = sum(wavelength <= target_context for wavelength in after)
    print(f"YaRN groups: {', '.join(f'{name} {len(bands)}' for name, bands in groups.items())}")
    print(f"bands completing a turn: {turning_trained} within {trained_context:,} in training; within "
          f"{target_context:,}: {turning_plain} without YaRN, {turning_yarn} with it")

    style.apply()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = {"fast": f"fast: kept (≥ {positions.YARN_BETA} turns in {trained_context:,})",
              "middle": "middle: blended",
              "slow": f"slow: stretched {positions.YARN_FACTOR}× (< {positions.YARN_ALPHA} turn)"}
    for (name, bands), color in zip(groups.items(), style.OTHER):  # same group colours as the real-heads figure
        for band in bands:
            ax.plot([band, band], [before[band], after[band]], color=color, linewidth=1, zorder=2)
        ax.scatter(bands, [before[band] for band in bands], s=26, facecolors=style.SURFACE, edgecolors=color,
                   linewidths=1.2, zorder=3)
        ax.scatter(bands, [after[band] for band in bands], s=26, color=color, zorder=3, label=labels[name])
    for context, text in ((trained_context, f"training length: {trained_context:,}"),
                          (target_context, f"YaRN target: {target_context:,}")):
        ax.axhline(context, color=style.INK, linewidth=1, linestyle="--", zorder=1)
        ax.text(0, context * 1.2, text, fontsize=8.5, color=style.INK)
    ax.set_yscale("log")
    ax.set_xlabel("RoPE band (pair of dimensions), fastest to slowest")
    ax.set_ylabel("Wavelength (positions per turn)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    title = f"YaRN stretches Model B's slow bands {positions.YARN_FACTOR}× and leaves the fast ones alone"
    fig.text(0.02, 0.98, title, fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.92, "Hollow: as trained. Filled: with YaRN.", fontsize=9.5, color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.85, bottom=0.13)
    os.makedirs(positions.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_CSV}, {OUT_PNG}")


if __name__ == "__main__":
    main()
