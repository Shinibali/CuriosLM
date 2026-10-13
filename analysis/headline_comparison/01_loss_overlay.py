"""Figure: Model A (GPT-2) vs Model B (Gemma-style) validation loss over the 9.8B-token headline runs.

Top: both models' validation loss vs tokens. Bottom: the gap (B − A) against the run-to-run noise band, measured
as the final-loss difference between two seeds of the same Gemma-style ladder run. Histories and configs are
pulled from wandb once and cached in data/, so the figure rebuilds offline.

    uv run --extra analysis python analysis/headline_comparison/01_loss_overlay.py [--refresh]
"""
import argparse
import os

import matplotlib.pyplot as plt

import headline  # this study's shared choices; also puts analysis/ on sys.path
from common import runs, style  # noqa: E402

HISTORY_CSV = os.path.join(headline.DATA, "val_history.csv")
OUT = os.path.join(headline.FIGURES, "loss_overlay.png")
RUNS = [(headline.LABELS[p], p) for p in (headline.MODEL_A, headline.MODEL_B)]  # plotting order = colour order
NOISE_RUNS = headline.NOISE_RUNS
CONFIG_JSON = headline.CONFIG_JSON


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-pull from wandb instead of using the cache")
    args = parser.parse_args()
    paths = [p for _, p in RUNS] + list(NOISE_RUNS)
    if args.refresh or not (os.path.exists(HISTORY_CSV) and os.path.exists(CONFIG_JSON)):
        os.makedirs(headline.DATA, exist_ok=True)
        runs.fetch(paths, ["iter", "tokens", "val/loss"], headline.CONFIG_KEYS, HISTORY_CSV, CONFIG_JSON)
    history, _ = runs.load(paths, HISTORY_CSV, CONFIG_JSON)
    full = {p: sorted((r["tokens"], r["val/loss"]) for r in rows) for p, rows in history.items()}  # from iter 0
    curves = {p: [point for point in points if point[0] > 0] for p, points in full.items()}  # trained evals only:
    # the gap and the noise compare trained models, so the untrained iter 0 stays out of them

    a, b = (curves[p] for _, p in RUNS)
    assert [t for t, _ in a] == [t for t, _ in b], "the two runs must be evaluated at the same token counts"
    tokens = [t / 1e9 for t, _ in a]  # billions
    gap = [lb - la for (_, la), (_, lb) in zip(a, b)]  # B − A: negative means Gemma-style is ahead
    noise = abs(curves[NOISE_RUNS[0]][-1][1] - curves[NOISE_RUNS[1]][-1][1])  # seed-to-seed final val-loss gap

    style.apply()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for (label, path), color in zip(RUNS, style.SERIES):  # from the untrained model at iter 0
        top.plot([t / 1e9 for t, _ in full[path]], [l for _, l in full[path]], color=color, label=label, marker="o",
                 markersize=3)
    style.mark_uniform_loss(top)
    top.set_ylabel("Validation loss")
    top.legend(loc="center right", fontsize=8.5)  # the empty middle: the uniform-guess label sits top right

    bottom.axhspan(-noise, noise, color=style.GRID, alpha=0.6, linewidth=0, zorder=0)  # differences in here are noise
    bottom.axhline(0, color=style.INK_SECONDARY, linewidth=0.8, zorder=1)
    bottom.plot(tokens, gap, color=style.INK, marker="o", markersize=3, zorder=3)  # neutral: the gap is neither model
    bottom.text(tokens[-1], noise, f"seed-to-seed noise ±{noise:.3f}", ha="right", va="bottom", fontsize=8.5,
                color=style.INK_SECONDARY)
    bottom.set_ylabel("Gap, B − A")
    bottom.set_xlabel("Tokens seen (billions)")
    bottom.set_xlim(left=0)  # the axes share x: training starts at 0 tokens, with no padding before it

    fig.text(0.02, 0.985, "Same loss by the end", fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.945, f"Gemma-style leads by {-min(gap):.3f} early; after {tokens[-1]:.1f}B tokens the gap is "
             f"{gap[-1]:+.3f}, inside run-to-run noise.", fontsize=9.5, color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.09, hspace=0.08)
    os.makedirs(headline.FIGURES, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}; final A {a[-1][1]:.4f}, B {b[-1][1]:.4f}, gap {gap[-1]:+.4f}, noise ±{noise:.4f}")


if __name__ == "__main__":
    main()
