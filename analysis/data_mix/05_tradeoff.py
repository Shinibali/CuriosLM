"""What swapping half the data bought and cost, on one page: the 50:50 run against Model B on every measure.

Left: held-out perplexity, % change (lower is better). Right: benchmark accuracy, change in points, with the paired
test's p-value. Model B's architecture and token budget are the same in both runs; only the data differs.

    uv run --extra analysis python analysis/data_mix/05_tradeoff.py
Needs 02's data/benchmarks.csv and 03's data/val_perplexity.csv. Writes figures/tradeoff.png.
"""
import csv
import os

import matplotlib.pyplot as plt

import mix  # this study's shared choices; also puts analysis/ on sys.path
from common import style  # noqa: E402

OUT = os.path.join(mix.FIGURES, "tradeoff.png")
ACCURACY = {"hellaswag": "acc_norm", "arc_easy": "acc", "lambada_openai": "acc"}  # the metric each task is ranked on
ALPHA = 0.05  # p-value below which a difference is drawn as significant


def main():
    with open(os.path.join(mix.DATA, "val_perplexity.csv")) as f:
        ppl = {(r["set"], r["model"]): r["perplexity"] for r in csv.DictReader(f)}
    with open(os.path.join(mix.DATA, "benchmarks.csv")) as f:
        bench = list(csv.DictReader(f))
    base, mixed = mix.LABELS[mix.BASE], mix.LABELS[mix.MIX]

    rows_ppl = [(f"{mix.display(d)} (held-out)", 100 * (float(ppl[(d, mixed)]) / float(ppl[(d, base)]) - 1)) for d in mix.val_sets()]
    rows_acc = []
    for task, metric in ACCURACY.items():
        r = next(r for r in bench if r["task"] == task and r["metric"] == metric)
        rows_acc.append((f"{mix.display(task)} ({metric})", 100 * float(r["diff"]), float(r["p"])))

    style.apply()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, rows, unit in ((left, rows_ppl, "%"), (right, rows_acc, " pt")):
        ys = range(len(rows))[::-1]
        lo, hi = min(r[1] for r in rows), max(r[1] for r in rows)
        pad = 0.25 * (hi - lo)
        for y, row in zip(ys, rows):
            better = row[1] < 0 if ax is left else row[1] > 0
            significant = ax is left or row[2] < ALPHA  # perplexity changes this size are far outside their CIs
            color = (style.GOOD if better else style.BAD) if significant else style.GRID
            ax.barh(y, row[1], height=0.55, color=color)
            p_text = "" if ax is left else ("  p < 0.001" if row[2] < 0.001 else f"  p = {row[2]:.2f}")
            ax.text(max(r[1] for r in rows) + pad / 4, y, f"{row[1]:+.1f}{unit}{p_text}", va="center", fontsize=8.5,
                    color=style.INK)  # every label in one column, right of the bars
        ax.axvline(0, color=style.INK_SECONDARY, linewidth=0.8)
        ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(min(lo, 0) - pad / 2, max(hi, 0) + 3 * pad)  # room for the label column
    left.set_title("Perplexity, % change (lower is better)", fontsize=10)
    right.set_title(f"Accuracy, points (grey: p ≥ {ALPHA})", fontsize=10)
    fig.text(0.01, 0.99, "Half the data swapped for Cosmopedia-v2: what changed",
             fontsize=12, fontweight="bold", va="top")
    fig.subplots_adjust(top=0.8, bottom=0.08, left=0.17, right=0.98, wspace=0.6)
    os.makedirs(mix.FIGURES, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
