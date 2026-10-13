"""Validation loss on each dataset of the mix, over training: the 50:50 run against Model B on the same architecture.

Model B (100% FineWeb-Edu) was only evaluated on FineWeb-Edu during training, so the Cosmopedia-v2 panel has the mix
alone; 03_val_perplexity.py scores both models on both sets after training.

    uv run --extra analysis python analysis/data_mix/01_loss_curves.py [--refresh]
Writes data/val_history.csv, figures/loss_curves.png.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt

import mix  # this study's shared choices; also puts analysis/ on sys.path
from common import style  # noqa: E402

HISTORY_CSV = os.path.join(mix.DATA, "val_history.csv")
OUT = os.path.join(mix.FIGURES, "loss_curves.png")


def fetch():
    """Every evaluation row of both runs: iter, tokens and each validation loss the run logged."""
    import wandb
    from common import runs
    api, rows = wandb.Api(), []
    for path in (mix.BASE, mix.MIX):
        run = api.run(f"{runs.WANDB_ENTITY}/{path}")
        keys = [mix.val_key(d) for d in mix.val_sets() if mix.val_key(d) in run.summary]  # the sets this run logged
        for x in run.scan_history(keys=["iter", "tokens", *keys]):
            rows += [dict(run=path, iter=x["iter"], tokens=x["tokens"], key=k, loss=x[k]) for k in keys
                     if x.get(k) is not None]
    with open(HISTORY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-pull from wandb instead of using the cache")
    args = parser.parse_args()
    os.makedirs(mix.DATA, exist_ok=True)
    if args.refresh or not os.path.exists(HISTORY_CSV):
        fetch()
    with open(HISTORY_CSV) as f:
        rows = list(csv.DictReader(f))  # from iter 0, the untrained model

    style.apply()
    sets = mix.val_sets()
    fig, axes = plt.subplots(1, len(sets), figsize=(10, 3.8))
    for ax, dataset in zip(axes, sets):
        ends = []  # (tokens, final loss) per curve, labelled after both are drawn
        for (path, label), color in (((mix.BASE, mix.LABELS[mix.BASE]), style.MODEL_B),
                                     ((mix.MIX, mix.LABELS[mix.MIX]), style.MODEL_B_MIX)):
            pts = sorted((float(r["tokens"]) / 1e9, float(r["loss"])) for r in rows
                         if r["run"] == path and r["key"] == mix.val_key(dataset))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", markersize=3, color=color, label=label)
                ends.append(pts[-1])
        label_gap = 0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # close finals would print on top of each other
        for (x, y), placed in zip(ends, style.spread_labels([y for _, y in ends], label_gap)):
            ax.text(x, placed, f"  {y:.3f}", va="center", fontsize=8.5, color=style.INK)
        style.mark_uniform_loss(ax, label=dataset == sets[0])  # labelled once
        ax.set_title(f"Validation loss on {mix.display(dataset)}", fontsize=10)
        ax.set_xlabel("Tokens seen (billions)")
        ax.set_xlim(0, ax.get_xlim()[1] * 1.12)  # room for the end labels
    axes[0].set_ylabel("Validation loss")
    axes[0].legend(loc="center right", fontsize=8.5)  # the empty middle: the uniform-guess label sits top right
    fig.text(0.01, 0.99, "Swapping half the data: a small loss on one set, a large gain on the other", fontsize=12,
             fontweight="bold", va="top")
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.07, right=0.98, wspace=0.2)
    os.makedirs(mix.FIGURES, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    finals = {(r["run"], r["key"]): float(r["loss"]) for r in sorted(rows, key=lambda r: int(r["iter"]))}  # last wins
    for (run, key), loss in finals.items():
        print(f"{mix.LABELS[run]}, {key}: final {loss:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
