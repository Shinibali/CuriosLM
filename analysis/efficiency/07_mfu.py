"""MFU over the whole of each headline run: Model A against Model B.

The training script logs MFU with the PaLM formula (6N + 12·L·d·T FLOPs per token: attention over the full window for
every token). Counting only the FLOPs a causal, and for Model B windowed, model actually performs gives a constant
fraction of that per model, the ratio of 02_flops.py's useful training FLOPs to its PaLM FLOPs, so the useful MFU is
the logged curve scaled by it. Both are drawn, so the gap between the two definitions is visible. The values are
plotted exactly as logged, every logged step, as wandb shows them: train.py already logs a running average (each new
reading weighted 0.1) and skips compile warm-up and eval steps, so the dips are real slow stretches of training.

    uv run --extra analysis python analysis/efficiency/07_mfu.py [--refresh]
Needs 02_flops.py's data/flops.csv and wandb (first run, or --refresh). Writes data/mfu_history.csv and
figures/mfu.png.
"""
import argparse
import csv
import os
import statistics
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # analysis/, for common
from common import run_ids, runs, style  # noqa: E402

DATA, FIGURES = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
HISTORY_CSV = os.path.join(DATA, "mfu_history.csv")
CONFIG_JSON = os.path.join(DATA, "mfu_run_configs.json")
FLOPS_CSV = os.path.join(DATA, "flops.csv")  # written by 02_flops.py
OUT_PNG = os.path.join(FIGURES, "mfu.png")
MODELS = {"Model A": (run_ids.HEADLINE_A, style.MODEL_A), "Model B": (run_ids.HEADLINE_B, style.MODEL_B)}


def useful_share(name):
    """Useful training FLOPs as a share of the PaLM formula's, for this model, from 02_flops.py."""
    with open(FLOPS_CSV) as f:
        row = next(row for row in csv.DictReader(f) if row["model"].startswith(name))
    return float(row["train_mflops_per_token"]) / float(row["palm_mflops_per_token"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="fetch the histories from wandb again")
    args = parser.parse_args()
    if args.refresh or not os.path.exists(HISTORY_CSV):
        os.makedirs(DATA, exist_ok=True)
        runs.fetch([path for path, _ in MODELS.values()], ["tokens", "mfu"], [], HISTORY_CSV, CONFIG_JSON)
    with open(HISTORY_CSV) as f:
        rows = list(csv.DictReader(f))

    style.apply()
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    for name, (path, color) in MODELS.items():
        points = sorted((float(r["tokens"]), float(r["mfu"])) for r in rows if r["run"] == path)
        tokens = [t / 1e9 for t, _ in points]
        logged = [m for _, m in points]  # as logged, no extra smoothing
        share = useful_share(name)
        useful = [m * share for m in logged]
        ax.plot(tokens, logged, color=color, linewidth=1, label=f"{name}, as logged (PaLM formula)")
        ax.plot(tokens, useful, color=color, linewidth=1, linestyle="--", label=f"{name}, useful FLOPs only")
        print(f"{name}: median MFU logged {statistics.median(m for _, m in points):.1f}%, useful "
              f"{statistics.median(m for _, m in points) * share:.1f}% (useful = {share:.3f} × logged)")
    ax.set_xlabel("Tokens trained (billions)")
    ax.set_ylabel("MFU (% of the GPUs' bf16 peak)")
    # zoomed to the lines, as wandb shows it: a line chart doesn't need a zero baseline, and at full scale the dips
    # (real slow stretches) shrink to nothing
    ax.margins(y=0.08)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5, ncol=2)  # above the plot, clear of the lines
    fig.suptitle("Model FLOPs utilization across the 9.8B-token runs, 4× H100", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {HISTORY_CSV}, {OUT_PNG}")


if __name__ == "__main__":
    main()
