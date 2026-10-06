"""Decode step time against bytes read per step, on each machine 01 was run on: memory-bound or overhead-bound?

On a machine where decoding is limited by memory traffic, step time rises with the bytes each step reads (weights
plus KV cache). Where it's limited by fixed per-step overhead (launching many small kernels, one token at a time),
step time stays flat however much is read, and the model that launches more kernels per step is slower.

    uv run --extra analysis python analysis/efficiency/04_devices.py
Needs 03's data/decode_bytes.csv for each machine in MACHINES. Writes figures/decode_regimes.png.
"""
import csv
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # analysis/, for common
from common import style  # noqa: E402

DATA, FIGURES = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
# Choices: which 01/03 runs to compare. The machine each file was measured on is recorded here, once.
MACHINES = {"": "MacBook M1 Pro (MPS, float32)", "_h100": "H100 SXM (CUDA, float32)"}


def main():
    style.apply()
    fig, axes = plt.subplots(1, len(MACHINES), figsize=(10, 4), sharey=False)
    for ax, (suffix, machine) in zip(axes, MACHINES.items()):
        with open(os.path.join(DATA, f"decode_bytes{suffix}.csv")) as f:
            rows = list(csv.DictReader(f))
        models = list(dict.fromkeys(r["model"] for r in rows))  # in file order: A, then B
        for model, color in zip(models, style.SERIES):
            pts = sorted((float(r["mb_per_step"]), float(r["ms_per_step"]), r) for r in rows if r["model"] == model)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color, markersize=6,
                    label=model.split(" (")[1][:-1])
        ax.set_title(machine, fontsize=10)
        ax.set_xlabel("MB read per decode step (weights + KV cache)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    axes[0].set_ylabel("ms per decode step")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.text(0.01, 0.99, "Same models, two machines, two different bottlenecks", fontsize=12, fontweight="bold",
             va="top")
    fig.text(0.01, 0.93, "Each point: batch 1 or 8 × a 128- or 512-token prompt. Rising: time grows with the work per step. Flat: fixed overhead.",
             fontsize=9, color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(top=0.8, bottom=0.14, left=0.07, right=0.98, wspace=0.18)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "decode_regimes.png"), dpi=200)
    print(f"wrote {os.path.join(FIGURES, 'decode_regimes.png')}")


if __name__ == "__main__":
    main()
