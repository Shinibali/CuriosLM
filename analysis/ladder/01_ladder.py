"""The architecture ladder: GPT-2 modernized one change at a time, 1.5B tokens per rung. Quality vs cost per rung.

Per rung: its one change, true parameters, final validation loss and the change from the previous rung (against the
seed-to-seed noise band), median training throughput and peak memory, and KV-cache size per sequence at the full
context. Runs are found on wandb by the names config/ladder_gpu.py builds; histories are cached in data/ so the
figures rebuild offline.

    uv run --extra analysis python analysis/ladder/01_ladder.py [--refresh]
Writes data/ladder.csv, figures/ladder_loss.png, figures/ladder_efficiency.png.
"""
import argparse
import csv
import json
import os
import statistics

import matplotlib.pyplot as plt

import ladder  # this study's shared choices; also puts analysis/ on sys.path
from common import models, runs, style  # noqa: E402

PATHS_JSON = os.path.join(ladder.DATA, "run_paths.json")
LOSS_CSV = os.path.join(ladder.DATA, "val_history.csv")
SPEED_CSV = os.path.join(ladder.DATA, "speed_history.csv")
CONFIG_JSON = os.path.join(ladder.DATA, "run_configs.json")
OUT_CSV = os.path.join(ladder.DATA, "ladder.csv")
LOSS_PNG = os.path.join(ladder.FIGURES, "ladder_loss.png")
EFFICIENCY_PNG = os.path.join(ladder.FIGURES, "ladder_efficiency.png")


def find_runs(values):
    """{rung: wandb run path}, matching each rung's run name (from the config file) in its wandb project."""
    import wandb
    api, paths = wandb.Api(), {}
    for rung, v in values.items():
        found = [r for r in api.runs(f"{runs.WANDB_ENTITY}/{v['wandb_project']}")
                 if r.name == v["wandb_run_name"] and r.state == "finished"]
        assert len(found) == 1, f"{rung}: expected one finished run named {v['wandb_run_name']}, found {len(found)}"
        paths[rung] = f"{v['wandb_project']}/{found[0].id}"
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-pull from wandb instead of using the cache")
    args = parser.parse_args()
    values = {r: ladder.config_values(r) for r in ladder.rungs()}
    cached = all(os.path.exists(p) for p in (PATHS_JSON, LOSS_CSV, SPEED_CSV, CONFIG_JSON))
    if args.refresh or not cached:
        os.makedirs(ladder.DATA, exist_ok=True)
        paths = find_runs(values)
        with open(PATHS_JSON, "w") as f:
            json.dump(paths, f, indent=2)
        all_paths = list(paths.values()) + [p for p in ladder.NOISE_RUNS if p not in paths.values()]
        runs.fetch(all_paths, ["iter", "val/loss"], ["out_dir"], LOSS_CSV, CONFIG_JSON)
        runs.fetch(all_paths, ["iter", "throughput/tokens_per_sec", "peak_mem_gb"], ["out_dir"], SPEED_CSV,
                   CONFIG_JSON)
    with open(PATHS_JSON) as f:
        paths = json.load(f)
    all_paths = list(paths.values()) + [p for p in ladder.NOISE_RUNS if p not in paths.values()]
    loss, _ = runs.load(all_paths, LOSS_CSV, CONFIG_JSON)
    speed, configs = runs.load(all_paths, SPEED_CSV, CONFIG_JSON)
    gpus = {configs[p]["world_size"] for p in paths.values()}
    assert len(gpus) == 1, f"rungs ran on different GPU counts {gpus}: their throughputs aren't comparable"
    block = {ladder.model_config(v).block_size for v in values.values()}.pop()  # the context length every rung uses

    final = {p: max(rows, key=lambda r: r["iter"]) for p, rows in loss.items()}  # last evaluation of each run
    noise = abs(final[ladder.NOISE_RUNS[0]]["val/loss"] - final[ladder.NOISE_RUNS[1]]["val/loss"])

    rows, previous = [], None
    for rung, path in paths.items():
        config = ladder.model_config(values[rung])
        logged = [r for r in speed[path] if r["iter"] > 0]  # iter 0 includes compilation
        val = final[path]["val/loss"]
        rows.append(dict(
            rung=rung, change=ladder.CHANGE[rung], params_m=ladder.n_params(config) / 1e6,
            val_loss=val, val_iter=int(final[path]["iter"]), delta=None if previous is None else val - previous,
            tokens_per_sec=statistics.median(r["throughput/tokens_per_sec"] for r in logged),  # median: uploads slow a few
            peak_mem_gb=max(r["peak_mem_gb"] for r in logged),
            kv_mb=models.kv_bytes_formula(config, config.block_size, 1, ladder.KV_BYTES) / 1e6,
        ))
        previous = val
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["noise"])
        w.writeheader()
        w.writerows({**r, "noise": noise} for r in rows)
    print(f"seed-to-seed noise ±{noise:.4f}")
    for r in rows:
        delta = "" if r["delta"] is None else f" ({r['delta']:+.4f} vs previous)"
        print(f"{r['rung']} ({r['change']}): {r['params_m']:.2f}M params, val loss {r['val_loss']:.4f}{delta}, "
              f"{r['tokens_per_sec'] / 1e3:.0f}k tok/s, peak {r['peak_mem_gb']:.1f} GB, KV cache {r['kv_mb']:.1f} MB")

    style.apply()
    labels = [f"{r['rung']}\n{r['change']}" for r in rows]

    # Figure 1: each rung's step against the noise band
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    steps = rows[1:]
    xs = range(len(steps))
    ax.axhspan(-noise, noise, color=style.GRID, alpha=0.7, linewidth=0, zorder=0)  # steps in here are noise
    ax.axhline(0, color=style.INK_SECONDARY, linewidth=0.8, zorder=1)
    for x, r in zip(xs, steps):
        clears = abs(r["delta"]) > ladder.CLEAR * noise  # highlight only steps well outside the band
        ax.plot([x, x], [0, r["delta"]], color=style.ACCENT if clears else style.INK_SECONDARY, linewidth=2,
                zorder=2)  # stem from zero to the step
        ax.plot(x, r["delta"], "o", markersize=8, color=style.ACCENT if clears else style.INK_SECONDARY, zorder=3)
        ax.text(x + 0.08, r["delta"], f"{r['delta']:+.3f}", va="center", fontsize=9, color=style.INK)
    ax.text(len(steps) - 0.5, noise, f"seed-to-seed noise ±{noise:.3f}", ha="right", va="bottom", fontsize=8.5,
            color=style.INK_SECONDARY)
    ax.set_xticks(list(xs), labels[1:], fontsize=8.5)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylabel("Change in validation loss\nvs previous rung")
    ax.grid(axis="x", visible=False)
    fig.text(0.02, 0.98, "Only RoPE clearly moves the loss", fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.92, f"Each rung adds one change to the one before. Highlighted: steps over {ladder.CLEAR}× the seed-to-seed gap.", fontsize=9.5,
             color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.83, bottom=0.18)
    os.makedirs(ladder.FIGURES, exist_ok=True)
    fig.savefig(LOSS_PNG, dpi=200)

    # Figure 2: what each rung costs to train and to serve, side by side (two scales, two panels)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    ys = range(len(rows))[::-1]  # a0 at the top
    for ax, key, scale, unit, title in ((left, "tokens_per_sec", 1e3, "k tok/s", f"Training throughput ({gpus.pop()} GPUs)"),
                                        (right, "kv_mb", 1, " MB", f"KV cache per sequence, {block:,} tokens")):
        vals = [r[key] / scale for r in rows]
        ax.barh(list(ys), vals, height=0.6, color=style.OTHER[0])
        for y, v in zip(ys, vals):
            ax.text(v, y, f" {v:.0f}{unit}" if v >= 10 else f" {v:.1f}{unit}", va="center", fontsize=9,
                    color=style.INK)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.3)  # room for the value labels
    left.set_yticks(list(ys), labels, fontsize=8.5)
    fig.text(0.02, 0.98, "The modern recipe trains slower but is far cheaper to serve", fontsize=12,
             fontweight="bold", va="top")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.82, bottom=0.08, wspace=0.12)
    fig.savefig(EFFICIENCY_PNG, dpi=200)
    print(f"wrote {OUT_CSV}, {LOSS_PNG}, {EFFICIENCY_PNG}")


if __name__ == "__main__":
    main()
