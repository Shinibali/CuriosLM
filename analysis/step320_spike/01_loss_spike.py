"""Figure: a one-step training-loss spike appears in every run at the same step.

The runs differ in architecture, run length (so warmup) and GPU count; the only thing they share is the data order,
so the spike comes from the data. Nothing about the runs is hardcoded: the spike step is detected from the logged
losses, tokens per step comes from the logged `tokens`/`iter`, and warmup from each run's wandb config. Each run's
history and config are pulled from wandb once and cached in data/, so the figure rebuilds without wandb access.

    uv run --extra analysis python analysis/step320_spike/01_loss_spike.py [--refresh]
"""
import argparse
import json
import os

import matplotlib.pyplot as plt

import page  # also puts analysis/ on sys.path
from common import run_ids, runs, style  # noqa: E402

CACHE = os.path.join(page.DATA, "loss_history.csv")
CONFIG_CACHE = os.path.join(page.DATA, "run_configs.json")
OUT = os.path.join(page.FIGURES, "loss_spike.png")
# Label, wandb run path (project/run id) -- in the order they take the categorical colours. Labels name what
# differs between runs in words a reader needs no project context for: two architectures, two run lengths (which
# also means two warmups) and 2 vs 4 GPUs. Only the data order is shared.
RUNS = [
    ("124M GPT-2, 1.5B-token run", run_ids.LADDER_A0),
    ("113M Gemma-style, 1.5B-token run", run_ids.LADDER_A4),
    ("124M GPT-2, 9.8B-token run", run_ids.HEADLINE_A),
]
CONFIG_KEYS = ["batch_size", "gradient_accumulation_steps", "block_size", "warmup_iters", "dataset", "out_dir"]
HALF_WINDOW = 40e6  # display choice: tokens shown either side of the spike, enough flat curve to frame it


def load():
    """{path: sorted [(iter, tokens, loss)]} and {path: config} for the runs in RUNS."""
    paths = [path for _, path in RUNS]
    history, configs = runs.load(paths, CACHE, CONFIG_CACHE)
    return {p: sorted((int(r["iter"]), int(r["tokens"]), r["train/loss_step"]) for r in rows)
            for p, rows in history.items()}, configs


def tokens_per_iter(history):
    """Tokens per optimizer step, as train.py logged it. Must be the same for every run or steps aren't comparable."""
    per_run = {t // i for h in history.values() for i, t, _ in h if i}  # tokens / iter; skip iter 0
    assert len(per_run) == 1, f"runs disagree on tokens per step: {per_run}"
    return per_run.pop()


def find_shared_spike(history):
    """The logged step where every run jumps: for each step, the smallest one-step loss rise across runs, maximized.

    Not simply each run's largest jump: a longer run can have bigger one-off spikes that the others never see.
    """
    jumps = {path: {b[0]: b[2] - a[2] for a, b in zip(h, h[1:])} for path, h in history.items()}  # {path: {iter: Δ}}
    common = set.intersection(*(set(j) for j in jumps.values()))  # steps every run logged
    spike = max(common, key=lambda it: min(j[it] for j in jumps.values()))  # max over steps of the min over runs
    return spike, {path: j[spike] for path, j in jumps.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-pull from wandb instead of using the cache")
    args = parser.parse_args()
    if args.refresh or not (os.path.exists(CACHE) and os.path.exists(CONFIG_CACHE)):
        runs.fetch([path for _, path in RUNS], ["iter", "tokens", "train/loss_step"], CONFIG_KEYS, CACHE,
                   CONFIG_CACHE)
    history, configs = load()

    per_step = tokens_per_iter(history)
    for path, c in configs.items():  # cross-check the logs against each run's config (grad accum is global)
        assert c["batch_size"] * c["gradient_accumulation_steps"] * c["block_size"] == per_step, path
    spike_iter, spike_jumps = find_shared_spike(history)
    spike_tokens = spike_iter * per_step  # tokens seen before the spike step
    window = (spike_tokens - HALF_WINDOW, spike_tokens + HALF_WINDOW)
    print(f"tokens per step {per_step:,}; shared spike at step {spike_iter} = {spike_tokens:,} tokens")
    with open(page.SPIKE_JSON, "w") as f:  # 02 replays the loader to this step to find the batch
        json.dump({"step": spike_iter}, f)
    for label, path in RUNS:
        warmup_end = configs[path]["warmup_iters"] * per_step  # in tokens, to compare with the window
        print(f"  {label}: jump +{spike_jumps[path]:.2f}; warmup ends at {warmup_end / 1e6:.0f}M tokens "
              f"({'inside or after' if warmup_end > window[0] else 'before'} the window)")

    style.apply()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axvline(spike_tokens / 1e6, color=style.INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(spike_tokens / 1e6 + 0.8, 0.98, f"step {spike_iter} · {spike_tokens / 1e6:.1f}M tokens",
            transform=ax.get_xaxis_transform(), va="top", fontsize=9, color=style.INK_SECONDARY)

    ends = []
    for (label, path), color in zip(RUNS, style.SERIES):
        pts = [(t / 1e6, loss) for _, t, loss in history[path] if window[0] <= t <= window[1]]
        x, y = zip(*pts)
        ax.plot(x, y, color=color, label=label, zorder=3)
        ends.append((x[-1], y[-1]))  # each line's last point anchors its direct label
    label_ys = style.spread_labels([e[1] for e in ends], min_gap=0.12)  # min_gap in loss units
    for (label, _), (x_end, _), y_lab, color in zip(RUNS, ends, label_ys, style.SERIES):
        ax.plot([x_end + 1.2], [y_lab], marker="o", markersize=5, color=color, clip_on=False)
        ax.text(x_end + 2.6, y_lab, label, va="center", fontsize=9, color=style.INK, clip_on=False)

    ax.set_xlim(window[0] / 1e6, window[1] / 1e6)
    ax.set_xlabel("Tokens seen (millions)")
    ax.set_ylabel("Training loss (one batch)")
    fig.text(0.125, 0.95, "One batch, same spike, every run", fontsize=12, fontweight="bold", va="top")
    fig.text(0.125, 0.895,
             "Different architectures, run lengths and GPU counts — only the order of the training data is shared.",
             fontsize=9.5, color=style.INK_SECONDARY, va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=len(RUNS), fontsize=8.5)
    fig.subplots_adjust(top=0.83, right=0.70, bottom=0.25)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
