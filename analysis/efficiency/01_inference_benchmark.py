"""Inference cost of Model A vs Model B: generation speed with and without the KV cache, prefill throughput, and
KV-cache memory (measured, and from the formula, up to the full context).

Standalone on purpose: eval timings mix in data loading and harness overhead. Every model runs on the same device
and dtype; each timing has warm-up runs, synchronizes the GPU before reading the clock, and reports the median.

    uv run --extra analysis python analysis/efficiency/01_inference_benchmark.py [--refresh] [--repeats 3] [--tag h100]
Writes data/inference.csv, data/kv_cache.csv, figures/inference_speed.png, figures/kv_cache.png. With --tag, the
timing CSV and speed figure get that suffix (inference_h100.csv), so measurements from several machines sit side by
side; KV-cache memory doesn't depend on the machine.
"""
import argparse
import csv
import os
import statistics
import sys
import time

import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # analysis/, for common
from common import models, run_ids, runs, style  # noqa: E402

DATA, FIGURES = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
CONFIG_JSON = os.path.join(DATA, "run_configs.json")
# Choices: which runs, and the measurement grid.
RUNS = [("124M GPT-2 (Model A)", run_ids.HEADLINE_A), ("113M Gemma-style (Model B)", run_ids.HEADLINE_B)]
PROMPT_LENS = (128, 512)  # decode: prompt length before generating
NEW_TOKENS = 256  # decode: tokens generated per run (prompt + new must fit in block_size)
PREFILL_LENS = (128, 512, 1024)  # prefill: one forward pass over this many tokens
BATCHES = (1, 8)


def sync(device):
    """Wait for queued GPU work, so the clock measures compute, not just the launch."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def median_seconds(fn, device, repeats, warmup):
    warmup()  # first calls pay one-off costs (kernel compilation, allocator growth); a short call is enough
    times = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)  # without this the clock stops before the GPU finishes
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def kv_bytes_measured(cache):
    return sum(t.numel() * t.element_size() for t in cache.k + cache.v if t is not None)


@torch.no_grad()
def benchmark(label, model, device, repeats):
    """(timing rows, kv rows) for one model."""
    config, vocab = model.config, model.config.vocab_size
    rows, kv_rows = [], []
    for batch in BATCHES:
        for p in PROMPT_LENS:  # decode: generate NEW_TOKENS after a p-token prompt
            prompt = torch.randint(0, vocab, (batch, p), device=device)
            for use_cache in (False, True):
                s = median_seconds(lambda: model.generate(prompt, NEW_TOKENS, top_k=1, use_cache=use_cache), device,
                                   repeats, warmup=lambda: model.generate(prompt, 4, top_k=1, use_cache=use_cache))
                rows.append(dict(model=label, phase="decode", batch=batch, tokens=p, kv_cache=use_cache,
                                 seconds=s, tokens_per_s=batch * NEW_TOKENS / s,  # generated tokens per second
                                 ms_per_token=1000 * s / NEW_TOKENS))
        for t in PREFILL_LENS:  # prefill: one forward pass over t tokens
            x = torch.randint(0, vocab, (batch, t), device=device)
            s = median_seconds(lambda: model(x), device, repeats, warmup=lambda: model(x))
            rows.append(dict(model=label, phase="prefill", batch=batch, tokens=t, kv_cache="", seconds=s,
                             tokens_per_s=batch * t / s, ms_per_token=""))
        print(f"  {label}: batch {batch} done")

    bytes_per_value = next(model.parameters()).element_size()
    for t in range(64, config.block_size + 1, 64):  # memory: cache size as the context fills
        cache = model.new_kv_cache(1, device)
        model(torch.randint(0, vocab, (1, t), device=device), kv_cache=cache)  # prefill t positions
        kv_rows.append(dict(model=label, seq_len=t, measured_mb=kv_bytes_measured(cache) / 1e6,
                            formula_mb=models.kv_bytes_formula(config, t, 1, bytes_per_value) / 1e6))
    return rows, kv_rows


def read_csv(path):
    """Rows back from a saved CSV, numbers and booleans restored."""
    def parse(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    with open(path) as f:
        return [{k: parse(v) for k, v in r.items()} for r in csv.DictReader(f)]


def plot(rows, kv_rows, suffix=""):
    style.apply()
    labels = [label for label, _ in RUNS]
    colors = dict(zip(labels, style.SERIES))
    def speed(phase, label, batch, tokens, use_cache=""):
        return next(r["tokens_per_s"] for r in rows if r["phase"] == phase and r["model"] == label
                    and r["batch"] == batch and r["tokens"] == tokens and r["kv_cache"] == use_cache)

    # decode: one panel per batch size, bars per model × cache setting, the cache's speed-up written above each pair
    fig, axes = plt.subplots(1, len(BATCHES) + 1, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    width = 0.2
    for ax, batch in zip(axes, BATCHES):
        for k, (label, use_cache) in enumerate((l, c) for l in labels for c in (False, True)):
            ys = [speed("decode", label, batch, p, use_cache) for p in PROMPT_LENS]
            xs = [i + (k - 1.5) * width for i in range(len(PROMPT_LENS))]
            ax.bar(xs, ys, width * 0.92, color=colors[label], alpha=1.0 if use_cache else 0.35,
                   label=f"{label.split(' (')[1][:-1]}, {'with' if use_cache else 'no'} cache")
            if use_cache:  # speed-up over the same model without the cache, above the cached bar
                for x, y, p in zip(xs, ys, PROMPT_LENS):
                    ax.text(x, y, f"{y / speed('decode', label, batch, p, False):.1f}×", ha="center", va="bottom",
                            fontsize=8.5, color=style.INK)
        ax.set_xticks(range(len(PROMPT_LENS)), [f"{p}-token prompt" for p in PROMPT_LENS])
        ax.set_title(f"Generation, batch {batch}", fontsize=10, loc="left")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)  # headroom for the speed-up labels
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"Generated tokens / s ({NEW_TOKENS} new)")

    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.06, right=0.99, wspace=0.22)  # before legends read positions

    # prefill: reading a prompt in one forward pass — the cache plays no part, so one bar per model
    ax = axes[-1]
    batch = max(BATCHES)
    for k, label in enumerate(labels):
        ys = [speed("prefill", label, batch, t) / 1000 for t in PREFILL_LENS]  # thousands of tokens / s
        xs = [i + (k - 0.5) * 0.35 for i in range(len(PREFILL_LENS))]
        ax.bar(xs, ys, 0.35 * 0.92, color=colors[label], label=label.split(' (')[1][:-1])
    ax.set_xticks(range(len(PREFILL_LENS)), [f"{t} tokens" for t in PREFILL_LENS])
    ax.set_title(f"Prefill (reading the prompt), batch {batch}", fontsize=10, loc="left")
    ax.set_ylabel("Prompt tokens / s (thousands)")
    ax.grid(axis="x", visible=False)
    # its own legend: prefill has no cache setting
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, fontsize=8.5, frameon=False)

    # generation legend under the two generation panels only
    left, right = axes[0].get_position().x0, axes[len(BATCHES) - 1].get_position().x1
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=((left + right) / 2, 0.0),
               ncol=4, fontsize=8.5, frameon=False)
    fig.text(0.01, 0.99, "Inference speed with and without the KV cache, and prefill", fontsize=12, fontweight="bold",
             va="top")
    fig.text(0.01, 0.93, "Labels: speed-up from the cache for the same model.", fontsize=9,
             color=style.INK_SECONDARY, va="top")
    fig.savefig(os.path.join(FIGURES, f"inference_speed{suffix}.png"), dpi=200)
    if suffix:  # the KV-cache figure is the same on every machine
        return

    # KV memory vs context: formula as a line, measurements as dots
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in labels:
        pts = [r for r in kv_rows if r["model"] == label]
        ax.plot([r["seq_len"] for r in pts], [r["formula_mb"] for r in pts], color=colors[label], label=label)
        ax.scatter([r["seq_len"] for r in pts], [r["measured_mb"] for r in pts], color=colors[label], s=10, zorder=3)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("KV cache per sequence (MB)")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.text(0.01, 0.99, "KV-cache memory: GQA and local windows", fontsize=12, fontweight="bold", va="top")
    fig.text(0.01, 0.93, "Lines: formula. Dots: measured from the cache tensors.", fontsize=9,
             color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.11, right=0.97)
    fig.savefig(os.path.join(FIGURES, "kv_cache.png"), dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-pull run configs from wandb")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per measurement (median is reported)")
    parser.add_argument("--plot-only", action="store_true", help="redraw the figures from the saved CSVs")
    parser.add_argument("--tag", default="", help="machine label for the timing files, e.g. h100 → inference_h100.csv")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    if args.plot_only:
        plot(*(read_csv(os.path.join(DATA, n)) for n in (f"inference{suffix}.csv", "kv_cache.csv")), suffix)
        return
    paths = [p for _, p in RUNS]
    if args.refresh or not os.path.exists(CONFIG_JSON):
        runs.fetch_configs(paths, ["out_dir"], CONFIG_JSON)
    configs = runs.load_configs(paths, CONFIG_JSON)
    device = models.device()

    rows, kv_rows = [], []
    for label, path in RUNS:
        model = models.load_model(configs[path]["out_dir"], device)
        r, k = benchmark(label, model, device, args.repeats)
        rows += r
        kv_rows += k
        del model
    for name, data in ((f"inference{suffix}.csv", rows), ("kv_cache.csv", kv_rows)):
        with open(os.path.join(DATA, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    plot(rows, kv_rows, suffix)
    print(f"device {device}")
    for r in rows:
        if r["phase"] == "decode":
            print(f"{r['model']}, batch {r['batch']}, prompt {r['tokens']}, cache {r['kv_cache']}: "
                  f"{r['tokens_per_s']:.1f} tok/s, {r['ms_per_token']:.1f} ms/token")
    for r in rows:
        if r["phase"] == "prefill":
            print(f"{r['model']}, batch {r['batch']}, prefill {r['tokens']}: {r['tokens_per_s']:.0f} tok/s")
    for label, _ in RUNS:
        last = [r for r in kv_rows if r["model"] == label][-1]
        print(f"{label}: KV cache at {last['seq_len']} tokens: measured {last['measured_mb']:.2f} MB, "
              f"formula {last['formula_mb']:.2f} MB")


if __name__ == "__main__":
    main()
