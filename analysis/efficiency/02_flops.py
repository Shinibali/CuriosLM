"""Training FLOPs of Model A vs Model B, counted layer by layer, and the MFU of their headline runs.

Forward FLOPs per token: every matrix multiply costs 2 × (inputs × outputs) per token, read from the model's own
nn.Linear shapes (attention projections, MLP, output layer). Attention itself adds 4 × d × keys per layer (query·key
scores, then the weighted sum of values), where `keys` is how many earlier tokens a query attends to, averaged over the
window: (T + 1) / 2 for a causal global layer, fewer for a local layer. Training costs 3× the forward pass (the
backward pass computes gradients for both activations and weights). Norms, activations and RoPE are elementwise:
few FLOPs, but they read and write memory, which is why they cost time without showing up here.

MFU is achieved FLOP/s over the GPUs' peak. It is reported two ways: with the PaLM formula the training script logs
(6N + 12·L·d·T per token: full attention over every position, no causal or window saving), and with the useful FLOPs
counted here.

    uv run --extra analysis python analysis/efficiency/02_flops.py [--refresh]
Needs both checkpoints (for the architecture) and wandb (first run, or --refresh: throughput, batch settings, GPU).
Writes data/flops.csv and figures/flops.png.
"""
import argparse
import csv
import json
import os
import statistics
import sys

import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.dirname(HERE), ROOT]  # analysis/ for common, the repo root for curioslm
from common import models, run_ids, runs, style  # noqa: E402
from curioslm import checkpoints, runtime  # noqa: E402
from curioslm.config import GPTConfig  # noqa: E402
from curioslm.model import GPT  # noqa: E402

DATA, FIGURES = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
RUNS_JSON = os.path.join(DATA, "flops_runs.json")
OUT_CSV = os.path.join(DATA, "flops.csv")
OUT_PNG = os.path.join(FIGURES, "flops.png")
# Choices: which runs (the headline pair), and the components the figure separates
RUNS = [("Model A (GPT-2)", run_ids.HEADLINE_A), ("Model B (Gemma-style)", run_ids.HEADLINE_B)]
COMPONENTS = ("attention projections", "attention scores", "MLP", "output layer")
CONFIG_KEYS = ("out_dir", "batch_size", "gradient_accumulation_steps", "block_size", "max_iters")


def fetch(refresh):
    """Per run: its config values, GPU count and model, and median logged throughput and MFU (cached in data/)."""
    if os.path.exists(RUNS_JSON) and not refresh:
        with open(RUNS_JSON) as f:
            return json.load(f)
    import wandb
    api, out = wandb.Api(), {}
    for _, path in RUNS:
        run = api.run(f"{runs.WANDB_ENTITY}/{path}")
        logged = [r for r in run.scan_history(keys=["iter", "throughput/tokens_per_sec", "mfu"])
                  if r["iter"] > 0 and r["throughput/tokens_per_sec"] is not None and r["mfu"] is not None]
        out[path] = {**{k: run.config[k] for k in CONFIG_KEYS}, "gpus": run.metadata["gpu_count"],
                     "gpu": run.metadata["gpu"],
                     "tokens_per_sec": statistics.median(r["throughput/tokens_per_sec"] for r in logged),
                     "logged_mfu": statistics.median(r["mfu"] for r in logged if r["mfu"] > 0)}
    os.makedirs(DATA, exist_ok=True)
    with open(RUNS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    return out


def architecture(out_dir):
    """The model a run trained, rebuilt from its checkpoint's saved architecture on the meta device (shapes only)."""
    path = os.path.join(ROOT, out_dir, "ckpt.pt")
    assert os.path.exists(path), f"{path} missing: run 01_inference_benchmark.py first (it downloads checkpoints)"
    config = GPTConfig(**checkpoints.load(path, map_location="cpu")["model_args"])
    with torch.device("meta"):
        return GPT(config), config


def forward_flops(model, config):
    """Forward FLOPs per token, by component, averaged over the positions of a full context window."""
    flops = dict.fromkeys(COMPONENTS, 0.0)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            part = ("output layer" if name == "lm_head" else "MLP" if ".mlp." in name else "attention projections")
            flops[part] += 2 * module.in_features * module.out_features  # one multiply-add per weight per token
    T, d = config.block_size, config.n_embd  # d = n_head × head_size: query heads, even under GQA
    for layer in range(config.n_layer):
        if config.is_global_layer(layer):
            keys = (T + 1) / 2  # causal: position p sees p + 1 keys; averaged over p = 0..T-1
        else:
            W = config.attention_window
            keys = sum(min(p + 1, W) for p in range(T)) / T  # local: at most W keys (q − W < k ≤ q)
        flops["attention scores"] += 4 * d * keys  # q·k for each key, then the weighted sum of v: 2 × 2 × d × keys
    return flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-pull throughput and settings from wandb")
    args = parser.parse_args()
    info = fetch(args.refresh)
    rows = []
    for label, path in RUNS:
        run = info[path]
        model, config = architecture(run["out_dir"])
        fwd = forward_flops(model, config)
        symbolic = models.forward_flops_formula(config)
        for part in COMPONENTS:  # the formula and the layer-shape count must agree, component by component
            assert abs(symbolic[part] - fwd[part]) < 1, f"{part}: formula {symbolic[part]:,.0f} vs count {fwd[part]:,.0f}"
        train_per_token = 3 * sum(fwd.values())  # backward ≈ 2 × forward
        n_params = sum(p.numel() for p in model.parameters())
        n_logged = model.get_num_params()  # what the logged MFU formula uses (no position table)
        L, T, d = config.n_layer, config.block_size, config.n_embd
        palm_per_token = 6 * n_logged + 12 * L * d * T  # the training script's estimate_mfu formula
        tokens = run["max_iters"] * run["batch_size"] * run["gradient_accumulation_steps"] * T
        peak = runtime.peak_flops_for_name(run["gpu"]) * run["gpus"]
        rows.append(dict(
            model=label, params_m=n_params / 1e6, **{f"fwd_{k.replace(' ', '_')}_mflops": v / 1e6 for k, v in fwd.items()},
            fwd_mflops_per_token=sum(fwd.values()) / 1e6, train_mflops_per_token=train_per_token / 1e6,
            six_n_mflops_per_token=6 * n_params / 1e6, palm_mflops_per_token=palm_per_token / 1e6,
            tokens_b=tokens / 1e9, train_flops_e18=train_per_token * tokens / 1e18,
            gpus=run["gpus"], gpu=run["gpu"], tokens_per_sec=run["tokens_per_sec"],
            mfu_useful=train_per_token * run["tokens_per_sec"] / peak,
            mfu_palm=palm_per_token * run["tokens_per_sec"] / peak, mfu_logged=run["logged_mfu"] / 100,  # logged in percent
            gpu_hours=tokens / run["tokens_per_sec"] / 3600 * run["gpus"],
        ))
    os.makedirs(DATA, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']}: {r['params_m']:.1f}M params")
        for k in COMPONENTS:
            v = r[f"fwd_{k.replace(' ', '_')}_mflops"]
            print(f"  forward {k}: {v:.1f} MFLOP/token ({v / r['fwd_mflops_per_token']:.1%})")
        print(f"  forward total {r['fwd_mflops_per_token']:.1f} (the formula agrees exactly), training {r['train_mflops_per_token']:.1f} MFLOP/token "
              f"(6N: {r['six_n_mflops_per_token']:.1f}, PaLM formula: {r['palm_mflops_per_token']:.1f})")
        print(f"  {r['tokens_b']:.2f}B tokens → {r['train_flops_e18']:.1f}e18 FLOPs; {r['tokens_per_sec'] / 1e6:.2f}M tok/s "
              f"on {r['gpus']}× {r['gpu']} → MFU {r['mfu_useful']:.1%} useful, {r['mfu_palm']:.1%} PaLM formula "
              f"(logged median {r['mfu_logged']:.1%}); {r['gpu_hours']:.1f} GPU-hours")

    style.apply()
    fig, ax = plt.subplots(figsize=(8, 2.8))
    shades = [style.COMPONENTS[k] for k in COMPONENTS]  # one colour per component, never the model colours
    for y, r in enumerate(reversed(rows)):  # Model A on top
        left = 0.0
        for k, color in zip(COMPONENTS, shades):
            v = r[f"fwd_{k.replace(' ', '_')}_mflops"]
            ax.barh(y, v, left=left, height=0.55, color=color, edgecolor=style.SURFACE, linewidth=2,
                    label=k if y == 0 else None)
            left += v
        ax.text(left, y, f"  {left:.0f} MFLOP", va="center", fontsize=9, color=style.INK)
    ax.set_yticks(range(len(rows)), [r["model"] for r in reversed(rows)])
    ax.set_xlabel("Forward FLOPs per token (millions); training ≈ 3×")
    ax.set_xlim(0, max(r["fwd_mflops_per_token"] for r in rows) * 1.2)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=8.5)
    fig.text(0.02, 0.97, "Where the FLOPs go, per token", fontsize=12, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.36)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_CSV}, {OUT_PNG}")


if __name__ == "__main__":
    main()
