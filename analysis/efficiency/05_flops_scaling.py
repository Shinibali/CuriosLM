"""FLOPs at larger scale, by arithmetic alone: the per-component formula checked against the trained models
(02_flops.py), applied to bigger models and longer contexts, plus the training-budget arithmetic the post quotes.

1. flops_by_size.png: each component's *share* of a model's forward FLOPs, GPT-2 small → a 7B-class model, at a
   1,024-token context (absolute values would span ~50×, so shares are the only comparable view).
2. flops_by_context.png: *absolute* forward GFLOPs per token for Model A's and Model B's recipes (Model B's read from
   its checkpoint) at a ~7B and a ~14B size, at 1k and 128k context.
3. The crossovers the formula predicts: attention passes the MLP at T ≈ 8d; the output layer outweighs all the MLPs
   while d·L < V/8.
4. Training-budget facts: the embedding share of parameters, the Chinchilla-optimal size for Model A's compute, and
   how many sequences one bad page competes with per step, here and at GPT-3's batch size.

    uv run --extra analysis python analysis/efficiency/05_flops_scaling.py
Needs 02's data (flops_runs.json, flops.csv) and both checkpoints. Writes data/flops_scaling.csv,
data/flops_scaling_facts.csv, figures/flops_by_size.png, figures/flops_by_context.png.
"""
import contextlib
import csv
import io
import json
import os
import sys

import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.dirname(HERE), ROOT]  # analysis/ for common, the repo root for curioslm
from common import models, style  # noqa: E402
from common.reference_models import FOURTEEN_B, GPT2_FAMILY, LONG_CONTEXT, SEVEN_B, VOCAB  # noqa: E402
from curioslm import checkpoints  # noqa: E402
from curioslm.config import GPTConfig  # noqa: E402
from curioslm.model import GPT  # noqa: E402

DATA, FIGURES = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
PARTS = ("attention projections", "attention scores", "MLP", "output layer")
SHORT_CONTEXT = 1024  # the context of the size comparison and the short end of the context comparison
# Facts quoted from cited sources (used only in the arithmetic below)
CHINCHILLA_TOKENS_PER_PARAM = 20  # Hoffmann et al. 2022
GPT3_BATCH_TOKENS = 3.2e6  # GPT-3's largest batch (Brown et al. 2020)


def param_label(config):
    """A config's total parameter count, e.g. "124M" or "6.7B", from the model built on the meta device (no memory)."""
    with torch.device("meta"), contextlib.redirect_stdout(io.StringIO()):  # GPT prints its size on construction
        count = sum(param.numel() for param in GPT(config).parameters())
    return f"{count / 1e9:.1f}B" if count >= 1e9 else f"{count / 1e6:.0f}M"


def trained_runs():
    """02_flops.py's record of the headline runs: {run path: out_dir, batch settings, GPU, ...}."""
    with open(os.path.join(DATA, "flops_runs.json")) as f:
        return json.load(f)


def model_b_recipe():
    """Model B's recipe as settings that carry to any size, read from its checkpoint."""
    for run in trained_runs().values():
        model_args = checkpoints.load(os.path.join(ROOT, run["out_dir"], "ckpt.pt"), map_location="cpu")["model_args"]
        config = GPTConfig(**model_args)
        if config.use_rope:  # Model B
            return dict(queries_per_kv_head=config.n_head // config.num_kv_heads, mlp_type=config.mlp_type,
                        norm_placement=config.norm_placement, attention_window=config.attention_window,
                        local_global_ratio=config.local_global_ratio)
    raise SystemExit("no RoPE (Model B) checkpoint found among 02_flops.py's runs")


def recipe_config(recipe, size, context):
    """A size (layers, width, heads) built with Model A's recipe (recipe=None) or Model B's (a model_b_recipe())."""
    base = dict(**size, block_size=context, vocab_size=VOCAB)
    if recipe is None:
        return GPTConfig(**base)
    return GPTConfig(**base, n_kv_head=size["n_head"] // recipe["queries_per_kv_head"], mlp_type=recipe["mlp_type"],
                     norm_placement=recipe["norm_placement"], position_encoding="rope",
                     attention_window=min(recipe["attention_window"], context),
                     local_global_ratio=recipe["local_global_ratio"])


def parts_row(panel, label, config):
    parts = models.forward_flops_formula(config)
    total = sum(parts.values())
    return dict(panel=panel, label=label, width=config.n_embd, layers=config.n_layer, context=config.block_size,
                total_gflops=total / 1e9, **{f"{part.replace(' ', '_')}_gflops": parts[part] / 1e9 for part in PARTS},
                **{f"{part.replace(' ', '_')}_share": parts[part] / total for part in PARTS})


def size_figure(colors):
    """flops_by_size.png: component shares from GPT-2 small to a 7B-class model, GPT-2 recipe."""
    sizes = [GPTConfig(n_layer=layers, n_embd=width, n_head=heads, block_size=SHORT_CONTEXT, vocab_size=VOCAB)
             for layers, width, heads in GPT2_FAMILY.values()]
    sizes.append(GPTConfig(**SEVEN_B, block_size=SHORT_CONTEXT, vocab_size=VOCAB))
    rows = [parts_row("size", param_label(config), config) for config in sizes]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottom = [0.0] * len(rows)
    for part, color in zip(PARTS, colors):
        shares = [100 * row[f"{part.replace(' ', '_')}_share"] for row in rows]
        ax.bar(range(len(rows)), shares, bottom=bottom, width=0.7, color=color, edgecolor=style.SURFACE,
               linewidth=1.5, label=part)
        bottom = [b + share for b, share in zip(bottom, shares)]
    ax.set_xticks(range(len(rows)), [row["label"] for row in rows])
    ax.set_xlabel(f"Model parameters (GPT-2 recipe, {SHORT_CONTEXT:,}-token context)")
    ax.set_ylabel("Share of each model's\nforward FLOPs (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8)
    fig.text(0.01, 0.99, f"As models grow with the context fixed at {SHORT_CONTEXT:,} tokens,\nthe MLP takes over and "
             "the output layer fades", fontsize=11.5, fontweight="bold", va="top", linespacing=1.3)
    fig.subplots_adjust(top=0.86, bottom=0.25, left=0.13, right=0.98)
    fig.savefig(os.path.join(FIGURES, "flops_by_size.png"), dpi=200)
    return rows


def context_figure(colors):
    """flops_by_context.png: absolute GFLOPs per token, Model A's vs Model B's recipe, two sizes, two contexts."""
    recipe = model_b_recipe()
    recipes = {"Model A's recipe (GPT-2: full attention)": None,
               f"Model B's recipe (GQA {recipe['queries_per_kv_head']}:1, {recipe['attention_window']}-token windows, "
               f"1 global in {recipe['local_global_ratio']})": recipe}
    rows = []
    fig, axes = plt.subplots(1, len(recipes), figsize=(11, 4.2), sharey=True)
    for ax, (name, model_recipe) in zip(axes, recipes.items()):
        panel_rows = []
        for size in (SEVEN_B, FOURTEEN_B):
            # the model's size at the short context: Model A's position table would otherwise grow with the context
            label = param_label(recipe_config(model_recipe, size, SHORT_CONTEXT))
            for context in (SHORT_CONTEXT, LONG_CONTEXT):
                panel_rows.append(parts_row(name, f"{label} model\n{context // 1024}k context",
                                            recipe_config(model_recipe, size, context)))
        rows += panel_rows
        bottom = [0.0] * len(panel_rows)
        for part, color in zip(PARTS, colors):
            values = [row[f"{part.replace(' ', '_')}_gflops"] for row in panel_rows]
            ax.bar(range(len(panel_rows)), values, bottom=bottom, width=0.65, color=color, edgecolor=style.SURFACE,
                   linewidth=1.5, label=part)
            bottom = [b + value for b, value in zip(bottom, values)]
        for x, (total, row) in enumerate(zip(bottom, panel_rows)):
            ax.text(x, total, f"{total:.0f}\n(output layer {row['output_layer_gflops']:.1f})", ha="center",
                    va="bottom", fontsize=8, color=style.INK)  # the output layer is a hairline here: give its size
        ax.set_xticks(range(len(panel_rows)), [row["label"] for row in panel_rows], fontsize=8.5)
        ax.set_title(name, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Forward GFLOPs per token")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.12)  # room for the two-line totals
    fig.text(0.01, 0.99, "Long context multiplies the work per token; local windows hold most of it back", fontsize=12,
             fontweight="bold", va="top")
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.07, right=0.99, wspace=0.06)
    fig.savefig(os.path.join(FIGURES, "flops_by_context.png"), dpi=200)
    return rows


def training_facts():
    """(fact, value) rows: embedding share by size, the Chinchilla optimum for Model A's compute, batch dilution."""
    facts = []
    for name, (layers, width, _) in GPT2_FAMILY.items():
        embedding = VOCAB * width  # token table (tied with the output layer)
        facts.append((f"embedding share of params, {name}", embedding / (embedding + 12 * layers * width * width)))
    with open(os.path.join(DATA, "flops.csv")) as f:
        model_a_compute = float(next(csv.DictReader(f))["train_flops_e18"]) * 1e18  # 02's Model A training FLOPs
    optimal_params = (model_a_compute / (6 * CHINCHILLA_TOKENS_PER_PARAM)) ** 0.5  # C = 6ND, D = 20N → N = √(C/120)
    facts += [("Chinchilla-optimal params for Model A's compute (millions)", optimal_params / 1e6),
              ("Chinchilla-optimal tokens for Model A's compute (billions)",
               CHINCHILLA_TOKENS_PER_PARAM * optimal_params / 1e9)]
    run = next(iter(trained_runs().values()))
    sequences_per_step = run["batch_size"] * run["gradient_accumulation_steps"]  # global micro-batches × sequences
    facts += [("sequences per optimizer step (ours)", sequences_per_step),
              (f"sequences of {run['block_size']:,} tokens per GPT-3 batch", GPT3_BATCH_TOKENS / run["block_size"])]
    return facts


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    style.apply()
    colors = [style.COMPONENTS[part] for part in PARTS]
    rows = size_figure(colors) + context_figure(colors)
    for row in rows:
        print(f"{row['panel']}, {row['label'].replace(chr(10), ' ')}: {row['total_gflops']:.1f} GFLOP/token, "
              f"attention scores {row['attention_scores_share']:.1%}, output layer {row['output_layer_share']:.1%}")
    small_width, small_layers = GPT2_FAMILY["GPT-2 small (124M)"][1], GPT2_FAMILY["GPT-2 small (124M)"][0]
    print(f"crossovers: attention passes the MLP at T ≈ 8d = {8 * small_width:,} tokens at d = {small_width}, "
          f"{8 * SEVEN_B['n_embd']:,} at d = {SEVEN_B['n_embd']}; the output layer outweighs all MLPs while "
          f"d·L < V/8 = {VOCAB / 8:,.0f} (GPT-2 small: d·L = {small_width * small_layers:,})")
    facts = training_facts()
    for name, value in facts:
        print(f"  {name}: {value:,.3f}")
    with open(os.path.join(DATA, "flops_scaling.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(DATA, "flops_scaling_facts.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fact", "value"])
        writer.writerows(facts)


if __name__ == "__main__":
    main()
