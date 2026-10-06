"""Real RoPE heads: how Model B's actual heads score keys by distance, measured on held-out text, band by band.

With each RoPE pair written as a complex number z = x₁ + i·x₂ (dimensions i and i + h/2, as model.py pairs them), the
pre-softmax score between a query at position m and a key at position n is
    Σᵢ Re[ zqᵢ(m) · conj(zkᵢ(n)) · e^{i (m − n) θᵢ} ] / √h,
using the queries and keys *before* rotation. Averaging over every query-key pair the same distance Δ apart gives each
band's average contribution at that distance; their sum is the head's average score against distance. The average
over pairs is a cross-correlation of the query and key sequences, computed with one FFT per head.

Heads are chosen from 03's band usage: the one with the most capacity in slow bands, the one with the least, and the
median. Queries and keys are captured with a forward hook on each attention module; the model is unchanged.

    uv run --extra analysis --extra eval python analysis/positions/04_real_heads.py [--windows 64]
Needs Model B's checkpoint, 03's data/band_usage.csv and the validation shard. Writes data/real_heads.npz and
figures/real_heads.png.
"""
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import positions  # this study's shared choices; also puts analysis/ on sys.path
from common import models, style  # noqa: E402
from common import perplexity as ppl  # noqa: E402

OUT_NPZ = os.path.join(positions.DATA, "real_heads.npz")
OUT_PNG = os.path.join(positions.FIGURES, "real_heads.png")


def pick_heads():
    """(layer, head, label): the highest, median and lowest slow-band weight, from 03's band_usage.csv."""
    with open(os.path.join(positions.DATA, "band_usage.csv")) as f:
        rows = sorted(csv.DictReader(f), key=lambda row: float(row["slow_band_share"]))
    picks = [(rows[-1], "highest"), (rows[len(rows) // 2], "median"), (rows[0], "lowest")]
    return [(int(row["layer"]), int(row["head"]), f"{rank} slow-band weight: {float(row['slow_band_share']):.0%}")
            for row, rank in picks]


@torch.no_grad()
def band_scores(model, tokens, num_windows, device, heads):
    """{(layer, head): (bands, context) average score contribution of each band at each distance 0 .. context − 1}."""
    config = model.config
    context, width = config.block_size, config.n_embd
    head_size = width // config.n_head
    num_bands = head_size // 2  # one band per pair of dimensions
    queries_per_kv_head = config.n_head // config.num_kv_heads  # GQA: query heads sharing each key head
    rope_freqs = positions.rope_frequencies(
        {"n_embd": width, "n_head": config.n_head, "rope_theta": config.rope_theta})[0].double()
    attention_inputs = {}  # layer → its attention input, ln_1(x), from the latest forward pass
    hooks = [model.transformer.h[layer].attn.register_forward_pre_hook(
        lambda module, inputs, layer=layer: attention_inputs.__setitem__(layer, inputs[0].detach()))
        for layer in {layer for layer, _, _ in heads}]
    totals = {(layer, head): torch.zeros(num_bands, context, dtype=torch.float64) for layer, head, _ in heads}
    windows = torch.from_numpy(tokens[:num_windows * context]).view(num_windows, context)
    distances = torch.arange(context, dtype=torch.float64)
    pairs_at_distance = torch.arange(context, 0, -1, dtype=torch.float64)[:, None]  # context − Δ pairs at distance Δ
    rotation = torch.exp(1j * distances[:, None] * rope_freqs[None, :])  # e^{iΔθ}: RoPE's turn at each distance
    for window in windows:
        model(window[None].to(device))
        for layer, head, _ in heads:
            attention_input = attention_inputs[layer][0].float().cpu()  # (context, width)
            projection = model.transformer.h[layer].attn.c_attn
            qkv = attention_input @ projection.weight.float().cpu().T + projection.bias.float().cpu()
            query = qkv[:, :width].view(context, config.n_head, head_size)[:, head]  # before rotation
            key = qkv[:, width:width + config.num_kv_heads * head_size].view(
                context, config.num_kv_heads, head_size)[:, head // queries_per_kv_head]
            # cast the parts first: .double() on a complex tensor would drop the imaginary part
            query, key = query.double(), key.double()
            query_pairs = torch.complex(query[:, :num_bands], query[:, num_bands:])  # band i = dims i and i + h/2
            key_pairs = torch.complex(key[:, :num_bands], key[:, num_bands:])
            fft_size = 2 * context  # zero-pad so the FFT's circular correlation equals the linear one
            correlation = torch.fft.ifft(torch.fft.fft(query_pairs, n=fft_size, dim=0)
                                         * torch.fft.fft(key_pairs, n=fft_size, dim=0).conj(), dim=0)[:context]
            mean_product = correlation / pairs_at_distance  # mean of query(n + Δ)·conj(key(n)) over n, per band
            totals[(layer, head)] += (mean_product * rotation).real.T / math.sqrt(head_size)
    for hook in hooks:
        hook.remove()
    return {key: total / num_windows for key, total in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=64, help="held-out windows of block_size tokens")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    heads = pick_heads()
    if not args.plot_only:
        run_config = positions.config(positions.ROPE)
        device = models.device()
        model = models.load_model(run_config["out_dir"], device)
        tokens = ppl.val_tokens(run_config["dataset"], args.windows * model.config.block_size)
        scores = band_scores(model, tokens, args.windows, device, heads)
        np.savez(OUT_NPZ, **{f"L{layer}H{head}": score.numpy() for (layer, head), score in scores.items()},
                 windows=args.windows, window=model.config.attention_window, block=model.config.block_size,
                 global_layers=[layer for layer in range(model.config.n_layer)
                                if model.config.is_global_layer(layer)])
    data = np.load(OUT_NPZ)
    local_window, global_layers = int(data["window"]), set(data["global_layers"].tolist())
    context, num_windows = int(data["block"]), int(data["windows"])
    model_b_args = positions.model_args(positions.out_dir(positions.ROPE))
    wavelengths = (2 * math.pi / positions.rope_frequencies(model_b_args)[0]).tolist()
    groups = positions.speed_groups(wavelengths, context)  # YaRN's thresholds, declared in positions.py
    speed_groups = {
        f"fast bands (≥ {positions.YARN_BETA} turns in {context:,} tokens)": groups["fast"],
        f"middle bands ({positions.YARN_ALPHA}–{positions.YARN_BETA} turns)": groups["middle"],
        f"slow bands (< {positions.YARN_ALPHA} turn: never completes one)": groups["slow"],
    }
    print(f"band speed groups: {', '.join(f'{name} {len(bands)}' for name, bands in groups.items())}")

    style.apply()
    fig, axes = plt.subplots(2, len(heads), figsize=(12, 5.8), sharey="row")
    for column, (layer, head, label) in enumerate(heads):
        band_score = data[f"L{layer}H{head}"]  # (bands, context): each band's contribution at each distance
        span = context if layer in global_layers else local_window  # a local layer never sees beyond its window
        distance = np.arange(span)
        bands_ax, total_ax = axes[0, column], axes[1, column]
        for (group_name, group_bands), color in zip(speed_groups.items(), style.OTHER):
            bands_ax.plot(distance + 1, band_score[group_bands, :span].sum(axis=0), color=color, linewidth=1.5,
                          label=group_name)
        total_ax.plot(distance + 1, band_score[:, :span].sum(axis=0), color=style.MODEL_B, linewidth=2)
        for ax in (bands_ax, total_ax):
            ax.set_xscale("log")  # plotted at Δ + 1 so Δ = 0 fits; ticks show the real distances
            ticks = [tick for tick in (0, 1, 10, 100, 1000) if tick < span]
            ax.set_xticks([tick + 1 for tick in ticks], [str(tick) for tick in ticks])
            ax.minorticks_off()
            ax.axhline(0, color=style.INK_SECONDARY, linewidth=0.8)
        kind = "global" if layer in global_layers else f"local, W = {local_window}"
        bands_ax.set_title(f"Layer {layer}, head {head} ({kind})\n{label}", fontsize=9)
        total_ax.set_xlabel("Distance between query and key (log scale)", fontsize=8.5)
    axes[0, 0].set_ylabel("Contribution to score\nby band speed")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.455), ncol=3, fontsize=8.5)  # in the row gap
    axes[1, 0].set_ylabel("Mean pre-softmax\nscore, q·k / √h")
    fig.text(0.01, 0.99, "Model B attention heads: score vs query–key distance, split by RoPE band speed",
             fontsize=12, fontweight="bold", va="top")
    fig.text(0.01, 0.945, f"Mean over every query–key pair at each distance, {num_windows} held-out FineWeb-Edu "
             f"windows of {context:,} tokens. The top row sums to the bottom row exactly.\nSlow-band weight: the "
             f"head's share of q·k weight in bands that never complete a turn within {context:,} tokens (see the "
             "band-usage figure).", fontsize=9, color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(top=0.8, bottom=0.1, left=0.09, right=0.98, wspace=0.08, hspace=0.62)
    os.makedirs(positions.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    for layer, head, label in heads:
        total = data[f"L{layer}H{head}"].sum(axis=0)
        print(f"layer {layer} head {head} ({label}): score at Δ=0 {total[0]:.2f}, Δ=1 {total[1]:.2f}, "
              f"Δ=10 {total[10]:.2f}, Δ=100 {total[100]:.2f}, Δ=500 {total[500]:.2f}")
    print(f"wrote {OUT_PNG}")

if __name__ == "__main__":
    main()
