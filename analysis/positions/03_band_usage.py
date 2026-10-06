"""Which RoPE bands do Model B's heads actually use? Read from the trained weights, no data needed.

RoPE pairs dimension i of a head with dimension i + h/2 and rotates the pair at band i's speed. How much a head can
use band i depends on how much its query and key projections put into that pair. For every head: take the rows of
W_q (and of the key projection it shares under GQA) that feed band i's two dimensions, fold in the RMSNorm gain that
precedes attention, and multiply the query's and key's norms. Normalized per head, that's the share of the head's
query-key capacity in each band. A proxy from weights alone: what real text activates can differ.

    uv run --extra analysis python analysis/positions/03_band_usage.py
Needs Model B's checkpoint. Writes data/band_usage.csv and figures/band_usage.png.
"""
import csv
import math
import os

import matplotlib.pyplot as plt
import torch

import positions  # this study's shared choices; also puts analysis/ on sys.path
from common import style  # noqa: E402
from curioslm import checkpoints  # noqa: E402
from curioslm.config import GPTConfig  # noqa: E402

OUT_CSV = os.path.join(positions.DATA, "band_usage.csv")
OUT_PNG = os.path.join(positions.FIGURES, "band_usage.png")


def main():
    out_dir = positions.out_dir(positions.ROPE)
    model_args = positions.model_args(out_dir)  # downloads the checkpoint if missing
    config = GPTConfig(**model_args)
    state_dict = checkpoints.strip_compile_prefix(
        checkpoints.load(os.path.join(positions.ROOT, out_dir, "ckpt.pt"), map_location="cpu")["model"])
    rope_freqs, head_size = positions.rope_frequencies(model_args)
    num_bands = head_size // 2  # one band per pair of dimensions
    queries_per_kv_head = config.n_head // config.num_kv_heads  # GQA: query heads sharing each key head
    width = config.n_embd
    wavelengths = (2 * math.pi / rope_freqs).tolist()
    slow_bands = positions.speed_groups(wavelengths, config.block_size)["slow"]  # fewer than one turn in the context

    def pair_norm(weights):
        """(heads, head dims, input) → (heads, bands): norm of the rows feeding each band's pair of dimensions,
        i and i + h/2."""
        return (weights[:, :num_bands].pow(2).sum(-1) + weights[:, num_bands:].pow(2).sum(-1)).sqrt()

    rows, shares = [], []
    for layer in range(config.n_layer):
        qkv_weight = state_dict[f"transformer.h.{layer}.attn.c_attn.weight"].float()
        norm_gain = state_dict[f"transformer.h.{layer}.ln_1.norm.weight"].float()  # the RMSNorm just before attention
        qkv_weight = qkv_weight * norm_gain[None, :]  # fold the gain into the projection's input columns
        query_weight = qkv_weight[:width].view(config.n_head, head_size, width)  # (heads, head dims, input)
        key_weight = qkv_weight[width:width + config.num_kv_heads * head_size].view(
            config.num_kv_heads, head_size, width)
        query_band_norm, key_band_norm = pair_norm(query_weight), pair_norm(key_weight)  # (heads | kv heads, bands)
        for head in range(config.n_head):
            band_weight = query_band_norm[head] * key_band_norm[head // queries_per_kv_head]  # with its shared key head
            share = band_weight / band_weight.sum()
            shares.append(share)
            slow_share = share[slow_bands].sum().item()
            rows.append(dict(layer=layer, head=head, global_layer=config.is_global_layer(layer),
                             slow_band_share=slow_share, **{f"band_{band}": share[band].item()
                                                             for band in range(num_bands)}))
    os.makedirs(positions.DATA, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    even_spread = len(slow_bands) / num_bands  # slow-band share if a head treated every band alike
    slow_shares = torch.tensor([row["slow_band_share"] for row in rows])
    print(f"slow bands (never turn within {config.block_size}) are {even_spread:.0%} of bands; their share of head "
          f"capacity: median {slow_shares.median():.0%}, range {slow_shares.min():.0%}–{slow_shares.max():.0%}")

    style.apply()
    fig, ax = plt.subplots(figsize=(9, 7))
    matrix = torch.stack(shares).numpy()
    image = ax.imshow(matrix, aspect="auto", cmap=style.SEQUENTIAL_CMAP, interpolation="nearest")
    for layer in range(1, config.n_layer):
        ax.axhline(layer * config.n_head - 0.5, color=style.SURFACE, linewidth=1)
    divider = num_bands - len(slow_bands) - 0.5  # between the last band that turns and the first that doesn't
    ax.axvline(divider, color=style.INK, linewidth=1, linestyle="--")
    ax.text(divider + 0.3, -1.5, f"never complete a turn within {config.block_size:,} tokens →", fontsize=8,
            color=style.INK, va="bottom")
    ax.set_yticks([layer * config.n_head + config.n_head / 2 - 0.5 for layer in range(config.n_layer)],
                  [f"layer {layer}{' (global)' if config.is_global_layer(layer) else ''}"
                   for layer in range(config.n_layer)], fontsize=8)
    ax.set_xlabel("RoPE band, fastest → slowest")
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Share of the head's q·k weight")
    colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.text(0.01, 0.99, "RoPE band usage per head, Model B (from the trained weights)", fontsize=12,
             fontweight="bold", va="top")
    fig.text(0.01, 0.955, f"One row per head ({config.n_head} per layer). Per band: ‖W_q rows‖ · ‖W_k rows‖ feeding "
             "that band's dimension pair,\nwith the pre-attention RMSNorm gain folded in, normalized per head.",
             fontsize=9,
             color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.16, right=0.93)
    os.makedirs(positions.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
