"""How far each layer looks, and how much of its attention rests on the first token, for every head of both
models.

Both models read the same full-length windows of the held-out FineWeb-Edu stream, pages packed back to back as in
training. For every head, two numbers, averaged over the queries in the second half of each window (so every query
could look at least half a window back, and a local window is the only thing that can stop it):
- attention distance: the expected distance from a query back to the keys it attends to, Σₖ p(k)·(q − k), the
  measure Dosovitskiy et al. (2020) used to show vision transformers' early layers looking locally and later ones
  globally;
- first-token share: the attention on position 0, the window's first token, which many heads use as a resting place
  when nothing else needs attention whatever that token is (the "attention sink" of Xiao et al., 2023). A query more
  than a window past position 0 can't reach it at all, which is the case for Model B's local layers here;
- separator share: the attention on document-separator (EOT) tokens anywhere before the query, averaged over the
  windows that contain one. A sink can also attach to a token rather than a position: a model with only relative
  positions (RoPE) has no way to tell position 0 apart, but it can recognise the separator token.

    uv run --extra analysis python analysis/attention/03_attention_distance.py [--windows 32] [--plot-only]
Needs both headline checkpoints and the validation shard. Writes data/attention_distance.npz and
figures/attention_distance.png.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import attention  # this study's shared choices; also puts analysis/ on sys.path
from common import models, style  # noqa: E402
from common import perplexity as ppl  # noqa: E402

OUT_NPZ = os.path.join(attention.DATA, "attention_distance.npz")
OUT_PNG = os.path.join(attention.FIGURES, "attention_distance.png")
WINDOWS = 32  # held-out windows of block_size tokens: ~33k tokens, every head averaged over ~16k queries
MODEL_COLORS = {"Model A": style.MODEL_A, "Model B": style.MODEL_B}


@torch.no_grad()
def head_statistics(model, windows, device):
    """(3, layers, heads): mean attention distance, first-token share and separator share, over the second half of
    every window's queries; the separator share only over windows that contain a separator."""
    length = model.config.block_size
    queries = torch.arange(length // 2, length, device=device)  # second half: each can look half a window back
    keys = torch.arange(length, device=device)
    distance = (queries[:, None] - keys[None, :]).clamp(min=0).float()  # (queries, keys); future keys: weight 0
    totals = np.zeros((3, model.config.n_layer, model.config.n_head))
    with_separator = 0  # windows containing at least one separator
    for window in windows:
        idx = torch.from_numpy(window)[None].to(device)
        is_separator = torch.from_numpy(window == ppl.EOT).to(device)  # (keys,)
        with_separator += bool(is_separator.any())
        for layer, probabilities in enumerate(model.attention_weights(idx)):
            rows = probabilities[0][:, queries].float()  # (heads, queries, keys)
            totals[0, layer] += (rows * distance).sum(-1).mean(-1).cpu().numpy()  # expected distance per query
            totals[1, layer] += rows[..., 0].mean(-1).cpu().numpy()  # attention on position 0
            totals[2, layer] += rows[..., is_separator].sum(-1).mean(-1).cpu().numpy()  # on every separator
    totals[:2] /= len(windows)
    totals[2] /= max(with_separator, 1)  # windows without a separator would only dilute this share
    return totals


def plot(stats, global_layers, window):
    style.apply()
    fig, (distance_ax, sink_ax, separator_ax) = plt.subplots(1, 3, figsize=(17, 4.4), layout="constrained")
    for offset, (name, (distance, sink, separator)) in zip((-0.15, 0.15), stats.items()):
        layers = np.arange(distance.shape[0])
        for ax, values in ((distance_ax, distance), (sink_ax, sink), (separator_ax, separator)):
            ax.scatter(np.repeat(layers + offset, values.shape[1]), values.ravel(), s=12, alpha=0.6,
                       color=MODEL_COLORS[name], label=name)  # one dot per head
            ax.plot(layers + offset, values.mean(axis=1), color=MODEL_COLORS[name], linewidth=1.2)  # layer mean
    distance_ax.axhline(window - 1, color=style.INK, linewidth=1, linestyle="--", zorder=1)
    distance_ax.text(1, window - 1, f"Model B's local window ({window} tokens)", fontsize=8.5, va="bottom", ha="right", transform=distance_ax.get_yaxis_transform())  # right end, clear of the legend
    for ax in (distance_ax, sink_ax, separator_ax):
        ax.set_xticks(range(len(next(iter(stats.values()))[0])))
        ax.set_xlabel("Layer")
        for layer in global_layers:  # Model B's global layers
            ax.axvspan(layer - 0.4, layer + 0.4, color=style.GRID, zorder=0)
    distance_ax.set_ylabel("Mean attention distance (tokens back)")
    distance_ax.set_title("How far each head looks", fontsize=10)
    sink_ax.set_ylabel("Share of attention on position 0")
    sink_ax.set_title("Attention resting on the first position", fontsize=10)
    separator_ax.set_ylabel("Share of attention on separator tokens")
    separator_ax.set_title("Attention resting on document separators", fontsize=10)
    top = max(sink_ax.get_ylim()[1], separator_ax.get_ylim()[1])  # one scale for both sink panels, nothing clipped
    for ax in (sink_ax, separator_ax):
        ax.set_ylim(top=top)
    handles, labels = distance_ax.get_legend_handles_labels()
    distance_ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, 0.92), fontsize=8.5)  # under the line
    fig.suptitle("Attention by layer: every head of both models, on full-length held-out windows (shaded: Model B's "
                 "global layers)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    os.makedirs(attention.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=WINDOWS)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        device = models.device()
        loaded = {name: attention.load(name, device) for name in attention.MODELS}
        config_b = loaded["Model B"].config
        length = config_b.block_size
        dataset = attention.config(attention.MODELS["Model B"])["dataset"]
        stream = ppl.val_tokens(dataset, args.windows * length)
        windows = stream[:args.windows * length].reshape(args.windows, length)
        stats = {name: head_statistics(model, windows, device) for name, model in loaded.items()}
        os.makedirs(attention.DATA, exist_ok=True)
        np.savez(OUT_NPZ, window=config_b.attention_window,
                 global_layers=[layer for layer in range(config_b.n_layer) if config_b.is_global_layer(layer)],
                 **{name.replace(" ", "_"): value for name, value in stats.items()})
    saved = np.load(OUT_NPZ)
    stats = {name: saved[name.replace(" ", "_")] for name in attention.MODELS}
    for name, (distance, sink, separator) in stats.items():
        print(f"{name}: mean distance by layer {np.round(distance.mean(axis=1)).astype(int).tolist()}; "
              f"first-token share by layer {np.round(sink.mean(axis=1), 2).tolist()}; "
              f"separator share by layer {np.round(separator.mean(axis=1), 2).tolist()}")
    plot(stats, saved["global_layers"].tolist(), int(saved["window"]))
    print(f"wrote {OUT_NPZ}, {OUT_PNG}")


if __name__ == "__main__":
    main()
