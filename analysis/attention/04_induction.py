"""Induction heads in both models, with the repeat closer than Model B's local window and further than it.

An induction head completes a repeated sequence: at a token it has seen before, it attends to the token that came
right after the earlier occurrence, so the model can copy it (Olsson et al., 2022). The standard test uses random
tokens, so nothing but the repeat can predict them: a block of random tokens, random filler, then the same block again.
The repeat distance (block plus filler) is set below and above Model B's local window, so a local layer can see the
earlier copy in the first case and can't in the second.

Per head, the induction score is the attention from each token of the second copy to the token after its match in
the first copy, averaged. Per model, the loss on the second copy shows whether the copying works at all: random tokens
cost more than a uniform guess (ln 50,257 ≈ 10.8 nats) on first sight, since a trained model's predictions are
confidently wrong on them, and close to 0 if the model copies.

Static figures: each model's strongest induction head on a short repeated sequence, token to token; and induction
score per head and layer at both distances, with the loss on each copy.

    uv run --extra analysis python analysis/attention/04_induction.py [--sequences 16] [--plot-only]
Needs both headline checkpoints. Writes data/induction.npz, figures/induction.png and figures/induction_heads.png.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

import attention  # this study's shared choices; also puts analysis/ on sys.path
from common import models, style  # noqa: E402
from curioslm.loaders.dataset import EOT  # noqa: E402

OUT_NPZ = os.path.join(attention.DATA, "induction.npz")
OUT_PNG = os.path.join(attention.FIGURES, "induction.png")
OUT_HEADS_PNG = os.path.join(attention.FIGURES, "induction_heads.png")
SEQUENCES = 16  # random sequences per distance, averaged
BLOCK = 64  # random tokens repeated
SHARE_OF_WINDOW = (0.5, 1.5)  # repeat distances as multiples of Model B's local window: inside it, and beyond it
SHORT_BLOCK = 12  # a short repeated block for the token-to-token figure, readable token by token
SEED = 0
STRONG = 0.3  # an induction score counted as a clear induction head in the printout: 30% of attention on one token
MODEL_COLORS = {"Model A": style.MODEL_A, "Model B": style.MODEL_B}


def repeated_sequences(count, distance, generator):
    """(count, 1 + BLOCK + filler + BLOCK) token ids: EOT, a random block, random filler, the same block again, so
    each token of the second copy sits `distance` positions after its match in the first."""
    filler = distance - BLOCK
    block = torch.randint(0, EOT, (count, BLOCK), generator=generator)  # any token but the document separator
    middle = torch.randint(0, EOT, (count, filler), generator=generator)
    start = torch.full((count, 1), EOT)
    return torch.cat([start, block, middle, block], dim=1)


@torch.no_grad()
def score(model, sequences, distance, device):
    """Induction score per (layer, head), and the mean loss on the first and the second copy of the block."""
    first = torch.arange(1, 1 + BLOCK)  # positions of the first copy
    second = first + distance  # their repeats
    scores = np.zeros((model.config.n_layer, model.config.n_head))
    losses = np.zeros(2)
    for sequence in sequences:
        idx = sequence[None].to(device)
        logits, _ = model(idx, idx)  # targets given: logits at every position
        token_loss = F.cross_entropy(logits[0, :-1].float(), idx[0, 1:], reduction="none").cpu()  # loss of token t+1
        losses += [token_loss[first[1:] - 1].mean().item(), token_loss[second[1:] - 1].mean().item()]
        for layer, probabilities in enumerate(model.attention_weights(idx)):
            # from each query token of the second copy (but its last) to the key token after its match in the first copy
            on_target = probabilities[0][:, second[:-1], first[1:]].float()  # (heads, BLOCK - 1)
            scores[layer] += on_target.mean(-1).cpu().numpy()
    return scores / len(sequences), losses / len(sequences)


@torch.no_grad()
def strongest_heads(loaded, data, generator, device):
    """Each model's strongest induction head (at the shorter distance) and its attention on a short repeated block:
    adds short_block (token ids) and, per model, <model>_head (layer, head) and <model>_head_attention (T, T)."""
    ids = torch.cat([torch.tensor([EOT]), *[torch.randint(0, EOT, (SHORT_BLOCK,), generator=generator)] * 2])
    data["short_block"] = ids.numpy()
    for name, model in loaded.items():
        key = name.replace(" ", "_")
        scores = data[f"{key}_scores"][0]  # the shorter distance
        layer, head = np.unravel_index(scores.argmax(), scores.shape)
        data[f"{key}_head"] = np.array([layer, head])
        data[f"{key}_head_attention"] = model.attention_weights(ids[None].to(device))[layer][0, head].float().cpu()\
            .numpy()


def head_title(data, name):
    key = name.replace(" ", "_")
    layer, head = data[f"{key}_head"].tolist()
    return f"{name}: layer {layer}, head {head} (induction score {data[f'{key}_scores'][0][layer, head]:.2f})"


def plot_heads(data, enc):
    """Static token-to-token view of each model's strongest induction head on the short repeated block: in the
    second copy, attention jumps to the token after each token's earlier twin, a stripe just off the diagonal."""
    labels = attention.token_labels(enc, data["short_block"].tolist())
    style.apply()
    fig, axes = plt.subplots(1, len(attention.MODELS), figsize=(6 * len(attention.MODELS), 6.2), layout="constrained")
    for column, (ax, name) in enumerate(zip(axes, attention.MODELS)):
        image = attention.draw_token_attention(ax, data[f"{name.replace(' ', '_')}_head_attention"], labels,
                                               show_rows=column == 0)
        ax.set_title(head_title(data, name), fontsize=10)
    fig.colorbar(image, ax=axes, shrink=0.6, label="Attention")
    fig.suptitle(f"Each model's strongest induction head on {SHORT_BLOCK} random tokens repeated (rows: the token "
                 "looking)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(OUT_HEADS_PNG, dpi=200)


def plot(data):
    distances, window = data["distances"].tolist(), int(data["window"])
    style.apply()
    fig, axes = plt.subplots(1, len(distances) + 1, figsize=(5 * (len(distances) + 1), 4.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1.4] * len(distances) + [1]})
    for column, distance in enumerate(distances):
        ax = axes[column]
        for offset, name in zip((-0.15, 0.15), attention.MODELS):
            scores = data[f"{name.replace(' ', '_')}_scores"][column]  # (layers, heads)
            layers = np.arange(scores.shape[0])
            ax.scatter(np.repeat(layers + offset, scores.shape[1]), scores.ravel(), s=12, alpha=0.6,
                       color=MODEL_COLORS[name], label=name)  # one dot per head
        for layer in data["global_layers"]:  # Model B's global layers
            ax.axvspan(layer - 0.4, layer + 0.4, color=style.GRID, zorder=0)
        ax.set_xticks(range(scores.shape[0]))
        ax.set_xlabel("Layer")
        ax.set_ylim(0, 1)
        where = "inside" if distance < window else "beyond"
        ax.set_title(f"Repeat {distance} tokens back ({where} B's {window}-token window)", fontsize=10)
    axes[0].set_ylabel("Induction score (attention to the copied-from token)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    loss_ax = axes[-1]
    width = 0.35
    for offset, name in zip((-width / 2, width / 2), attention.MODELS):
        losses = data[f"{name.replace(' ', '_')}_losses"]  # (distances, [first copy, second copy])
        loss_ax.bar(np.arange(len(distances)) + offset, losses[:, 1], width, color=MODEL_COLORS[name], label=name)
    first_copy = np.mean([data[f"{name.replace(' ', '_')}_losses"][:, 0] for name in attention.MODELS])
    loss_ax.axhline(first_copy, color=style.INK, linewidth=1, linestyle="--")
    loss_ax.text(-0.5, first_copy, "first copy (unseen)", fontsize=8.5, va="bottom")
    loss_ax.set_xticks(range(len(distances)), [f"{distance} back" for distance in distances])
    loss_ax.set_ylabel("Mean loss on the second copy (nats)")
    loss_ax.set_title("Does the model copy?", fontsize=10)
    fig.suptitle("Induction heads: repeated random tokens, closer and further than Model B's local window (shaded: "
                 "Model B's global layers)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    os.makedirs(attention.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=SEQUENCES)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    enc = tiktoken.get_encoding("gpt2")
    if not args.plot_only:
        device = models.device()
        loaded = {name: attention.load(name, device) for name in attention.MODELS}
        config_b = loaded["Model B"].config
        window = config_b.attention_window
        distances = [int(share * window) for share in SHARE_OF_WINDOW]
        assert 1 + BLOCK + max(distances) <= config_b.block_size, "the second copy must fit in the context"
        generator = torch.Generator().manual_seed(SEED)
        data = {"distances": distances, "window": window,
                "global_layers": [layer for layer in range(config_b.n_layer) if config_b.is_global_layer(layer)]}
        sequences = {distance: repeated_sequences(args.sequences, distance, generator)
                     for distance in distances}  # the same sequences for both models
        for name, model in loaded.items():
            results = [score(model, sequences[distance], distance, device) for distance in distances]
            data[f"{name.replace(' ', '_')}_scores"] = np.stack([scores for scores, _ in results])
            data[f"{name.replace(' ', '_')}_losses"] = np.stack([losses for _, losses in results])
        strongest_heads(loaded, data, generator, device)
        os.makedirs(attention.DATA, exist_ok=True)
        np.savez(OUT_NPZ, **data)
    data = dict(np.load(OUT_NPZ))
    os.makedirs(attention.FIGURES, exist_ok=True)
    for name in attention.MODELS:
        for column, distance in enumerate(data["distances"].tolist()):
            scores = data[f"{name.replace(' ', '_')}_scores"][column]
            layer, head = np.unravel_index(scores.argmax(), scores.shape)
            first, second = data[f"{name.replace(' ', '_')}_losses"][column]
            print(f"{name}, repeat {distance} back: strongest induction head layer {layer} head {head} "
                  f"({scores[layer, head]:.2f}); heads scoring > {STRONG}: {(scores > STRONG).sum()}; "
                  f"loss first copy {first:.2f}, second copy {second:.2f}")
    plot(data)
    plot_heads(data, enc)
    print(f"wrote {OUT_NPZ}, {OUT_PNG}, {OUT_HEADS_PNG}")


if __name__ == "__main__":
    main()
