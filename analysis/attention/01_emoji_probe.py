"""The emoji probe: how each model reads two emoji that GPT-2's tokenizer splits into a shared first token and two of
their own.

Both models read the probe from a document boundary (EOT first, as every page starts in training), then continue it
greedily (always the most likely next token) for CONTINUATION tokens, so the continuation is reproducible. Small
models decoded greedily tend to fall into loops, so the script also prints SAMPLED seeded continuations drawn the way
sample.py draws them (its temperature and top-k defaults), to show the same models without the loop. For every
layer, every head's attention probabilities come from the model's explicit attention path (GPT.attention_weights),
over the probe and the model's own continuation. Attention is causal, so the probe's part is the same as reading the
probe alone. Two static figures:
- per model, token-to-token attention in every layer (mean of its heads) over the probe and the first probe-length of
  the continuation, where the copying starts;
- from the token that completes each emoji, how much attention each earlier token gets, layer by layer, averaged over
  the layer's heads.
It also prints each model's top next-token guesses at every emoji token, which shows where each emoji becomes
unambiguous.

    uv run --extra analysis python analysis/attention/01_emoji_probe.py [--plot-only | --text-only]
Needs both headline checkpoints (downloaded if missing). Writes data/emoji_attention.npz, figures/emoji_tokens_a.png,
figures/emoji_tokens_b.png and figures/emoji_attention.png.
"""
import argparse
import os
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch

import attention  # this study's shared choices; also puts analysis/ on sys.path
from common import models, style  # noqa: E402
from curioslm.config import RuntimeConfig, SampleConfig  # noqa: E402
from curioslm.loaders.dataset import EOT  # noqa: E402

OUT_NPZ = os.path.join(attention.DATA, "emoji_attention.npz")
OUT_PNG = os.path.join(attention.FIGURES, "emoji_attention.png")
OUT_TOKENS_PNG = os.path.join(attention.FIGURES, "emoji_tokens_{model}.png")  # one per model: a, b
GRID_COLUMNS = 4  # layers per row in the token-to-token figures: 12 layers -> 3 rows
TOP_GUESSES = 3  # next-token guesses printed per emoji token
CONTINUATION = 50  # tokens generated after the probe, greedily and by sampling
SAMPLED = 2  # sampled continuations per model, seeded so they reproduce


def emoji_spans(enc, ids):
    """[(emoji, [positions])]: runs of consecutive tokens that hold only part of a character, joined back up."""
    spans, current = [], []
    for position, token in enumerate(ids):
        try:
            enc.decode_single_token_bytes(token).decode("utf-8")
            whole = True
        except UnicodeDecodeError:
            whole = False
        if not whole and current and _completes(enc, [ids[p] for p in current]):
            spans.append(current)  # the previous character is complete: this partial token starts a new one
            current = []
        if not whole:
            current.append(position)
        elif current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return [(enc.decode([ids[p] for p in span]).strip(), span) for span in spans]


def emoji_name(emoji):
    """'hugging face' for 🤗: plot fonts have no emoji glyphs, so figures name them instead."""
    return unicodedata.name(emoji[0]).lower()


def _completes(enc, tokens):
    """True if these tokens' bytes decode to whole characters."""
    try:
        b"".join(enc.decode_single_token_bytes(token) for token in tokens).decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


@torch.no_grad()
def probe(model, ids, enc, spans, device):
    """(layers, heads, T, T) attention probabilities, after printing the model's top guesses at every emoji token."""
    idx = torch.tensor([ids], device=device)
    logits, _ = model(idx, idx)  # targets given: logits at every position
    for emoji, span in spans:
        for position in span:
            guesses = logits[0, position].softmax(-1).topk(TOP_GUESSES)
            shown = ", ".join(f"'{attention.token_labels(enc, [token])[0]}' {probability:.0%}"
                              for probability, token in zip(guesses.values.tolist(), guesses.indices.tolist()))
            print(f"  {emoji} token {span.index(position) + 1}/{len(span)}: next-token guesses {shown}")
    return torch.stack([layer[0] for layer in model.attention_weights(idx)]).float().cpu().numpy()


@torch.no_grad()
def greedy_continuation(model, ids, device):
    """The model's CONTINUATION most likely next tokens after ids, one at a time (top-1 sampling)."""
    idx = torch.tensor([ids], device=device)
    return model.generate(idx, CONTINUATION, top_k=1)[0, len(ids):].tolist()


@torch.no_grad()
def sampled_continuations(model, ids, device):
    """SAMPLED continuations of CONTINUATION tokens, drawn with sample.py's temperature and top-k, from a fixed seed."""
    torch.manual_seed(RuntimeConfig().seed)  # sample.py's seed
    sampling = SampleConfig()
    idx = torch.tensor([ids], device=device)
    return [model.generate(idx, CONTINUATION, temperature=sampling.temperature, top_k=sampling.top_k)[0, len(ids):]
            .tolist() for _ in range(SAMPLED)]


def plot_tokens(enc, sequences, weights):
    """Per model, token-to-token attention in every layer (mean of its heads), over the probe and its first copied
    line. Copying shows as a stripe parallel to the diagonal in the lower half."""
    style.apply()
    paths = []
    for name, model_weights in weights.items():
        shown = model_weights.shape[-1]  # the probe and the first probe-length of the continuation
        labels = attention.token_labels(enc, sequences[name][:shown])
        layer_means = model_weights.astype(np.float32).mean(axis=1)  # (layers, T, T)
        rows = -(-len(layer_means) // GRID_COLUMNS)  # ceiling division
        fig, axes = plt.subplots(rows, GRID_COLUMNS, figsize=(4 * GRID_COLUMNS, 4.2 * rows), layout="constrained")
        for layer, ax in enumerate(axes.flat):
            image = attention.draw_token_attention(ax, layer_means[layer], labels,
                                                   show_rows=layer % GRID_COLUMNS == 0,
                                                   show_columns=layer // GRID_COLUMNS == rows - 1)
            ax.set_title(f"Layer {layer}", fontsize=10)
        fig.colorbar(image, ax=axes, shrink=0.5, label="Attention (mean over the layer's heads)")
        fig.suptitle(f"{name}: which tokens each token attends to, every layer, on the probe and its first copied "
                     "line (rows: the token looking)", x=0.01, ha="left", fontsize=12, fontweight="bold")
        path = OUT_TOKENS_PNG.format(model=name.split()[-1].lower())
        fig.savefig(path, dpi=150)
        paths.append(path)
    return paths


def plot(labels, spans, weights):
    """Rows: models; columns: emoji. Each panel: from the emoji's last token, attention to every token up to it,
    per layer, averaged over heads."""
    style.apply()
    fig, axes = plt.subplots(len(weights), len(spans), figsize=(6 * len(spans), 2.6 * len(weights)), squeeze=False,
                             sharex="col")  # one set of token labels per column, under the bottom row
    for row, (name, model_weights) in enumerate(weights.items()):
        for column, (emoji, span) in enumerate(spans):
            query = span[-1]  # the token that completes the emoji
            rows = model_weights[:, :, query, :query + 1].mean(axis=1)  # (layers, keys up to the query)
            ax = axes[row, column]
            image = ax.imshow(rows, cmap=style.SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(query + 1), labels[:query + 1], rotation=90, fontsize=7)
            ax.tick_params(axis="x", labelbottom=row == len(weights) - 1)
            ax.set_yticks(range(len(rows)), [str(layer) for layer in range(len(rows))], fontsize=7)
            ax.grid(False)
            ax.set_title(f"{name}: from the last token of the {emoji_name(emoji)} emoji", fontsize=10)
            ax.set_ylabel("Layer")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01, label="Attention (mean over heads)")
    probe_named = attention.PROBE
    for emoji, _ in spans:
        probe_named = probe_named.replace(emoji, f"[{emoji_name(emoji)}]")
    fig.suptitle(f"Where each model looks as an emoji completes: '{probe_named}'", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    os.makedirs(attention.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--text-only", action="store_true",
                        help="print the greedy and sampled continuations only: no attention, nothing written")
    args = parser.parse_args()
    enc = tiktoken.get_encoding("gpt2")
    ids = [EOT] + enc.encode(attention.PROBE)  # start at a document boundary, as every page does in training
    labels, spans = attention.token_labels(enc, ids), emoji_spans(enc, ids)
    print("tokens: " + " | ".join(labels))
    print("emoji: " + "; ".join(f"{emoji} = tokens {span}, ids {[ids[p] for p in span]}" for emoji, span in spans))
    if args.text_only:  # generation uses the model's normal forward pass, so it needs no attention weights
        device = models.device()
        for name in attention.MODELS:
            model = attention.load(name, device)
            print(f"{name}, greedy: {enc.decode(greedy_continuation(model, ids, device))!r}")
            for number, sample in enumerate(sampled_continuations(model, ids, device), start=1):
                print(f"{name}, sampled {number}: {enc.decode(sample)!r}")
        return
    if not args.plot_only:
        device = models.device()
        saved = {}
        for name in attention.MODELS:
            print(name)
            model = attention.load(name, device)
            sequence = ids + greedy_continuation(model, ids, device)
            saved[f"{name.replace(' ', '_')}_sequence"] = np.array(sequence)
            saved[f"{name.replace(' ', '_')}_sampled"] = np.array(sampled_continuations(model, ids, device))
            # attention over the probe and one probe-length of continuation, where the copying starts; float16 is
            # finer than any figure or view shows
            shown = 2 * len(ids)
            saved[f"{name.replace(' ', '_')}_weights"] = probe(model, sequence, enc, spans, device)[
                :, :, :shown, :shown].astype(np.float16)
        os.makedirs(attention.DATA, exist_ok=True)
        np.savez_compressed(OUT_NPZ, **saved)
    saved = np.load(OUT_NPZ)
    sequences = {name: saved[f"{name.replace(' ', '_')}_sequence"].tolist() for name in attention.MODELS}
    weights = {name: saved[f"{name.replace(' ', '_')}_weights"] for name in attention.MODELS}
    for name, sequence in sequences.items():
        print(f"{name}, greedy: {enc.decode(sequence[len(ids):])!r}")  # the continuation only, as --text-only
        for number, sample in enumerate(saved[f"{name.replace(' ', '_')}_sampled"].tolist(), start=1):
            print(f"{name}, sampled {number}: {enc.decode(sample)!r}")
    os.makedirs(attention.FIGURES, exist_ok=True)
    plot(labels, spans, {name: value[:, :, :len(ids), :len(ids)].astype(np.float32)
                         for name, value in weights.items()})  # the probe part
    token_figures = plot_tokens(enc, sequences, weights)
    print(f"wrote {OUT_NPZ}, {OUT_PNG}, {', '.join(token_figures)}")


if __name__ == "__main__":
    main()