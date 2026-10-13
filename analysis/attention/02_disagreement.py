"""Where Model A and Model B disagree, and what their attention was doing there.

Both models read the same held-out FineWeb-Edu windows. At every token, the disagreement between their next-token
distributions is the KL divergence KL(A ‖ B) = Σ p_A · (log p_A − log p_B): the extra surprise, in nats, of
predicting with B's distribution when A's is taken as the reference. 0 means they agree exactly. For the tokens
where they disagree most, the script prints the context, the true next token and each model's top guess, and takes
each model's attention from that token (the model's explicit attention path, GPT.attention_weights).

A static figure for the post: the distribution of per-token KL, and from the strongest-disagreement token, each
model's attention to the tokens before it, layer by layer, averaged over heads.

    uv run --extra analysis python analysis/attention/02_disagreement.py [--windows 20] [--plot-only]
Needs both headline checkpoints and the validation shard. Writes data/disagreement.npz and figures/disagreement.png.
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
from common import perplexity as ppl  # noqa: E402

OUT_NPZ = os.path.join(attention.DATA, "disagreement.npz")
OUT_PNG = os.path.join(attention.FIGURES, "disagreement.png")
WINDOWS = 20  # held-out windows of block_size tokens: ~20k tokens, enough for a stable KL distribution
TOP = 10  # strongest disagreements printed
CONTEXT = 24  # tokens before a disagreement shown: in the printout and the attention heatmaps


@torch.no_grad()
def compare(model_a, model_b, windows, device):
    """Per token: KL(A ‖ B), each model's loss, and each model's top guess and its probability."""
    results = {key: [] for key in ("kl", "loss_a", "loss_b", "guess_a", "guess_b", "confidence_a", "confidence_b")}
    for window in windows:  # window: (T + 1,) token ids
        idx = torch.from_numpy(window[:-1])[None].to(device)  # (1, T) inputs
        targets = torch.from_numpy(window[1:])[None].to(device)  # (1, T) the true next token at each position
        # targets given, so the model returns logits at every position; [0][0] = logits of the one sequence
        log_a = F.log_softmax(model_a(idx, targets)[0][0].float(), dim=-1)  # (T, vocab) log p_A(next token)
        log_b = F.log_softmax(model_b(idx, targets)[0][0].float(), dim=-1)  # (T, vocab) log p_B(next token)
        results["kl"].append((log_a.exp() * (log_a - log_b)).sum(-1))  # (T,) KL(A ‖ B) = Σ p_A·(log p_A − log p_B)
        for name, log_probs in (("a", log_a), ("b", log_b)):
            # gather picks each row's log-probability of its true token: (T, 1) index -> (T, 1) -> (T,); minus =
            # cross-entropy loss at that position
            results[f"loss_{name}"].append(-log_probs.gather(-1, targets[0][:, None])[:, 0])
            confidence, guess = log_probs.exp().max(-1)  # (T,) top probability and (T,) its token id, per position
            results[f"guess_{name}"].append(guess)  # the model's favourite next token
            results[f"confidence_{name}"].append(confidence)  # how much probability it gave that token
    return {key: torch.stack(values).cpu().numpy() for key, values in results.items()}  # each (windows, T)


@torch.no_grad()
def passage_attention(model, window, start, position, device):
    """(layers, heads, P, P): the model's attention among the passage tokens start..position, every layer, as computed
    when reading the window from its beginning. Attention is causal, so the input can stop at `position`: tokens after
    it can't change anything up to it. Rows don't sum to 1 here: weight on keys before `start` isn't shown."""
    idx = torch.from_numpy(window[:position + 1])[None].to(device)
    return np.stack([layer[0, :, start:, start:].float().cpu().numpy() for layer in model.attention_weights(idx)])


def compact(data):
    """Smaller arrays for saving: token ids as uint16 (the vocabulary fits); KL, losses and confidences stay float32,
    so the numbers the post quotes are exact."""
    return {key: value.astype(np.uint16) if key.startswith("guess") else value for key, value in data.items()}


def strongest(kl, count):
    """(window, position) of the `count` largest KL values, largest first."""
    # kl is (windows, T); axis=None sorts it flattened, ascending; [::-1] makes it largest first; keep `count`
    order = np.argsort(kl, axis=None)[::-1][:count]  # (count,) indices into the flattened windows × T array
    return [np.unravel_index(flat, kl.shape) for flat in order]  # each flat index -> (window, position)


def report(data, enc):
    print(f"mean KL(A ‖ B) {data['kl'].mean():.3f} nats per token over {data['kl'].size:,} tokens; "
          f"median {np.median(data['kl']):.3f}")
    for rank, (window, position) in enumerate(strongest(data["kl"], TOP), start=1):
        tokens = data["windows"][window]
        context = enc.decode(tokens[max(0, position - CONTEXT + 1):position + 1].tolist())
        truth = enc.decode([int(tokens[position + 1])])
        guesses = []
        for name, key in (("A", "a"), ("B", "b")):
            guess = enc.decode([int(data[f"guess_{key}"][window, position])])
            guesses.append(f"{name} {guess!r} {data[f'confidence_{key}'][window, position]:.0%}")
        print(f"{rank:>2}. KL {data['kl'][window, position]:.2f}  …{context!r} → true {truth!r}; "
              f"{', '.join(guesses)}; loss A {data['loss_a'][window, position]:.2f}, "
              f"B {data['loss_b'][window, position]:.2f}")


def plot(data, enc, rows):
    window, position = strongest(data["kl"], 1)[0]
    start = max(0, position - CONTEXT + 1)
    labels = attention.token_labels(enc, data["windows"][window][start:position + 1].tolist())
    style.apply()
    fig, axes = plt.subplots(1, 1 + len(rows), figsize=(6 + 5 * len(rows), 4.2),
                             gridspec_kw={"width_ratios": [1] + [1.4] * len(rows)},
                             layout="constrained")  # keeps the colorbar clear of the last panel
    kl = data["kl"].ravel()
    bins = np.geomspace(max(kl.min(), 1e-4), kl.max(), 50)  # KL spans orders of magnitude: log-spaced bins
    axes[0].hist(kl, bins=bins, color=style.ACCENT)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("KL(A ‖ B) at one token (nats)")
    axes[0].set_ylabel("Tokens")
    axes[0].set_title("Disagreement at each token", fontsize=10)
    for ax, (name, model_rows) in zip(axes[1:], rows.items()):
        shown = model_rows[:, :, -1, :].mean(axis=1)  # (layers, passage keys): the last row is the disagreement token
        earlier = 1 - shown.sum(axis=1, keepdims=True)  # the rest of each row went to tokens before the passage
        per_layer = np.concatenate([earlier, shown], axis=1)
        image = ax.imshow(per_layer, cmap=style.SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels) + 1), ["(earlier tokens)"] + labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(per_layer)), [str(layer) for layer in range(len(per_layer))], fontsize=7)
        ax.grid(False)
        ax.set_ylabel("Layer")
        ax.set_title(f"{name}: attention from that token", fontsize=10)
    fig.colorbar(image, ax=axes[-1], label="Attention (mean over heads)")
    fig.suptitle(f"Where Model A and Model B disagree: {data['kl'].size:,} held-out tokens, and their attention at the "
                 "strongest disagreement", x=0.01, ha="left", fontsize=12, fontweight="bold")
    os.makedirs(attention.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=WINDOWS)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    enc = tiktoken.get_encoding("gpt2")
    if not args.plot_only:
        device = models.device()
        model_a, model_b = (attention.load(name, device) for name in attention.MODELS)
        length = model_a.config.block_size
        assert length == model_b.config.block_size, "the models must read the same windows"
        dataset = attention.config(attention.MODELS["Model B"])["dataset"]
        stream = ppl.val_tokens(dataset, args.windows * length + 1)
        windows = np.stack([stream[start:start + length + 1] for start in range(0, args.windows * length, length)])
        data = compare(model_a, model_b, windows, device)
        window, position = strongest(data["kl"], 1)[0]
        start = max(0, position - CONTEXT + 1)
        rows = {name: passage_attention(model, windows[window], start, position, device)
                for name, model in zip(attention.MODELS, (model_a, model_b))}
        os.makedirs(attention.DATA, exist_ok=True)
        saved_rows = {f"rows_{name.replace(' ', '_')}": value.astype(np.float16) for name, value in rows.items()}
        np.savez_compressed(OUT_NPZ, windows=windows.astype(np.uint16), **compact(data), **saved_rows)
    data = {key: value.astype(np.int64 if value.dtype == np.uint16 else np.float32)
            for key, value in np.load(OUT_NPZ).items()}  # back to the types the analysis computes with
    rows = {name: data[f"rows_{name.replace(' ', '_')}"] for name in attention.MODELS}
    report(data, enc)
    plot(data, enc, rows)
    print(f"wrote {OUT_NPZ}, {OUT_PNG}")


if __name__ == "__main__":
    main()
