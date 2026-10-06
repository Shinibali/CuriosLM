"""Does YaRN let Model B read past its training length? Loss at every position of long held-out documents.

Each document is a held-out FineWeb-Edu page at least YARN_FACTOR × 1,024 tokens long, scored from its start in one
forward pass, so every window holds one page and position p is conditioned on exactly p earlier tokens of it. With
causal attention, the loss at position p doesn't depend on what comes after it, so one pass gives every length:
the perplexity of a page's first L tokens is exp(mean loss over positions below L).

Models, all on the same pages:
- Model A, learned positions: has no positions past its table, so it stops at 1,024;
- Model B, plain RoPE run past its training length;
- Model B with YaRN, zero-shot (no training), at YARN_FACTOR;
- Model B with YaRN after the long-context fine-tune (--finetuned <out_dir>), which must use the same factor.

Two views: perplexity of the first L tokens against L (as in the YaRN paper), and mean loss at each position. The
second shows whether a model uses the extra context: if it does, loss keeps falling past 1,024 instead of flattening
or rising. Differences between models come with a paired bootstrap over pages (same resampled pages for both).

    uv run --extra analysis python analysis/positions/07_yarn_context.py [--docs 500] [--finetuned OUT_DIR]
Needs both headline checkpoints and the validation shard. Writes data/yarn_context.npz (each model's mean loss at
every position, and each page's mean loss over its first L tokens, which is all the report and figure use),
data/yarn_context.csv and figures/yarn_context.png. --plot-only redraws from the npz.
"""
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import positions  # this study's shared choices; also puts analysis/ on sys.path
from common import models, style  # noqa: E402
from common.perplexity import EOT, ShardedDataLoader  # noqa: E402
from curioslm import checkpoints  # noqa: E402

OUT_NPZ = os.path.join(positions.DATA, "yarn_context.npz")
OUT_CSV = os.path.join(positions.DATA, "yarn_context.csv")
OUT_PNG = os.path.join(positions.FIGURES, "yarn_context.png")
RESAMPLES = 2000  # paired bootstrap draws over pages
LABELS = {"A": "Model A, learned positions", "B": "Model B, plain RoPE", "B_yarn": "Model B + YaRN, zero-shot",
          "B_yarn_ft": "Model B + YaRN, fine-tuned"}
COLORS = {"A": style.MODEL_A, "B": style.MODEL_B, "B_yarn": style.OTHER[0], "B_yarn_ft": style.OTHER[1]}


def long_documents(dataset, length, num_docs):
    """(num_docs, length + 1) token windows: the first held-out pages with at least `length` tokens, each starting at
    the EOT that opens it (as in training), so window i holds inputs [:length] and targets [1:]."""
    loader = ShardedDataLoader(dataset, "val", block_size=1, batch_size=1, rank=0, world_size=1, device="cpu",
                               device_type="cpu")  # only its shard lookup is used
    tokens = loader.tokens
    eots = np.flatnonzero(tokens == EOT)
    page_lengths = np.diff(eots) - 1  # tokens between one EOT and the next
    starts = eots[:-1][page_lengths >= length]  # the EOT opening each long-enough page
    assert len(starts) >= num_docs, f"only {len(starts)} pages reach {length} tokens"
    print(f"{len(starts):,} of {len(page_lengths):,} validation pages reach {length:,} tokens; using the first "
          f"{num_docs}")
    return np.stack([tokens[start:start + length + 1] for start in starts[:num_docs]]).astype(np.int64)


@torch.no_grad()
def position_losses(model, windows, device):
    """(docs, positions) loss of every target, one page per forward pass."""
    length = model.config.block_size
    out = np.empty((len(windows), length), dtype=np.float32)
    for doc, window in enumerate(windows):
        page = torch.from_numpy(window[:length + 1])[None].to(device)
        logits, _ = model(page[:, :-1], page[:, 1:])  # targets given: logits at every position
        out[doc] = F.cross_entropy(logits[0].float(), page[0, 1:], reduction="none").cpu().numpy()
    return out


def load(out_dir, device, **overrides):
    """A checkpoint with context overrides (block_size, YaRN), eval mode."""
    positions.model_args(out_dir)  # downloads the checkpoint if it isn't on disk
    checkpoint = checkpoints.load(os.path.join(positions.ROOT, out_dir, "ckpt.pt"), map_location="cpu")
    model, model_args = checkpoints.build_model(checkpoint, overrides={"dropout": 0.0, **overrides})
    return model.eval().to(device), model_args


def report_lengths(trained, target):
    """The prefix lengths reported: a quarter of the training length, doubling up to the target (256 … 4,096)."""
    return [trained // 4 * 2 ** step for step in range(int(math.log2(4 * target / trained)) + 1)]


def summarize(losses, lengths):
    """Keep what the report and figure need from (pages, positions) losses: the mean loss at each position, and
    each page's mean loss over its first L tokens for every reported L the model reaches (pages, lengths)."""
    reached = [length for length in lengths if length <= losses.shape[1]]
    return losses.mean(axis=0), np.stack([losses[:, :length].mean(axis=1) for length in reached], axis=1)


def prefix_perplexity(page_prefix, lengths):
    """{L: perplexity of the first L tokens}, pooled over pages (every page contributes L tokens, so the pooled
    mean is the mean of the page means)."""
    return {length: math.exp(page_prefix[:, column].mean()) for column, length in enumerate(lengths)
            if column < page_prefix.shape[1]}


def paired_interval(page_means_a, page_means_b, seed=0):
    """95% interval for perplexity(b) − perplexity(a) from per-page mean losses, resampling pages for both."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(page_means_a), size=(RESAMPLES, len(page_means_a)))  # same pages for both models
    diffs = np.exp(page_means_b[draws].mean(axis=1)) - np.exp(page_means_a[draws].mean(axis=1))
    return np.percentile(diffs, [2.5, 97.5])


def evaluate(num_docs, finetuned):
    device = models.device()
    trained = positions.model_args(positions.out_dir(positions.ROPE))["block_size"]
    target = positions.YARN_FACTOR * trained
    learned_dir, rope_dir = positions.out_dir(positions.LEARNED), positions.out_dir(positions.ROPE)
    windows = long_documents(positions.config(positions.ROPE)["dataset"], target, num_docs)
    runs = {"A": (learned_dir, {}),  # stops at its table's 1,024 rows
            "B": (rope_dir, {"block_size": target}),
            "B_yarn": (rope_dir, {"block_size": target, "yarn_factor": float(positions.YARN_FACTOR),
                                  "yarn_original_context": trained})}
    if finetuned:
        runs["B_yarn_ft"] = (finetuned, {})  # its checkpoint already records its YaRN settings
    lengths, summary = report_lengths(trained, target), {}
    for key, (out_dir, overrides) in runs.items():
        model, model_args = load(out_dir, device, **overrides)
        if key == "B_yarn_ft":
            assert (model_args.get("yarn_factor"), model_args.get("yarn_original_context"),
                    model_args["block_size"]) == (positions.YARN_FACTOR, trained, target), \
                f"fine-tuned run's YaRN settings {model_args} differ from the zero-shot ones"
        losses = position_losses(model, windows, device)
        summary[f"{key}_position"], summary[f"{key}_pages"] = summarize(losses, lengths)
        print(f"{LABELS[key]}: scored {num_docs} pages × {losses.shape[1]:,} positions")
    np.savez(OUT_NPZ, trained=trained, target=target, num_docs=num_docs, **summary)


def report(data):
    trained, target = int(data["trained"]), int(data["target"])
    keys = [key for key in LABELS if f"{key}_position" in data]
    lengths = report_lengths(trained, target)
    rows = []
    for key in keys:
        for length, ppl in prefix_perplexity(data[f"{key}_pages"], lengths).items():
            rows.append(dict(model=LABELS[key], length=length, perplexity=ppl))
            print(f"{LABELS[key]:<28} first {length:>5,} tokens: perplexity {ppl:.2f}")
    comparisons = [("B", "B_yarn"), ("B_yarn", "B_yarn_ft")]
    for a, b in comparisons:
        if f"{a}_pages" in data and f"{b}_pages" in data:
            page_means_a, page_means_b = data[f"{a}_pages"][:, -1], data[f"{b}_pages"][:, -1]  # first `target` tokens
            low, high = paired_interval(page_means_a, page_means_b)
            difference = math.exp(page_means_b.mean()) - math.exp(page_means_a.mean())
            print(f"{LABELS[b]} − {LABELS[a]}, first {target:,} tokens: perplexity {difference:+.2f}, "
                  f"95% CI [{low:+.2f}, {high:+.2f}]")
    for key in keys:  # is the context past the training length used? later positions vs the last trained stretch
        position_mean = data[f"{key}_position"]
        if len(position_mean) > trained:
            before, after = position_mean[trained // 2:trained].mean(), position_mean[2 * trained:].mean()
            print(f"{LABELS[key]}: mean loss at positions {trained // 2:,}–{trained:,} {before:.3f}, "
                  f"{2 * trained:,}–{target:,} {after:.3f}")
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "length", "perplexity"])
        writer.writeheader()
        writer.writerows(rows)
    return keys, lengths


def plot(data, keys, lengths):
    trained, target = int(data["trained"]), int(data["target"])
    num_docs = int(data["num_docs"])
    style.apply()
    fig, (prefix_ax, position_ax) = plt.subplots(1, 2, figsize=(12, 4.4))
    for key in keys:
        position_mean = data[f"{key}_position"]
        span = len(position_mean)
        every_length = np.arange(1, span + 1)
        prefix_ax.plot(every_length[15:], np.exp(np.cumsum(position_mean) / every_length)[15:],
                       color=COLORS[key], label=LABELS[key])  # skip the first 16: a handful of tokens is noisy
        edges = np.unique(np.geomspace(16, span, 40).astype(int))  # log-spaced position buckets
        centres = np.sqrt(edges[:-1] * edges[1:])
        position_ax.plot(centres, [position_mean[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:])],
                         color=COLORS[key], label=LABELS[key])
    for ax in (prefix_ax, position_ax):
        ax.set_xscale("log", base=2)
        ticks = [16, 64] + lengths  # the curves start at position 16
        ax.set_xticks(ticks, [f"{tick:,}" for tick in ticks])
        ax.minorticks_off()
        ax.axvline(trained, color=style.INK, linewidth=1, linestyle="--", zorder=1)
        ax.text(trained * 1.05, 0.97, f"training length\n{trained:,}", transform=ax.get_xaxis_transform(),
                fontsize=8.5, va="top")
    prefix_ax.set_xlabel("Tokens read from the start of the page, L")
    prefix_ax.set_ylabel("Perplexity of the first L tokens")
    prefix_ax.set_title("Perplexity vs length", fontsize=10)
    position_ax.set_xlabel("Position in the page")
    position_ax.set_ylabel("Mean loss at that position")
    position_ax.set_title("Loss at each position", fontsize=10)
    upper = max(np.exp(data[f"{key}_position"][trained // 2:trained].mean()) for key in keys)  # the useful range
    prefix_ax.set_ylim(top=upper * 3)
    handles, labels = prefix_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=len(keys), fontsize=8.5)
    fig.text(0.01, 0.99, f"Stretching Model B from {trained:,} to {target:,} tokens", fontsize=12,
             fontweight="bold", va="top")
    fig.text(0.01, 0.945, f"{num_docs} held-out FineWeb-Edu pages of at least {target:,} tokens, each read from its "
             "start.", fontsize=9, color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.06, right=0.98, wspace=0.22)
    os.makedirs(positions.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_CSV}, {OUT_PNG}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=500, help="held-out pages scored (each at least the target long)")
    parser.add_argument("--finetuned", help="out_dir of the YaRN fine-tune run, once it exists")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        os.makedirs(positions.DATA, exist_ok=True)
        evaluate(args.docs, args.finetuned)
    data = dict(np.load(OUT_NPZ))
    plot(data, *report(data))


if __name__ == "__main__":
    main()
