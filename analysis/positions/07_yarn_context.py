"""Does YaRN let Model B read past its training length? Loss at every position of long held-out documents.

Each document is a held-out FineWeb-Edu page at least YarnStudy.factor × 1,024 tokens long, scored from its start in
one forward pass, so every window holds one page and position p is conditioned on exactly p earlier tokens of it. With
causal attention, the loss at position p doesn't depend on what comes after it, so one pass gives every length:
the perplexity of a page's first L tokens is exp(mean loss over positions below L).

Models, all on the same pages:
- Model A, learned positions: has no positions past its table, so it stops at 1,024;
- Model B, plain RoPE run past its training length;
- Model B with YaRN, zero-shot (no training), at YarnStudy.factor (curioslm/config.py);
- Model B with YaRN after the long-context fine-tune, at every snapshot the run kept (--finetuned <its out_dir>, with
  its ckpt-<iter>.pt files on disk), so the effect is seen growing with fine-tuning steps; same factor required.

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
import glob
import math
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as F

import positions  # this study's shared choices; also puts analysis/ on sys.path
from common import models, style  # noqa: E402
from common import perplexity as ppl  # noqa: E402
from curioslm import checkpoints  # noqa: E402
from curioslm.loaders.long_page_loader import LongPageLoader  # noqa: E402

OUT_NPZ = os.path.join(positions.DATA, "yarn_context.npz")
OUT_CSV = os.path.join(positions.DATA, "yarn_context.csv")
OUT_PNG = os.path.join(positions.FIGURES, "yarn_context.png")
DOCS = 500  # held-out pages scored: enough for a paired 95% interval of about ±0.5 perplexity at the target
SHORTEST_SHARE = 4  # the shortest reported prefix is 1/4 of the training length; lengths double up to the target
FIRST_PLOTTED = 16  # curves start here: averages over a handful of tokens are noise
POSITION_BUCKETS = 40  # log-spaced buckets in the loss-by-position panel
Y_HEADROOM = 3  # perplexity axis tops out at 3x the worst model's loss just below the training length
LABELS = {"A": "Model A, learned positions", "B": "Model B, plain RoPE", "B_yarn": "Model B + YaRN, zero-shot"}
COLORS = {"A": style.MODEL_A, "B": style.MODEL_B, "B_yarn": style.OTHER[0]}
FINETUNED = "B_yarn_ft"  # + fine-tuning steps, e.g. B_yarn_ft25
# fine-tune snapshots in one teal hue, darker with more steps
FINETUNE_RAMP = LinearSegmentedColormap.from_list("finetune", ["#9fd4d4", style.ACCENT, "#06393a"])


def label(key):
    return LABELS.get(key) or f"Model B + YaRN, fine-tuned {int(key[len(FINETUNED):])} steps"


def model_keys(data):
    """The models in a saved summary, in legend order: fixed ones, then fine-tune snapshots by step."""
    finetuned = sorted((key[:-len("_position")] for key in data if key.startswith(FINETUNED) and
                        key.endswith("_position")), key=lambda key: int(key[len(FINETUNED):]))
    return [key for key in LABELS if f"{key}_position" in data] + finetuned


def colors(keys):
    finetuned = [key for key in keys if key.startswith(FINETUNED)]
    shades = {key: FINETUNE_RAMP(share) for key, share in zip(finetuned, np.linspace(0, 1, len(finetuned)))}
    return {key: COLORS.get(key, shades.get(key)) for key in keys}


def long_documents(dataset, length, num_docs):
    """(num_docs, length + 1) token windows: the first held-out pages with at least `length` tokens, each starting at
    the EOT that opens it, so window i holds inputs [:length] and targets [1:]. The pages the fine-tune's loader picks
    (loaders/long_page_loader.py), from the validation shard."""
    pages = LongPageLoader(dataset, "val", block_size=length, batch_size=1, rank=0, world_size=1, device="cpu",
                           device_type="cpu").pages  # the validation split is one shard
    assert len(pages) >= num_docs, f"only {len(pages)} pages reach {length} tokens"
    print(f"{len(pages):,} validation pages reach {length:,} tokens; using the first {num_docs}")
    return pages[:num_docs].astype(np.int64)


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


def load(path, device, **overrides):
    """A checkpoint file with context overrides (block_size, YaRN), eval mode. Returns (model, model_args, step)."""
    checkpoint = checkpoints.load(path, map_location="cpu")
    model, model_args = checkpoints.build_model(checkpoint, overrides={"dropout": 0.0, **overrides})
    return model.eval().to(device), model_args, checkpoint["iter_num"]


def checkpoint_path(out_dir):
    """A headline run's ckpt.pt, downloaded from the staging repo first if it isn't on disk."""
    positions.model_args(out_dir)
    return os.path.join(positions.ROOT, out_dir, "ckpt.pt")


def report_lengths(trained, target):
    """The prefix lengths reported: a quarter of the training length, doubling up to the target (256 … 4,096)."""
    shortest = trained // SHORTEST_SHARE
    return [shortest * 2 ** step for step in range(int(math.log2(target / shortest)) + 1)]


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


def evaluate(num_docs, finetuned):
    device = models.device()
    trained = positions.model_args(positions.out_dir(positions.ROPE))["block_size"]
    target = int(positions.YARN.factor * trained)
    learned_dir, rope_dir = positions.out_dir(positions.LEARNED), positions.out_dir(positions.ROPE)
    windows = long_documents(positions.config(positions.ROPE)["dataset"], target, num_docs)
    runs = [("A", checkpoint_path(learned_dir), {}),  # stops at its table's 1,024 rows
            ("B", checkpoint_path(rope_dir), {"block_size": target}),
            ("B_yarn", checkpoint_path(rope_dir), {"block_size": target, "yarn_factor": positions.YARN.factor,
                                                   "yarn_original_context": trained})]
    if finetuned:  # snapshots record their own YaRN settings
        snapshots = glob.glob(os.path.join(positions.ROOT, finetuned, "ckpt-*.pt"))
        assert snapshots, f"no ckpt-<iter>.pt snapshots in {finetuned}: copy them from the pod or the staging repo"
        runs += [(FINETUNED, path, {}) for path in sorted(snapshots, key=lambda path: int(path[:-3].split("-")[-1]))]
    lengths, summary = report_lengths(trained, target), {}
    for key, path, overrides in runs:
        model, model_args, step = load(path, device, **overrides)
        if key == "B":
            headline_step = step  # fine-tuning steps are counted from here
        if key == FINETUNED:
            assert (model_args.get("yarn_factor"), model_args.get("yarn_original_context"),
                    model_args["block_size"]) == (positions.YARN.factor, trained, target), \
                f"fine-tuned run's YaRN settings {model_args} differ from the zero-shot ones"
            key = f"{FINETUNED}{step - headline_step}"
        losses = position_losses(model, windows, device)
        summary[f"{key}_position"], summary[f"{key}_pages"] = summarize(losses, lengths)
        print(f"{label(key)}: scored {num_docs} pages × {losses.shape[1]:,} positions")
    np.savez(OUT_NPZ, trained=trained, target=target, num_docs=num_docs, **summary)


def report(data):
    trained, target = int(data["trained"]), int(data["target"])
    keys = model_keys(data)
    lengths = report_lengths(trained, target)
    rows = []
    for key in keys:
        for length, perplexity in prefix_perplexity(data[f"{key}_pages"], lengths).items():
            rows.append(dict(model=label(key), length=length, perplexity=perplexity))
            print(f"{label(key):<42} first {length:>5,} tokens: perplexity {perplexity:.2f}")
    comparisons = [("B", "B_yarn")] + [("B_yarn", key) for key in keys if key.startswith(FINETUNED)]
    for a, b in comparisons:
        if f"{a}_pages" in data and f"{b}_pages" in data:
            page_means_a, page_means_b = data[f"{a}_pages"][:, -1], data[f"{b}_pages"][:, -1]  # first `target` tokens
            counts = np.full(len(page_means_a), target)  # every page contributes `target` scored tokens
            low, high = ppl.paired_bootstrap(page_means_a * counts, page_means_b * counts, counts, positions.BOOTSTRAP)
            difference = math.exp(page_means_b.mean()) - math.exp(page_means_a.mean())
            print(f"{label(b)} − {label(a)}, first {target:,} tokens: perplexity {difference:+.2f}, "
                  f"95% CI [{low:+.2f}, {high:+.2f}]")
    for key in keys:  # is the context past the training length used? later positions vs the last trained stretch
        position_mean = data[f"{key}_position"]
        if len(position_mean) > trained:
            before, after = position_mean[trained // 2:trained].mean(), position_mean[2 * trained:].mean()
            print(f"{label(key)}: mean loss at positions {trained // 2:,}–{trained:,} {before:.3f}, "
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
    key_colors = colors(keys)
    for key in keys:
        position_mean = data[f"{key}_position"]
        span = len(position_mean)
        every_length = np.arange(1, span + 1)
        prefix_ax.plot(every_length[FIRST_PLOTTED - 1:],
                       np.exp(np.cumsum(position_mean) / every_length)[FIRST_PLOTTED - 1:],
                       color=key_colors[key], label=label(key))
        edges = np.unique(np.geomspace(FIRST_PLOTTED, span, POSITION_BUCKETS).astype(int))
        centres = np.sqrt(edges[:-1] * edges[1:])
        position_ax.plot(centres, [position_mean[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:])],
                         color=key_colors[key], label=label(key))
    for ax in (prefix_ax, position_ax):
        ax.set_xscale("log", base=2)
        ticks = [FIRST_PLOTTED, FIRST_PLOTTED * SHORTEST_SHARE] + lengths
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
    prefix_ax.set_ylim(top=upper * Y_HEADROOM)
    handles, labels = prefix_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=min(len(keys), 4), fontsize=8.5)
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
    parser.add_argument("--docs", type=int, default=DOCS, help="held-out pages scored (each at least the target long)")
    parser.add_argument("--finetuned", help="out_dir of the YaRN fine-tune run; every ckpt-<iter>.pt in it is scored")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        os.makedirs(positions.DATA, exist_ok=True)
        evaluate(args.docs, args.finetuned)
    data = dict(np.load(OUT_NPZ))
    plot(data, *report(data))


if __name__ == "__main__":
    main()
