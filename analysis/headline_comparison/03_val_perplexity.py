"""Perplexity of Model A, Model B and stock GPT-2 on held-out text: the runs' own training dataset plus the other
sets in headline.VAL_SETS.

A fixed slice of each validation shard is cut into non-overlapping block_size windows and scored token by token;
token losses are grouped by document (documents are separated by the end-of-text token), so A vs B is compared
with a paired bootstrap over documents rather than a naive standard error. Run 01_loss_overlay.py first: it fetches
the runs' out_dir and dataset from wandb.

    uv run --extra analysis --extra eval python analysis/headline_comparison/03_val_perplexity.py [--tokens 50000]
Writes data/val_perplexity.csv (per model and set) and data/val_doc_losses.npz (per-document sums, for re-analysis).
"""
import argparse
import csv
import os

import numpy as np
import tiktoken

import headline  # this study's shared choices; also puts the repo root and analysis/ on sys.path
from common import perplexity as ppl  # noqa: E402
from curioslm.eval_adapter import default_device, load_model  # noqa: E402

EOT = tiktoken.get_encoding("gpt2").eot_token  # the document separator prepare.py writes between documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2_000_000,
                        help="tokens per set (default: thousands of documents, enough for a document-level bootstrap)")
    parser.add_argument("--batch-size", type=int, default=4, help="windows per forward pass; full-vocab logits are "
                        "batch × block_size × vocab floats, so keep it small on a Mac")
    args = parser.parse_args()
    os.chdir(headline.ROOT)  # run directories are relative to the repo root
    configs = {r: headline.run_config(r) for r in (headline.MODEL_A, headline.MODEL_B)}  # out_dir, dataset from wandb
    datasets = {c["dataset"] for c in configs.values()}  # the runs' training dataset (one, for a fair comparison)
    assert len(datasets) == 1, f"the compared runs trained on different datasets: {datasets}"
    models = {"A": configs[headline.MODEL_A]["out_dir"], "B": configs[headline.MODEL_B]["out_dir"],
              headline.REFERENCE: headline.REFERENCE}
    dev = default_device()
    rows, saved = [], {}
    for set_name in [*datasets, *headline.VAL_SETS]:  # training-distribution text first, then the others
        tokens = ppl.val_tokens(set_name, args.tokens)  # the same tokens for every model
        per_model, blocks = {}, set()
        for label, out_dir in models.items():
            model = load_model(out_dir).eval().to(dev)  # downloads the checkpoint if missing
            blocks.add(model.config.block_size)
            per_model[label] = ppl.doc_sums(tokens, ppl.window_losses(model, tokens, dev, args.batch_size), EOT)
            sums, counts = per_model[label]
            value = ppl.perplexity(sums, counts)  # e^(mean loss per token) over the whole slice
            rows.append(dict(set=set_name, model=label, tokens=int(counts.sum()), docs=len(counts), perplexity=value))
            saved[f"{set_name}_{label}_sums"], saved[f"{set_name}_{label}_counts"] = sums, counts
            print(f"{set_name}, {label}: perplexity {value:.3f} ({counts.sum():,} tokens, {len(counts):,} docs)")
            del model  # free memory before the next one loads
        assert len(blocks) == 1, f"models use different block sizes {blocks}: windows wouldn't be comparable"
        lo, hi = ppl.paired_bootstrap(per_model["A"][0], per_model["B"][0], per_model["A"][1], headline.BOOTSTRAP)
        print(f"{set_name}, B − A: 95% CI [{lo:+.3f}, {hi:+.3f}]")
        rows.append(dict(set=set_name, model="B-A 95% CI", tokens="", docs="", perplexity=f"[{lo:+.3f}, {hi:+.3f}]"))

    os.makedirs(headline.DATA, exist_ok=True)
    with open(os.path.join(headline.DATA, "val_perplexity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez(os.path.join(headline.DATA, "val_doc_losses.npz"), **saved)


if __name__ == "__main__":
    main()
