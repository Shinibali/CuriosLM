"""Held-out perplexity on every dataset in the mix, for Model B, the 50:50 run, Model A and stock GPT-2, with a paired
document-level bootstrap for the 50:50 run against Model B.

    uv run --extra analysis --extra eval python analysis/data_mix/03_val_perplexity.py [--tokens N]
Needs the checkpoints (downloaded if missing) and each dataset's validation shard. Writes data/val_perplexity.csv.
"""
import argparse
import csv
import os


import mix  # this study's shared choices; also puts the repo root and analysis/ on sys.path
from common import models  # noqa: E402
from common import perplexity as ppl  # noqa: E402
from curioslm.eval_adapter import load_model  # noqa: E402

OUT = os.path.join(mix.DATA, "val_perplexity.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2_000_000, help="held-out tokens per set")
    parser.add_argument("--batch-size", type=int, default=4, help="windows per forward pass (keep small on a Mac)")
    args = parser.parse_args()
    os.chdir(mix.ROOT)  # run directories are relative to the repo root
    names = {mix.LABELS[r]: mix.config(r)["out_dir"] for r in (mix.BASE, mix.MIX, mix.MODEL_A)}
    names["GPT-2 (OpenAI)"] = mix.REFERENCE
    dev, rows = models.device(), []
    for dataset in mix.val_sets():
        tokens = ppl.val_tokens(dataset, args.tokens)  # the same tokens for every model
        sums = {}
        for label, out_dir in names.items():
            model = load_model(out_dir).eval().to(dev)
            sums[label] = ppl.doc_sums(tokens, ppl.window_losses(model, tokens, dev, args.batch_size), ppl.EOT)
            value = ppl.perplexity(*sums[label])
            rows.append(dict(set=dataset, model=label, perplexity=value, docs=len(sums[label][1])))
            print(f"{dataset}, {label}: perplexity {value:.3f}")
            del model
        base, mixed = sums[mix.LABELS[mix.BASE]], sums[mix.LABELS[mix.MIX]]
        lo, hi = ppl.paired_bootstrap(base[0], mixed[0], base[1], mix.BOOTSTRAP)  # mix − base
        rows.append(dict(set=dataset, model="50:50 − 100% FW, 95% CI", perplexity=f"[{lo:+.3f}, {hi:+.3f}]", docs=""))
        print(f"{dataset}, 50:50 − 100% FineWeb-Edu: 95% CI [{lo:+.3f}, {hi:+.3f}]")
    os.makedirs(mix.DATA, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
