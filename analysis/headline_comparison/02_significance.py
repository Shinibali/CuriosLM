"""Are Model A and Model B different on the benchmarks, or tied? Paired tests on the per-example eval samples.

Both models answer the same questions, so tests compare them question by question:
  - accuracy: exact McNemar test — only questions where exactly one model is right carry information.
  - LAMBADA perplexity: paired bootstrap over passages of the perplexity difference.

    uv run --extra analysis python analysis/headline_comparison/02_significance.py
Finds each model's samples via evaluate.py's results.json; run 01_loss_overlay.py first (it fetches the
models' out_dir from wandb). Writes data/significance.csv.
"""
import csv
import os

import headline  # this study's shared choices; also puts analysis/ on sys.path
from common import evals  # noqa: E402

OUT = os.path.join(headline.DATA, "significance.csv")


def main():
    model_a, model_b = (headline.run_config(r)["out_dir"] for r in (headline.MODEL_A, headline.MODEL_B))
    rows = evals.compare(evals.all_samples(model_a), evals.all_samples(model_b), headline.BOOTSTRAP)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    evals.print_comparison(rows)


if __name__ == "__main__":
    main()
