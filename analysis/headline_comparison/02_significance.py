"""Are Model A and Model B different on the benchmarks, or tied? Paired tests on the per-example eval samples.

Both models answer the same questions, so tests compare them question by question:
  - accuracy: exact McNemar test — only questions where exactly one model is right carry information.
  - LAMBADA perplexity: paired bootstrap over passages of the perplexity difference.

    uv run --extra analysis python analysis/headline_comparison/02_significance.py
Finds each model's samples via evaluate.py's results.json; run 01_loss_overlay.py first (it fetches the
models' out_dir from wandb). Writes data/significance.csv.
"""
import csv
import glob
import json
import math
import os
import random

import headline  # this study's shared choices

OUT = os.path.join(headline.DATA, "significance.csv")
ACCURACY_METRICS = ("acc", "acc_norm")  # per-question 0/1 metrics lm-eval records in its samples

ppl = lambda lls, idx: math.exp(-sum(lls[i] for i in idx) / len(idx))

def result_dirs(out_dir):
    """Every evaluate.py output folder for this model, found by the `model` field it records in results.json."""
    dirs = []
    for f in glob.glob(os.path.join(headline.ROOT, "evals", "**", "results.json"), recursive=True):
        with open(f) as fh:
            if json.load(fh)["model"] == out_dir:
                dirs.append(os.path.dirname(f))
    return dirs


def all_samples(out_dir):
    """{task: {doc_id: sample}} across all of a model's eval runs."""
    out = {}
    for d in result_dirs(out_dir):
        for f in glob.glob(os.path.join(d, "samples_*.jsonl")):
            task = os.path.basename(f)[len("samples_"):-len(".jsonl")]
            with open(f) as fh:
                out[task] = {s["doc_id"]: s for s in map(json.loads, fh)}
    return out


def mcnemar_exact(b_only, a_only):
    """Two-sided exact McNemar p-value: under 'no difference', each discordant question is a fair coin flip."""
    n, k = b_only + a_only, min(b_only, a_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n  # P(X <= k), X ~ Binomial(n, 1/2)
    return min(1.0, 2 * tail)


def perplexity_row(task, a, b):
    """exp(−mean log-likelihood) per model, with a paired bootstrap over passages (same draw for both)."""
    ids = sorted(a)
    ll_a, ll_b = [a[d]["perplexity"] for d in ids], [b[d]["perplexity"] for d in ids]  # per-passage log-likelihoods
    rng = random.Random(0)
    diffs = sorted(ppl(ll_b, idx) - ppl(ll_a, idx)
                   for idx in ([rng.randrange(len(ids)) for _ in ids] for _ in range(headline.BOOTSTRAP)))
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]  # 95% interval for B − A
    full = range(len(ids))
    return dict(task=task, metric="perplexity", n=len(ids), a=ppl(ll_a, full), b=ppl(ll_b, full),
                diff=ppl(ll_b, full) - ppl(ll_a, full), a_only="", b_only="", p=f"95% CI [{lo:+.2f}, {hi:+.2f}]")


def main():
    model_a, model_b = (headline.run_config(r)["out_dir"] for r in (headline.MODEL_A, headline.MODEL_B))
    samples_a, samples_b = all_samples(model_a), all_samples(model_b)
    tasks = sorted(samples_a.keys() & samples_b.keys())  # only tasks both models were scored on
    assert tasks, f"no shared eval samples for {model_a} and {model_b}: run evaluate.py on both"
    rows = []
    for task in tasks:
        a, b = samples_a[task], samples_b[task]
        assert a.keys() == b.keys(), f"{task}: the two runs scored different questions"
        first = next(iter(a.values()))
        for metric in (m for m in ACCURACY_METRICS if m in first):
            a_only = sum(a[d][metric] > b[d][metric] for d in a)  # A right, B wrong
            b_only = sum(b[d][metric] > a[d][metric] for d in a)  # B right, A wrong
            n = len(a)
            rows.append(dict(task=task, metric=metric, n=n, a=sum(s[metric] for s in a.values()) / n,
                             b=sum(s[metric] for s in b.values()) / n, diff=(b_only - a_only) / n,  # concordant cancel
                             a_only=a_only, b_only=b_only, p=mcnemar_exact(b_only, a_only)))
        if "perplexity" in first:  # LAMBADA-style: per-passage log-likelihood of the target
            rows.append(perplexity_row(task, a, b))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'task':16s}{'metric':12s}{'A':>10s}{'B':>10s}{'B−A':>9s}  {'A-only':>7s}{'B-only':>7s}  p / CI")
    for r in rows:
        p = r["p"] if isinstance(r["p"], str) else f"p = {r['p']:.3f}"  # perplexity rows carry a CI instead
        print(f"{r['task']:16s}{r['metric']:12s}{r['a']:10.4f}{r['b']:10.4f}{r['diff']:+9.4f}  {str(r['a_only']):>7s}"
              f"{str(r['b_only']):>7s}  {p}")


if __name__ == "__main__":
    main()
