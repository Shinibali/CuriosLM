"""evaluate.py's saved results, and paired comparisons of two models on the same benchmark questions — shared by every
study that compares models on benchmarks."""
import glob
import json
import math
import os
import random
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACCURACY_METRICS = ("acc", "acc_norm")  # per-question 0/1 metrics lm-eval records in its samples


def result_dirs(out_dir):
    """Every evaluate.py output folder for this model, found by the `model` field it records in results.json."""
    dirs = []
    for f in glob.glob(os.path.join(ROOT, "evals", "**", "results.json"), recursive=True):
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


def perplexity(lls, idx=None):
    """exp(−mean log-likelihood) over the chosen passages (all of them by default)."""
    idx = range(len(lls)) if idx is None else idx
    return math.exp(-sum(lls[i] for i in idx) / len(idx))


def perplexity_row(task, a, b, n_resamples):
    """exp(−mean log-likelihood) per model, with a paired bootstrap over passages (same draw for both)."""
    ids = sorted(a)
    ll_a, ll_b = [a[d]["perplexity"] for d in ids], [b[d]["perplexity"] for d in ids]  # per-passage log-likelihoods
    rng = random.Random(0)
    diffs = sorted(perplexity(ll_b, idx) - perplexity(ll_a, idx)
                   for idx in ([rng.randrange(len(ids)) for _ in ids] for _ in range(n_resamples)))
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]  # 95% interval for B − A
    pa, pb = perplexity(ll_a), perplexity(ll_b)  # over all passages
    return dict(task=task, metric="perplexity", n=len(ids), a=pa, b=pb, diff=pb - pa, a_only="", b_only="",
                p=f"95% CI [{lo:+.2f}, {hi:+.2f}]")


def compare(samples_a, samples_b, n_resamples):
    """Paired test rows for every task both models were scored on: McNemar per accuracy metric, and a paired
    bootstrap for LAMBADA-style perplexity. Each row: task, metric, n, a, b, diff (B − A), a_only, b_only, p."""
    tasks = sorted(samples_a.keys() & samples_b.keys())  # only tasks both models were scored on
    assert tasks, "no shared eval samples: run evaluate.py on both models"
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
            rows.append(perplexity_row(task, a, b, n_resamples))
    return rows


def print_comparison(rows, label_a="A", label_b="B"):
    """Print compare()'s rows: each task and metric, both models' scores, and the p-value or CI."""
    for r in rows:
        p = r["p"] if isinstance(r["p"], str) else f"p = {r['p']:.3f}"  # perplexity rows carry a CI instead
        print(f"{r['task']} {r['metric']}: {label_a} {r['a']:.4f}, {label_b} {r['b']:.4f}, diff {r['diff']:+.4f}, {p}")


TARGET_GROUPS = ("capitalized", "lowercase")


def target_group(sample):
    """LAMBADA: 'capitalized' if the target word starts with an upper-case letter (mostly names), else 'lowercase'."""
    return TARGET_GROUPS[0] if sample["target"].strip()[:1].isupper() else TARGET_GROUPS[1]


def lambada_summary(samples):
    """Perplexity, median per-passage perplexity and accuracy over a list of LAMBADA samples."""
    lls = [s["perplexity"] for s in samples]  # lm-eval stores each passage's target log-likelihood here
    return dict(n=len(samples), perplexity=perplexity(lls),
                median_perplexity=math.exp(-statistics.median(lls)),  # exp is monotone: median of per-passage ppl
                accuracy=sum(s["acc"] for s in samples) / len(samples))
