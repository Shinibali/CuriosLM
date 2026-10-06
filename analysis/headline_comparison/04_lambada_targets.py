"""Why do Models A and B trail GPT-2 on LAMBADA? Split the passages by whether the target word is capitalized.

Capitalized targets are mostly character names. Predicting one means copying a name from earlier in the passage,
and that is the kind of fiction and dialogue FineWeb-Edu's educational filter removes. Per model and per group:
perplexity (exp of the mean negative log-likelihood), median per-passage perplexity and accuracy.

    uv run --extra analysis python analysis/headline_comparison/04_lambada_targets.py
Needs 01's outputs and evaluate.py's LAMBADA samples for A, B and GPT-2. Writes data/lambada_targets.csv,
figures/lambada_targets.png, and prints the passages where GPT-2 beats Model A by the most.
"""
import csv
import os
import textwrap

import headline  # this study's shared choices
from common import evals, style  # noqa: E402

TASK = "lambada_openai"
OUT_CSV = os.path.join(headline.DATA, "lambada_targets.csv")
OUT_PNG = os.path.join(headline.FIGURES, "lambada_targets.png")
GROUPS = evals.TARGET_GROUPS
WORST = 5  # passages to print where GPT-2's target log-likelihood beats Model A's by the most


def main():
    models = {headline.LABELS[r]: headline.run_config(r)["out_dir"] for r in (headline.MODEL_A, headline.MODEL_B)}
    models["GPT-2 (OpenAI)"] = headline.REFERENCE
    samples = {}
    for label, out_dir in models.items():
        found = evals.all_samples(out_dir).get(TASK)
        assert found, f"no {TASK} samples for {out_dir}: run evaluate.py on it"
        samples[label] = found
    ids = sorted(next(iter(samples.values())))
    assert all(sorted(s) == ids for s in samples.values()), "the models were scored on different passages"

    rows = []
    for label, by_id in samples.items():
        for g in GROUPS:
            in_group = [by_id[d] for d in ids if evals.target_group(by_id[d]) == g]
            rows.append(dict(model=label, group=g, **evals.lambada_summary(in_group)))
    os.makedirs(headline.DATA, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']}, {r['group']} targets ({r['n']} passages): perplexity {r['perplexity']:.1f}, "
              f"median {r['median_perplexity']:.1f}, accuracy {r['accuracy']:.3f}")

    label_a, label_ref = list(models)[0], list(models)[-1]
    gap = sorted(ids, key=lambda d: samples[label_a][d]["perplexity"] - samples[label_ref][d]["perplexity"])
    print(f"\nPassages where GPT-2 beats {label_a} most (target log-likelihood, A vs GPT-2):")
    for d in gap[:WORST]:
        s = samples[label_a][d]
        tail = textwrap.shorten(s["doc"]["text"][-160:], 160, placeholder="…")  # the end of the passage
        print(f"  [{s['perplexity']:.1f} vs {samples[label_ref][d]['perplexity']:.1f}] …{tail}")

    style.apply()
    plt = style.plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.8 / len(models)  # bars per group share 80% of the slot
    for i, label in enumerate(models):
        values = [next(r["perplexity"] for r in rows if r["model"] == label and r["group"] == g) for g in GROUPS]
        xs = [k + (i - (len(models) - 1) / 2) * width for k in range(len(GROUPS))]  # centre the bars in each group
        ax.bar(xs, values, width * 0.92, color=style.SERIES[i], label=label)  # 0.92: a small gap between bars
        for x, v in zip(xs, values):
            ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=style.INK_SECONDARY)
    counts = {g: next(r["n"] for r in rows if r["group"] == g) for g in GROUPS}
    ax.set_xticks(range(len(GROUPS)), [f"{g} target\n({counts[g]:,} passages)" for g in GROUPS])
    ax.set_ylabel("perplexity on the target word ↓")
    ax.grid(axis="x", visible=False)
    ax.set_title("LAMBADA: the gap to GPT-2 is mostly names")
    ax.legend(loc="upper right")
    fig.tight_layout()
    os.makedirs(headline.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"\nwrote {OUT_CSV} and {OUT_PNG}")


if __name__ == "__main__":
    main()
