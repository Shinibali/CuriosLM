"""Did Cosmopedia's synthetic stories bring back the skill FineWeb-Edu lacks? LAMBADA split by capitalized (mostly
names) vs lowercase target, for Model B, the 50:50 run and stock GPT-2.

    uv run --extra analysis python analysis/data_mix/04_lambada_targets.py
Needs evaluate.py's LAMBADA samples for all three. Writes data/lambada_targets.csv.
"""
import csv
import os

import mix  # this study's shared choices; also puts analysis/ on sys.path
from common import evals  # noqa: E402

TASK = "lambada_openai"
OUT = os.path.join(mix.DATA, "lambada_targets.csv")


def main():
    names = {mix.LABELS[r]: mix.config(r)["out_dir"] for r in (mix.BASE, mix.MIX)}
    names["GPT-2 (OpenAI)"] = mix.REFERENCE
    rows = []
    for label, out_dir in names.items():
        by_id = evals.all_samples(out_dir).get(TASK)
        assert by_id, f"no {TASK} samples for {out_dir}: run evaluate.py on it"
        for g in evals.TARGET_GROUPS:
            in_group = [s for s in by_id.values() if evals.target_group(s) == g]
            rows.append(dict(model=label, group=g, **evals.lambada_summary(in_group)))
    os.makedirs(mix.DATA, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']}, {r['group']} targets ({r['n']} passages): perplexity {r['perplexity']:.1f}, "
              f"median {r['median_perplexity']:.1f}, accuracy {r['accuracy']:.3f}")


if __name__ == "__main__":
    main()
