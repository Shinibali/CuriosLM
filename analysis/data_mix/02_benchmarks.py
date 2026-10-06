"""Did swapping half the data change the benchmarks? Paired tests of the 50:50 run against Model B (same
architecture, 100% FineWeb-Edu), on the same questions.

    uv run --extra analysis python analysis/data_mix/02_benchmarks.py
Needs evaluate.py run on both models. Writes data/benchmarks.csv.
"""
import csv
import os

import mix  # this study's shared choices; also puts analysis/ on sys.path
from common import evals  # noqa: E402

OUT = os.path.join(mix.DATA, "benchmarks.csv")


def main():
    base, mixed = (mix.config(r)["out_dir"] for r in (mix.BASE, mix.MIX))
    rows = evals.compare(evals.all_samples(base), evals.all_samples(mixed), mix.BOOTSTRAP)  # "a" = base, "b" = mix
    os.makedirs(mix.DATA, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    evals.print_comparison(rows, "100% FW", "50:50")


if __name__ == "__main__":
    main()
