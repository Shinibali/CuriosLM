"""Run lm-eval benchmarks on CuriosLM checkpoints, or stock GPT-2 as a reference, through the curioslm adapter.

    uv run --extra eval python evaluate.py out-ladder-fineweb_edu-a4 --tasks hellaswag --limit 50   # smoke test
    uv run --extra eval python evaluate.py out-ladder-fineweb_edu-a0-9.8B out-ladder-fineweb_edu-a4-9.8B gpt2

Writes evals/<model>/results.json and per-example samples_<task>.jsonl (for the paired significance tests).
"""
import argparse
import json
import os

import lm_eval
from lm_eval.utils import make_table

from curioslm.eval_adapter import CurioslmLM

TASKS = ["hellaswag", "arc_easy", "lambada_openai"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="run directories (out-...) and/or gpt2")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--limit", type=int, default=None, help="examples per task, for smoke tests")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="evals")
    args = parser.parse_args()

    for name in args.models:
        lm = CurioslmLM(out_dir=name, device=args.device, batch_size=args.batch_size)
        results = lm_eval.simple_evaluate(model=lm, tasks=args.tasks, limit=args.limit, log_samples=True)
        out_dir = os.path.join(args.out, os.path.basename(os.path.normpath(name)))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump({"model": name, "limit": args.limit, "results": results["results"]}, f, indent=2, default=str)
        for task, samples in results["samples"].items():
            with open(os.path.join(out_dir, f"samples_{task}.jsonl"), "w") as f:
                f.writelines(json.dumps(s, default=str) + "\n" for s in samples)
        print(f"== {name}\n{make_table(results)}")


if __name__ == "__main__":
    main()
