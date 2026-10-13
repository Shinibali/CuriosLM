"""Is decoding limited by memory traffic? Decode step time from 01's measurements against the bytes each step reads.

Every decode step reads all the weights once (shared by the batch) plus each sequence's whole KV cache, which grows
by one position per step. Averaged over the generated tokens, that gives megabytes read per step. If decode is
memory-bound, step time grows linearly with it, and the slope is the effective bandwidth. A second fit adds the batch
size, for per-sequence work that isn't memory traffic.

    uv run --extra analysis python analysis/efficiency/03_decode_bytes.py [--tag h100]
Needs 01's data/inference.csv (or inference_<tag>.csv) and run_configs.json, and both checkpoints (for parameter
counts and cache shapes). Writes data/decode_bytes.csv (or decode_bytes_<tag>.csv).
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.dirname(HERE), ROOT]  # analysis/ for common, the repo root for curioslm
from common import models, run_ids, runs  # noqa: E402

DATA = os.path.join(HERE, "data")
LABELS = {"124M GPT-2 (Model A)": run_ids.HEADLINE_A, "113M Gemma 3-style (Model B)": run_ids.HEADLINE_B}  # as in 01
BYTES = 4  # float32, the dtype 01 measured in


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="machine label of 01's timing file, e.g. h100")
    tag = parser.parse_args().tag
    suffix = f"_{tag}" if tag else ""
    with open(os.path.join(DATA, f"inference{suffix}.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["phase"] == "decode" and r["kv_cache"] == "True"]
    configs = runs.load_configs(list(LABELS.values()), os.path.join(DATA, "run_configs.json"))
    loaded = {}
    for label, path in LABELS.items():
        model = models.load_model(configs[path]["out_dir"], "cpu")
        loaded[label] = (sum(p.numel() for p in model.parameters()) * BYTES, model.config)
        del model
    new_tokens = None
    out = []
    for r in rows:
        weights, config = loaded[r["model"]]
        batch, prompt = int(r["batch"]), int(r["tokens"])
        seconds = float(r["seconds"])
        new_tokens = round(float(r["tokens_per_s"]) * seconds / batch)  # tokens generated per sequence
        cache = [models.kv_bytes_formula(config, prompt + i, batch, BYTES) for i in range(new_tokens)]  # per step
        out.append(dict(model=r["model"], batch=batch, prompt=prompt,
                        mb_per_step=(weights + sum(cache) / len(cache)) / 1e6, ms_per_step=1000 * seconds / new_tokens))
    mb = np.array([o["mb_per_step"] for o in out])
    ms = np.array([o["ms_per_step"] for o in out])
    batch = np.array([o["batch"] for o in out])
    for name, X in (("bytes only", np.c_[np.ones(len(mb)), mb]), ("bytes + batch", np.c_[np.ones(len(mb)), mb, batch])):
        coef, *_ = np.linalg.lstsq(X, ms, rcond=None)  # least squares: ms ≈ X @ coef
        r2 = 1 - ((ms - X @ coef) ** 2).sum() / ((ms - ms.mean()) ** 2).sum()
        bandwidth = f"effective {1 / coef[1]:.0f} GB/s" if coef[1] > 0 else "time doesn't grow with bytes"
        print(f"{name}: R² {r2:.2f}, slope {coef[1]:.4f} ms/MB, {bandwidth}")
    with open(os.path.join(DATA, f"decode_bytes{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(f"{o['model']}, batch {o['batch']}, prompt {o['prompt']}: {o['mb_per_step']:.0f} MB/step, "
              f"{o['ms_per_step']:.1f} ms/step")


if __name__ == "__main__":
    main()
