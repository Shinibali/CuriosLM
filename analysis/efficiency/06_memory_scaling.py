"""Memory and bandwidth at larger scale, by arithmetic alone: the KV-cache formula checked against the trained models
(01's measured caches), applied to a 7B-class model at long contexts, plus the memory facts the post quotes.

1. KV cache per sequence for a 7B-class model at 8k–128k tokens: full multi-head attention, GQA, and GQA plus Gemma
   3-style local windows.
2. The largest model whose mixed-precision Adam state fits one GPU.
3. Decode arithmetic intensity: the H100's ridge point (FLOPs per byte it can sustain) against reading the KV cache,
   whose intensity is query heads per key/value head, whatever the batch.

    uv run --extra analysis python analysis/efficiency/06_memory_scaling.py
Needs 02's data/flops_runs.json (the GPU the runs used) and both checkpoints. Writes data/memory_scaling.csv and
data/memory_scaling_facts.csv.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.dirname(HERE), ROOT]  # analysis/ for common, the repo root for curioslm
from common import models  # noqa: E402
from common.reference_models import LONG_CONTEXT, SEVEN_B, VOCAB  # noqa: E402
from curioslm import checkpoints, runtime  # noqa: E402
from curioslm.config import GPTConfig  # noqa: E402

DATA = os.path.join(HERE, "data")
# Choices: the long-context variants compared, and the dtype the cache is served in
CONTEXTS = (8192, 32768, LONG_CONTEXT)
GQA_KV_HEADS = 8  # 4 query heads per key/value head, as in Model B
GEMMA3_WINDOW, GEMMA3_LOCAL_GLOBAL_RATIO = 1024, 6  # Gemma 3's layout: 1,024-token windows, 1 global layer in 6
KV_BYTES = 2  # bf16, as served
# Facts quoted from cited sources (used only in the arithmetic below)
ADAM_BYTES_PER_PARAM = 16  # bf16 weights + grads, fp32 master weights + two Adam moments (Rajbhandari et al., ZeRO)
GPU_MEMORY_GB = 80  # H100 80GB
H100_BANDWIDTH = 3.35e12  # HBM3 bytes/s, H100 SXM


def kv_cache_rows():
    """KV cache per sequence (GB) for the 7B-class shape at each context, three attention variants."""
    rows = []
    for context in CONTEXTS:
        variants = {
            "MHA, all global": GPTConfig(**SEVEN_B, block_size=context, vocab_size=VOCAB),
            "GQA, all global": GPTConfig(**SEVEN_B, block_size=context, vocab_size=VOCAB, n_kv_head=GQA_KV_HEADS),
            "GQA + Gemma 3 local windows": GPTConfig(
                **SEVEN_B, block_size=context, vocab_size=VOCAB, n_kv_head=GQA_KV_HEADS,
                attention_window=min(GEMMA3_WINDOW, context), local_global_ratio=GEMMA3_LOCAL_GLOBAL_RATIO),
        }
        for variant, config in variants.items():
            rows.append(dict(context=context, variant=variant,
                             kv_gb_per_sequence=models.kv_bytes_formula(config, context, 1, KV_BYTES) / 1e9))
    return rows


def memory_facts():
    """(fact, value) rows: the one-GPU Adam limit, the ridge point, decode attention intensity for Models A and B."""
    with open(os.path.join(DATA, "flops_runs.json")) as f:
        runs = json.load(f)  # 02_flops.py's record of the headline runs, including the GPU they ran on
    gpu = next(iter(runs.values()))["gpu"]
    facts = [("params whose bf16 + Adam state fits one GPU (billions)",
              GPU_MEMORY_GB * 1e9 / ADAM_BYTES_PER_PARAM / 1e9),
             (f"ridge point, bf16 FLOPs per byte, {gpu}", runtime.peak_flops_for_name(gpu) / H100_BANDWIDTH)]
    for label, run in zip(("Model A", "Model B"), runs.values()):  # 02 records A then B
        model_args = checkpoints.load(os.path.join(ROOT, run["out_dir"], "ckpt.pt"), map_location="cpu")["model_args"]
        config = GPTConfig(**model_args)
        # per cached position: 4·n_h·h FLOPs (all query heads) against 2·n_kv·h values × 2 bytes → n_h / n_kv per byte
        facts.append((f"decode attention intensity, FLOPs per byte (bf16), {label}",
                      config.n_head / config.num_kv_heads))
    return facts


def main():
    os.makedirs(DATA, exist_ok=True)
    rows = kv_cache_rows()
    for row in rows:
        print(f"{row['context']:,} context, {row['variant']}: KV cache {row['kv_gb_per_sequence']:.2f} GB per sequence")
    facts = memory_facts()
    for name, value in facts:
        print(f"  {name}: {value:,.3f}")
    with open(os.path.join(DATA, "memory_scaling.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(DATA, "memory_scaling_facts.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fact", "value"])
        writer.writerows(facts)


if __name__ == "__main__":
    main()
