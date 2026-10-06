# efficiency

Cost of Model A (124M GPT-2) vs Model B (113M Gemma-style): training FLOPs and MFU, generation speed with and
without the KV cache, prefill throughput, KV-cache memory, and whether decoding is limited by memory traffic. Models are named in `common/run_ids.py`; their checkpoint folders
come from wandb.

| Script | Needs | Writes |
|---|---|---|
| `01_inference_benchmark.py` | wandb (first run, or `--refresh`), both checkpoints | `data/inference.csv` (decode and prefill timings: median of `--repeats`, after a warm-up, GPU synchronized), `data/kv_cache.csv` (cache MB per sequence, measured and from the formula, 64-1,024 tokens), `figures/inference_speed.png` (generation with speed-up labels, plus prefill), `figures/kv_cache.png`. `--plot-only` redraws the figures from the CSVs; `--tag h100` reads/writes `inference_h100.csv` and `inference_speed_h100.png`, for timings from another machine |
| `02_flops.py` | both checkpoints (01 downloads them), wandb (first run, or `--refresh`) | `data/flops_runs.json` (each headline run's batch settings, GPU, median throughput and logged MFU), `data/flops.csv` (forward FLOPs per token by component from the models' own layer shapes, training FLOPs, total compute, MFU from useful FLOPs and from the logged PaLM formula), `figures/flops.png` |
| `03_decode_bytes.py` | 01's outputs, both checkpoints | `data/decode_bytes.csv`: megabytes each decode step reads (weights + KV cache) against measured step time, with linear fits giving the effective bandwidth; `--tag h100` for another machine's timings |
| `04_devices.py` | 03's outputs for each machine | `figures/decode_regimes.png`: ms per decode step against MB read per step, one panel per machine (memory-bound vs overhead-bound) |
| `05_flops_scaling.py` | 02's outputs, both checkpoints | arithmetic only: `figures/flops_by_size.png` (each component's *share* of forward FLOPs, 124M → 6.7B, GPT-2 recipe), `figures/flops_by_context.png` (*absolute* forward GFLOPs per token, Model A's vs Model B's recipe at ~7B and ~14B, 1k vs 128k context), `data/flops_scaling.csv`, and `data/flops_scaling_facts.csv` (embedding share by size, the Chinchilla-optimal size for Model A's compute, sequences per step here and at GPT-3's batch). Prints the crossovers T ≈ 8d and d·L < V/8 |
| `06_memory_scaling.py` | 02's `flops_runs.json`, both checkpoints | arithmetic only: `data/memory_scaling.csv` (KV cache per sequence for a 7B-class model at 8k–128k: MHA, GQA, GQA + Gemma 3 local windows), `data/memory_scaling_facts.csv` (the one-GPU limit for mixed-precision Adam state, the GPU's bf16 ridge point from `curioslm/runtime.py`, decode attention intensity for Models A and B) |

Timings depend on the hardware (the device is printed and should be stated with any number); memory does not.
