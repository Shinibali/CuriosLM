# efficiency

Inference cost of Model A (124M GPT-2) vs Model B (113M Gemma-style): generation speed with and without the KV
cache, prefill throughput, and KV-cache memory. Models are named in `common/run_ids.py`; their checkpoint folders
come from wandb.

| Script | Needs | Writes |
|---|---|---|
| `01_inference_benchmark.py` | wandb (first run, or `--refresh`), both checkpoints | `data/inference.csv` (decode and prefill timings: median of `--repeats`, after a warm-up, GPU synchronized), `data/kv_cache.csv` (cache MB per sequence, measured and from the formula, 64-1,024 tokens), `figures/inference_speed.png` (generation with speed-up labels, plus prefill), `figures/kv_cache.png`. `--plot-only` redraws the figures from the CSVs |

Timings depend on the hardware (the device is printed and should be stated with any number); memory does not.
