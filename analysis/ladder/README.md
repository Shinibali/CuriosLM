# ladder

GPT-2 modernized one change at a time (rungs a0 → a4), 1.5B FineWeb-Edu tokens per rung: what each change does to
quality and to cost.

`ladder.py` reads every rung's settings by executing `config/ladder_gpu.py` the way `train.py` does, so no rung,
architecture setting or run id is typed here. Runs are found on wandb by the run name that file builds. Parameter
counts come from building each rung's model on the meta device (shapes only), which matches the checkpoints without
downloading them.

| Script | Needs | Writes |
|---|---|---|
| `01_ladder.py` | wandb (first run, or `--refresh`) | `data/ladder.csv` (per rung: change, true params, final val loss and step vs previous rung, median training tokens/s, peak memory, KV-cache MB at the full context), `figures/ladder_loss.png` (each rung's step against the seed-to-seed noise band), `figures/ladder_efficiency.png` (training throughput and KV-cache size per rung) |
