# data_mix

Model B's architecture trained twice on 9.8B tokens: once on 100% FineWeb-Edu (the headline Model B), once on a
deterministic 50:50 blend of FineWeb-Edu and Cosmopedia-v2. Only the data differs. Model A and stock GPT-2 appear as
references.

`mix.py` declares the runs compared (by wandb run path) and the bootstrap size. Each run's `out_dir` and data mix
are fetched from wandb; the validation sets are the datasets in the mix.

| Script | Needs | Writes |
|---|---|---|
| `01_loss_curves.py` | wandb (first run, or `--refresh`) | `data/run_configs.json`, `data/val_history.csv`, `figures/loss_curves.png`: validation loss on each dataset over training |
| `02_benchmarks.py` | `evaluate.py` run on both models | `data/benchmarks.csv`: paired McNemar per accuracy, paired bootstrap on LAMBADA perplexity, 50:50 vs 100% FineWeb-Edu |
| `03_val_perplexity.py` | the checkpoints, each dataset's validation shard, the `eval` extra | `data/val_perplexity.csv`: held-out perplexity of all four models on each dataset, with a paired document-level bootstrap for 50:50 vs 100% FineWeb-Edu |
| `04_lambada_targets.py` | `evaluate.py` LAMBADA samples | `data/lambada_targets.csv`: LAMBADA by capitalized (mostly names) vs lowercase target |
| `05_tradeoff.py` | 02's and 03's outputs | `figures/tradeoff.png`: every measure's change from 100% FineWeb-Edu to the 50:50 mix (perplexity in %, accuracy in points with paired p-values) |
