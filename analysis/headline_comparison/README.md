# headline_comparison

Model A (124M GPT-2) vs Model B (113M Gemma-style), each trained on the same 9.8B FineWeb-Edu tokens.

`headline.py` declares, once, which runs are compared (by wandb run path) and the choices every script shares
(the GPT-2 reference, the seed-noise pair, extra validation sets, bootstrap size). Facts about the runs — their
`out_dir` and training dataset — aren't typed anywhere: `01` fetches them from each run's wandb config.

| Script | Needs | Writes |
|---|---|---|
| `01_loss_overlay.py` | wandb (first run, or `--refresh`) | `data/val_history.csv`, `data/run_configs.json` (each run's `out_dir`, dataset, GPU count), `figures/loss_overlay.png`: validation loss vs tokens, and the B − A gap against the seed-to-seed noise band |
| `02_significance.py` | 01's outputs; `evaluate.py` run on both models | `data/significance.csv`: paired McNemar test per benchmark accuracy, paired bootstrap on LAMBADA perplexity. Finds each model's samples through the `model` field in `evals/**/results.json` |
| `03_val_perplexity.py` | 01's outputs, the validation shards, both checkpoints, the `eval` extra | `data/val_perplexity.csv`, `data/val_doc_losses.npz`: perplexity of A, B and GPT-2 on the runs' training set and Cosmopedia-v2 val, with a paired document-level bootstrap |
| `04_lambada_targets.py` | 01's outputs; `evaluate.py` LAMBADA samples for A, B and GPT-2 | `data/lambada_targets.csv`, `figures/lambada_targets.png`: LAMBADA perplexity, median and accuracy split by capitalized (mostly names) vs lowercase target, plus the passages where GPT-2 beats Model A most |
