# analysis

Scripts that turn runs and checkpoints into the post's figures. They are not needed for training.

```bash
uv sync --extra analysis
uv run --extra analysis python analysis/<study>/01_<name>.py   # from the repo root
uv sync --extra analysis --extra eval                         # scripts that score text with a checkpoint also need eval
```

**Layout:** one folder per study, i.e. per finding or section of the post.

- **Entry points** are numbered in the order they run: `01_…`, `02_…`. A later script reads an earlier one's
  outputs from `data/`, never imports it.
- **Shared code within a study** is an unnumbered module next to the scripts.
- **Code shared across studies** goes in `common/`. Today: `models.py` (load a checkpoint, per-token losses, continuation
  log-likelihoods through the eval adapter), `perplexity.py` (windowed losses — `window=` for context-length
  studies, `reshape(-1, window)` for loss vs position — per-document sums, paired bootstrap), `runs.py` (wandb
  cache), `run_ids.py`, `style.py`.
- **No hardcoded facts:** wandb run paths are typed once, in `common/run_ids.py`; everything else about a run
  (out_dir, dataset, batch settings) is fetched from wandb. Real choices a study's scripts share live once in that
  study's module.
- **Outputs:** each study writes only to its own `data/` (pulled or computed data, cached so figures rebuild
  offline) and `figures/`.

| Study | What it shows |
|---|---|
| [`step320_spike/`](step320_spike/) | A one-step loss spike shared by every run, traced to one "baby talk" page in the data |
| [`headline_comparison/`](headline_comparison/) | Model A (GPT-2) vs Model B (Gemma-style) over the 9.8B-token headline runs |
| [`efficiency/`](efficiency/) | Inference cost: generation speed with/without the KV cache, prefill, KV-cache memory |
