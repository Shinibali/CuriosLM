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
  log-likelihoods through the eval adapter, the KV-cache size formula), `perplexity.py` (windowed losses — `window=` for context-length
  studies, `reshape(-1, window)` for loss vs position — per-document sums, paired bootstrap), `evals.py` (evaluate.py results, paired McNemar and bootstrap comparisons, the LAMBADA target split), `runs.py` (wandb
  cache), `run_ids.py`, `reference_models.py` (the model shapes the scaling scripts extrapolate to), `style.py`.
- **No hardcoded facts:** wandb run paths are typed once, in `common/run_ids.py`; everything else about a run
  (out_dir, dataset, batch settings) is fetched from wandb. Real choices a study's scripts share live once in that
  study's module.
- **Outputs:** each study writes only to its own `data/` (pulled or computed data, cached so figures rebuild
  offline) and `figures/`.

| Study | What it shows |
|---|---|
| [`step320_spike/`](step320_spike/) | A one-step loss spike shared by every run, traced to one "baby talk" page in the data |
| [`headline_comparison/`](headline_comparison/) | Model A (GPT-2) vs Model B (Gemma-style) over the 9.8B-token headline runs |
| [`data_mix/`](data_mix/) | Model B on 100% FineWeb-Edu vs a 50:50 FineWeb-Edu : Cosmopedia-v2 blend: loss, benchmarks, perplexity, LAMBADA |
| [`ladder/`](ladder/) | GPT-2 modernized one change at a time: each change's effect on loss, training speed and KV-cache size |
| [`positions/`](positions/) | How each model represents position: RoPE's frequency bands, learned position embeddings as waves, how Model B's heads use the bands, and stretching Model B's context 4× with YaRN |
| [`attention/`](attention/) | What each model attends to: how far each layer looks and the attention sink, induction heads inside and beyond Model B's window, where the models disagree most (per-token KL), and an emoji probe |
| [`efficiency/`](efficiency/) | Inference cost: generation speed with/without the KV cache, prefill, KV-cache memory |
