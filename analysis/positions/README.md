# positions

How the two models represent position: RoPE's fixed frequencies (Model B), the learned position table (Model A),
how Model B's heads actually use RoPE's bands, and stretching Model B's context with YaRN. Model choices and the shared checkpoint and RoPE-frequency helpers live in `positions.py`; each model's `out_dir` is
fetched from its wandb run, and architecture comes from its checkpoint.

| Script | Needs | Writes |
|---|---|---|
| `01_rope_bands.py` | Model B's checkpoint | `data/rope_bands.csv`, `figures/rope_bands.png`: each RoPE band's wavelength (read from the model's own rotation table) against the training context |
| `02_learned_positions.py` | Model A's and Model B's checkpoints | `data/learned_positions.csv`, `figures/learned_positions.png`: Model A's position similarity matrix and top principal components, and similarity vs distance compared with Model B's RoPE (read from its checkpoint); it also prints how the learned table's variation splits by wavelength |
| `03_band_usage.py` | Model B's checkpoint | `data/band_usage.csv`, `figures/band_usage.png`: each head's share of query-key capacity in each RoPE band, from the trained weights (projection rows feeding each band's pair, with the pre-attention RMSNorm gain folded in) |
| `04_real_heads.py` | Model B's checkpoint, 03's outputs, the validation shard | `data/real_heads.npz`, `figures/real_heads.png`: three real heads (most, median and least slow-band capacity), average pre-softmax score vs distance on held-out text, split into fast / middle / slow band contributions; the band split reproduces the model's own scores exactly |
| `05_rope_scaling.py` | Model B's checkpoint | arithmetic only: `data/rope_scaling.csv`: bands slower than a 128k context for RoPE bases 10k / 500k / 1M (head size 128), Model B bands that first complete a turn at 2× / 4× / 8× context, YaRN's split of Model B's bands (from `positions.speed_groups`), and a learned position table's size at 128k |
| `06_yarn_bands.py` | Model B's checkpoint | `data/yarn_bands.csv`, `figures/yarn_bands.png`: each RoPE band's wavelength as trained and with YaRN at `positions.YARN_FACTOR` (both read from `model.py`'s rotation tables), coloured by YaRN's fast / middle / slow groups; prints how many bands complete a turn within the stretched context with and without YaRN |
| `07_yarn_context.py` | both headline checkpoints, the validation shard; `--finetuned <out_dir>` adds the YaRN fine-tune | `data/yarn_context.npz`, `data/yarn_context.csv`, `figures/yarn_context.png`: loss at every position of held-out pages at least YARN_FACTOR × 1,024 tokens long, for Model A, Model B with plain RoPE, Model B with YaRN zero-shot (and fine-tuned); perplexity of the first L tokens and loss by position, with paired page-bootstrap intervals. `--docs N` (default 500), `--plot-only` |
