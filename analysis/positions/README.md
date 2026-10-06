# positions

How the two models represent position: RoPE's fixed frequencies (Model B), the learned position table (Model A), and
how Model B's heads actually use RoPE's bands. Model choices and the shared checkpoint and RoPE-frequency helpers live in `positions.py`; each model's `out_dir` is
fetched from its wandb run, and architecture comes from its checkpoint.

| Script | Needs | Writes |
|---|---|---|
| `01_rope_bands.py` | Model B's checkpoint | `data/rope_bands.csv`, `figures/rope_bands.png`: each RoPE band's wavelength (read from the model's own rotation table) against the training context |
| `02_learned_positions.py` | Model A's and Model B's checkpoints | `data/learned_positions.csv`, `figures/learned_positions.png`: Model A's position similarity matrix and top principal components, and similarity vs distance compared with Model B's RoPE (read from its checkpoint); it also prints how the learned table's variation splits by wavelength |
| `03_band_usage.py` | Model B's checkpoint | `data/band_usage.csv`, `figures/band_usage.png`: each head's share of query-key capacity in each RoPE band, from the trained weights (projection rows feeding each band's pair, with the pre-attention RMSNorm gain folded in) |
| `04_real_heads.py` | Model B's checkpoint, 03's outputs, the validation shard | `data/real_heads.npz`, `figures/real_heads.png`: three real heads (most, median and least slow-band capacity), average pre-softmax score vs distance on held-out text, split into fast / middle / slow band contributions; the band split reproduces the model's own scores exactly |
| `05_rope_scaling.py` | Model B's checkpoint | arithmetic only: `data/rope_scaling.csv`: bands slower than a 128k context for RoPE bases 10k / 500k / 1M (head size 128), Model B bands that first complete a turn at 2× / 4× / 8× context, YaRN's split of Model B's bands (from `positions.speed_groups`), and a learned position table's size at 128k |
