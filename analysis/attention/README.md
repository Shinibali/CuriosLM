# attention

What each model attends to, read from its own attention probabilities: the explicit attention path in `model.py`
(`GPT.attention_weights`, evaluation only). All figures are static (matplotlib). Model choices, the probe text
and token labels live in `attention.py`; each model's `out_dir` is fetched from its wandb run, and architecture comes
from its checkpoint.

| Script | Needs | Writes |
|---|---|---|
| `01_emoji_probe.py` | both headline checkpoints | `data/emoji_attention.npz`, `figures/emoji_tokens_a.png` and `emoji_tokens_b.png` (per model, token-to-token attention in every layer, mean of its heads, on the probe and its first copied line), `figures/emoji_attention.png`: attention from the token that completes each emoji, per layer (mean over heads); prints each model's next-token guesses at every emoji token, and its greedy and two seeded sampled 50-token continuations. `--plot-only`; `--text-only` prints the continuations only (needs no attention path) |
| `02_disagreement.py` | both headline checkpoints, the validation shard | `data/disagreement.npz`, `figures/disagreement.png`: the distribution of per-token KL(A ‖ B), and each model's attention from the strongest-disagreement token; prints the 10 strongest disagreements with context, the true token and each model's guess. `--windows N` (default 20), `--plot-only` |
| `03_attention_distance.py` | both headline checkpoints, the validation shard | `data/attention_distance.npz`, `figures/attention_distance.png`: for every head of both models, the mean attention distance (Σ p·(q − k), Dosovitskiy et al. 2020) the share of attention on the window's first position (the attention sink, Xiao et al. 2023), and the share on document-separator tokens (a sink tied to a token rather than a position), over the second half of each full-length held-out window; Model B's global layers shaded. `--windows N` (default 32), `--plot-only` |
| `04_induction.py` | both headline checkpoints | `data/induction.npz`, `figures/induction_heads.png` (each model's strongest induction head, token to token, on a short repeated block), `figures/induction.png`: induction score per head (Olsson et al. 2022) with a repeated random block 0.5× and 1.5× Model B's local window back, and each model's loss on the second copy. `--sequences N` (default 16), `--plot-only` |

All four need the explicit attention path in `model.py` (`CausalSelfAttention._attention_with_weights`); its contract is
`tests/curioslm/test_attention_weights.py`.
