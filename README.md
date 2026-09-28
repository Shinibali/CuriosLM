# CuriosLM

From-scratch GPT-2 → Gemma architecture ladder: a small language model built up rung by rung
(RoPE, GQA, GeGLU + sandwich norm, local/global sliding-window attention) from a vanilla
nanoGPT-style baseline.

This repo is self-contained — cloning it and running the commands below is everything needed
to set it up.

## First-time environment setup (new clone, or a new machine)

```bash
git clone <this-repo-url>
cd CuriosLM
uv sync
```

This assumes `uv` itself is already installed (a one-time global tool install, separate from
this project — see [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) if
not). `uv sync` reads this repo's own `pyproject.toml`/`uv.lock` and builds a `.venv` right
here with every dependency locked to an exact, reproducible version.

## Setup (every session)

No manual venv activation needed — every command below uses `uv run`, which finds this
repo's `pyproject.toml` automatically (walks up from wherever you run it) and ensures the
environment is up to date before running.

## Train

```bash
uv run python train.py config/train_shakespeare.py --device=mps --compile=True --position_encoding=rope --max_iters=100 --eval_interval=50 --eval_iters=5 --mlp_type=geglu --norm_placement=sandwich --local_global_ratio=4
```

Shakespeare smoke-test recipe (`--device=mps` is Apple Silicon; use `--device=cuda` or
`--device=cpu` as applicable).

**FineWeb-Edu, Model B recipe** (RoPE + GQA + GeGLU/sandwich norm + local/global attention):
```bash
uv run python train.py config/train_fineweb_edu.py --device=mps
```
Everything specific to this recipe (dataset, architecture flags, `out_dir`) lives in the
config file itself — no need to repeat `--position_encoding=rope --mlp_type=geglu` etc. on
the command line every time. Auto-downloads any missing shards from the private HF staging
repo before training starts (see the shard upload/download section below) — the first run
on a fresh machine will pull the full ~19GB before anything trains. `--max_iters=`/
`--eval_interval=` etc. still override normally for anything beyond the dev-scale defaults.

**The architecture ladder (A0-A4, real 124M/~1.5B-token scale, not the dev configs above):**
```bash
LADDER_RUNG=a2 uv run python train.py config/ladder_gpu.py --device=cuda
# rerun the same ladder on Cosmopedia-v2 instead of FineWeb-Edu (Phase 6's robustness check):
LADDER_RUNG=a2 LADDER_DATASET=cosmopedia_v2 uv run python train.py config/ladder_gpu.py --device=cuda
```
One file covers the whole ladder, on any dataset — `config/ladder_gpu.py` holds the shared
settings (124M size, training hyperparameters) plus a `RUNGS` dict of each rung's architecture
delta over the one before it. `LADDER_RUNG` picks the rung, `LADDER_DATASET` picks the dataset
(default `fineweb_edu`); rerunning the ladder on a second dataset is just a different env var,
not a second config file. `out_dir`/`wandb_run_name` always include the dataset name, so the
same rung on two datasets never collides. `a0`'s entry is empty on purpose — `GPTConfig`'s
defaults already are the vanilla baseline. `batch_size`/`gradient_accumulation_steps`/
`max_iters`/`lr_decay_iters` are placeholders sized for a single GPU — recompute for the actual
GPU/node count, since `max_iters` has to hit the target token budget at whatever
`tokens_per_iter` that hardware gives.

## Sample

```bash
uv run python sample.py --out_dir=out-shakespeare --device=mps --num_samples=2 --max_new_tokens=100
```
`--out_dir` picks the run. If `<out_dir>/ckpt.pt` isn't on local disk (e.g. it was trained on a GPU pod),
it's downloaded from the checkpoint staging repo first — so `--out_dir=out-ladder-fineweb_edu-a2` samples
that rung wherever it was trained.

## Data prep

**Shakespeare** (tiny dataset, same one Karpathy's char-rnn project made famous — but tokenized here with GPT-2 BPE via `tiktoken`; used for the smoke tests above):
```bash
uv run python data/shakespeare/prepare.py
```

**FineWeb-Edu** (real ladder/headline-run data — `sample-10BT`, ~100 shards, ~19GB, ~25 min real wall-clock):
```bash
uv run python data/fineweb_edu/prepare.py
```
Safe to interrupt (Ctrl+C) and rerun — it checkpoints after every shard and resumes automatically, never re-downloading or duplicating already-written data.

## FineWeb-Edu shards <-> private HF staging repo

```bash
uv run python -m curioslm.loaders.hf_shards upload     # local .npy shards -> the private repo (default if no subcommand given)
uv run python -m curioslm.loaders.hf_shards download   # private repo -> local data/fineweb_edu/
uv run python -m curioslm.loaders.hf_shards cleanup    # delete local shards ONLY once confirmed uploaded (matching filename + byte size)
```
Staging between local prep and wherever GPU training actually runs; `curioslm/config.py`'s `DataConfig.hf_dataset_repo_id` names the repo. Requires being logged in (`hf auth login`) with write access to it. Upload/download are bandwidth-bound (~19GB). `cleanup` only deletes local shards it has verified are on the remote with a matching size — anything unconfirmed is left alone and reported.

## Checkpoints <-> private HF staging repo

Fully automatic — no command to run. Rented GPU pods have ephemeral disk, so `train.py` uploads
`out_dir/ckpt.pt` to the private `CuriosLM/checkpoints-staging` repo
(`IOConfig.hf_checkpoint_repo_id`) after every local save, on a background thread so training
never blocks on the transfer. The remote path mirrors `out_dir` (e.g.
`out-ladder-fineweb_edu-a2/ckpt.pt`), so the repo listing itself says which run each checkpoint
came from. `init_from='resume'` pulls from there when the checkpoint isn't on the local disk;
`init_from='scratch'` refuses to start if `out_dir` already holds a *different* run's checkpoint
(compared by `wandb_run_name`), so an `out_dir` collision fails loudly rather than silently
overwriting. Separate from the polished, public model repos, which get pushed by hand with a
model card when a headline model is finalized.

## Tests

```bash
uv run python -m pytest -q
```
