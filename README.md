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

Two optional extras, for work beyond training:
```bash
uv sync --extra eval        # lm-eval, for evaluate.py
uv sync --extra analysis    # matplotlib + circuitsvis, for analysis/
```

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
LADDER_RUNG=a2 LADDER_DATASET=cosmopedia_v2 uv run ...   # same rung on Cosmopedia-v2 instead of FineWeb-Edu
LADDER_RUNG=a4 LADDER_SEED=1338 uv run ...                # a second seed of a rung (default 1337)
LADDER_RUNG=a4 LADDER_BUDGET=9.8B uv run ...              # the headline runs: 20,000 steps, ~9.8B tokens
```
On several GPUs, launch with `torchrun --standalone --nproc_per_node=<N> train.py config/ladder_gpu.py ...`.
One file covers the whole ladder, on any dataset — `config/ladder_gpu.py` holds the shared
settings (124M size, training hyperparameters) plus a `RUNGS` dict of each rung's architecture
delta over the one before it. `LADDER_RUNG` picks the rung, `LADDER_DATASET` picks the dataset
(default `fineweb_edu`); rerunning the ladder on a second dataset is just a different env var,
not a second config file. `out_dir`/`wandb_run_name` include the dataset, the budget (when not 1.5B) and
the seed (when set), so runs never collide. `a0`'s entry is empty on purpose — `GPTConfig`'s defaults already
are the vanilla baseline. Batch settings are calibrated for H100s (micro-batch 60) and gradient accumulation is
global — `train.py` divides it by the GPU count — so every step is 491,520 tokens on 1, 2, 4 or 8 GPUs.

**The YaRN fine-tune (Model B, 1,024 → 4,096 tokens):**
```bash
uv run python train.py config/yarn_finetune.py --device=cuda
```
Continues the 9.8B a4 run (`init_from='extend'`, `extend_from=` its `out_dir`) with every headline setting
unchanged (the config execs `ladder_gpu.py`) except: `block_size` 4,096 with `yarn_factor` 4 from
`yarn_original_context` 1,024, micro-batch 15 so a step is still 491,520 tokens, `long_pages=True` (each sequence is
one FineWeb-Edu page of 4,096+ tokens; `curioslm/loaders/long_page_loader.py`), 200 more steps at the schedule's
`min_lr`, and `snapshot_interval=25` (weights-only `ckpt-<iter>.pt` copies, never overwritten). In-run evals happen
at the snapshot steps on 20 batches, and `checkpoint_interval` writes the resumable `ckpt.pt` only halfway and at
the end (0, the default, saves at every eval as the pretraining runs do); each eval and save prints its time. Writes
to `<headline out_dir>-yarn4x` as a new wandb run in `curioslm-yarn`; the headline checkpoint is only read.

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

**Cosmopedia-v2** (for the data-mix run — a 51-shard, ~5.1B-token slice of the corpus, seeded shuffle, ~30 min):
```bash
uv run python data/cosmopedia_v2/prepare.py
```
Both are thin settings files over `data/sharding.py` (tokenize, shard, resume). Safe to interrupt (Ctrl+C) and
rerun — they checkpoint after every shard and resume automatically, never duplicating already-written data.

## FineWeb-Edu shards <-> private HF staging repo

```bash
uv run python -m curioslm.loaders.hf_shards upload     # local .npy shards -> the private repo (default if no subcommand given)
uv run python -m curioslm.loaders.hf_shards download   # private repo -> local data/fineweb_edu/
uv run python -m curioslm.loaders.hf_shards cleanup    # delete local shards ONLY once confirmed uploaded (matching filename + byte size)
```
Staging between local prep and wherever GPU training actually runs; `curioslm/config.py`'s `DataConfig.hf_dataset_repo_id` names the repo. The CLI handles `DataConfig`'s dataset (FineWeb-Edu); for Cosmopedia-v2, call the same functions with its names, e.g.
`uv run python -c "from curioslm.loaders.hf_shards import upload_shards; upload_shards('cosmopedia_v2', 'CuriosLM/cosmopedia_v2-shards')"`. Requires being logged in (`hf auth login`) with write access to it. Upload/download are bandwidth-bound (~19GB). `cleanup` only deletes local shards it has verified are on the remote with a matching size — anything unconfirmed is left alone and reported.

## Checkpoints <-> private HF staging repo

Fully automatic — no command to run. Rented GPU pods have ephemeral disk, so `train.py` uploads
`out_dir/ckpt.pt` to the private `CuriosLM/checkpoints-staging` repo
(`IOConfig.hf_checkpoint_repo_id`) after every local save, on a background thread so training
never blocks on the transfer. The remote path mirrors `out_dir` (e.g.
`out-ladder-fineweb_edu-a2/ckpt.pt`), so the repo listing itself says which run each checkpoint
came from. Each upload then prunes the repo — squashes its history and permanently deletes replaced versions older
than 10 minutes — since an overwritten file otherwise keeps every old version and counts it against the
private-storage quota; the repo holds roughly one checkpoint per run. `init_from='resume'` pulls from there when the checkpoint isn't on the local disk,
as does `init_from='extend'` for the run it continues; snapshots (`snapshot_interval`) are uploaded beside `ckpt.pt`.
`init_from='scratch'` and `'extend'` refuse to start if `out_dir` already holds a *different* run's checkpoint
(compared by `wandb_run_name`), so an `out_dir` collision fails loudly rather than silently
overwriting. Separate from the polished, public model repos, which get pushed by hand with a
model card when a headline model is finalized.

## GPU pods

```bash
ssh -i ~/.ssh/runpod_ed25519 -p <port> root@<ip> 'bash -s' < scripts/pod_setup.sh
```
One command from the Mac prepares a fresh pod: checks credentials, installs uv, clones, `uv sync --locked`,
checks CUDA, downloads the shards. Credentials come from the pod's environment, never the repo.

## Evaluate

```bash
uv run --extra eval python evaluate.py out-ladder-fineweb_edu-a4 --tasks hellaswag --limit 50    # smoke test
uv run --extra eval python evaluate.py out-ladder-fineweb_edu-a0-9.8B out-ladder-fineweb_edu-a4-9.8B gpt2
```
Benchmarks (default: HellaSwag, ARC-Easy, LAMBADA) run through `curioslm/eval_adapter.py`, which plugs our
checkpoints into [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — task
definitions and scoring are lm-eval's own, so numbers compare with published ones. Each argument is a run
directory (its checkpoint is downloaded from the staging repo if missing) or `gpt2` for OpenAI's GPT-2 small
as a reference. Writes `evals/<model>/results.json` plus per-example samples for paired significance tests.
Runs fine on a Mac (MPS); `--device` overrides.

## Analysis

Scripts that turn runs and checkpoints into figures live in `analysis/`, one folder per study — see
[`analysis/README.md`](analysis/README.md). Run them from the repo root with `uv run --extra analysis ...`.

## Tests

```bash
uv run --extra eval python -m pytest -q
```
The eval-adapter tests need the `eval` extra and are skipped without it.
