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

## Sample

```bash
uv run python sample.py --out_dir=out-shakespeare --device=mps --num_samples=2 --max_new_tokens=100
```

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
Staging between local prep and wherever GPU training actually runs — see `S1_REQUIREMENTS.md` for the full reasoning, or `curioslm/config.py`'s `DataConfig.hf_repo_id` for which repo. Requires being logged in (`huggingface-cli login`) with write access to that repo. Upload/download are bandwidth-bound (~19GB) — expect real time depending on your connection. `cleanup` never deletes anything it hasn't verified is safely on the remote repo first — anything unconfirmed is left alone and reported.

## Tests

```bash
uv run python -m pytest -q
```
