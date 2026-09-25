"""
Library code for CuriosLM, a from-scratch small language model. The runnable scripts
(train.py, sample.py) live in the parent directory; everything importable lives here.

    config.py        every tunable, declared once
    model.py         the GPT architecture
    checkpoints.py   checkpoint format: save, load, rebuild
    runtime.py       device, precision, paths, peak-FLOPS
    configurator.py  nanoGPT's CLI/config-file override hack (exec'd, not imported)
    loaders/         how training gets its tokens (code, unlike top-level data/, which
                     holds the dataset files themselves):
                       dataset.py      data/<name>/ layout, tokenizer, random-sample batching
                       shard_loader.py DDP-aware sequential shard-cycling loader
                       hf_shards.py    upload/download shards to/from the private HF repo
"""
