"""
Library code for CuriosLM, a from-scratch small language model. The scripts you actually run — train.py and
sample.py — live in the parent directory; everything importable lives here.

    config.py        every tunable, declared once
    model.py         the GPT architecture
    checkpoints.py   checkpoint format: save, load, rebuild
    runtime.py       device, precision, paths, peak-FLOPS
    configurator.py  nanoGPT's CLI/config-file override hack (exec'd, not imported)
    loaders/         everything about how training gets its tokens (named apart from the
                     top-level data/ directory -- that one holds actual dataset files, this
                     one holds the code that reads them):
                       dataset.py      data/<name>/ layout, tokenizer, single-file random-sample batching
                       shard_loader.py DDP-aware sequential shard-cycling loader (FineWeb-Edu scale)
                       hf_shards.py    upload/download shards to/from the private HF staging repo
"""
