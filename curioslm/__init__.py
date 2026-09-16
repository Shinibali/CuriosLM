"""
Library code for CuriosLM, a from-scratch small language model. The scripts you actually run — train.py and
sample.py — live in the parent directory; everything importable lives here.

    config.py        every tunable, declared once
    model.py         the GPT architecture
    checkpoints.py   checkpoint format: save, load, rebuild
    dataset.py       data/<name>/ layout, tokenizer, batch sampling
    runtime.py       device, precision, paths, peak-FLOPS
    configurator.py  nanoGPT's CLI/config-file override hack (exec'd, not imported)
"""
