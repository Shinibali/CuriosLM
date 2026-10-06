"""Shared by this study's scripts: the ladder's rungs and settings, read from the committed config, and the choices
the scripts share. No rung, run id or architecture setting is typed here: each rung's settings come from executing
config/ladder_gpu.py the way train.py does, and its wandb run is found by the run name that file builds."""
import dataclasses
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(ANALYSIS)
for path in (ROOT, ANALYSIS):  # curioslm, and analysis/common
    if path not in sys.path:
        sys.path.insert(0, path)
from curioslm.config import GPTConfig  # noqa: E402
from curioslm.model import GPT  # noqa: E402
from common import run_ids  # noqa: E402

DATA, FIGURES = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
CONFIG_FILE = os.path.join(ROOT, "config", "ladder_gpu.py")

# Choices
NOISE_RUNS = (run_ids.LADDER_A4, run_ids.LADDER_A4_SEED2)  # one rung, two seeds: the seed-to-seed noise band
# A rung's step is itself a difference of two single-seed runs, like the seed pair: one seed pair is a rough
# estimate of that noise, so only steps well beyond it count as clear.
CLEAR = 2
KV_BYTES = 4  # bytes per cached value: float32, as in the efficiency study's measurements
CHANGE = {  # each rung's one change, for labels (the settings themselves come from the config file)
    "a0": "GPT-2 baseline", "a1": "+ RoPE", "a2": "+ GQA", "a3": "+ GeGLU, sandwich norm", "a4": "+ local/global",
}


def config_values(rung):
    """The globals config/ladder_gpu.py sets for a rung (the 1.5B-token budget, the default seed and dataset)."""
    saved = {k: os.environ.pop(k) for k in ("LADDER_BUDGET", "LADDER_SEED", "LADDER_MIX", "LADDER_DATASET")
             if k in os.environ}  # the ladder runs used every default
    os.environ["LADDER_RUNG"] = rung
    try:
        values = {}
        with open(CONFIG_FILE) as f:
            exec(f.read(), values)  # as train.py's configurator does
    finally:
        os.environ.pop("LADDER_RUNG")
        os.environ.update(saved)
    return values


def rungs():
    """The rung names, in ladder order, from the config file's RUNGS table."""
    return list(config_values("a0")["RUNGS"])


def model_config(values):
    """GPTConfig from a config file's globals: only the fields GPTConfig has, the rest keep their defaults."""
    fields = {f.name for f in dataclasses.fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in values.items() if k in fields})


def n_params(config):
    """True parameter count, position embeddings included (nanoGPT's logged n_params leaves them out). The model is
    built on the meta device: shapes only, no memory, so this matches the checkpoint without downloading it."""
    with torch.device("meta"):
        return sum(p.numel() for p in GPT(config).parameters())
