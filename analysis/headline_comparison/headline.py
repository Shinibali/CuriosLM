"""Shared by this study's scripts: which runs are compared and the analysis choices they share — declared once here.
Facts about the runs (out_dir, dataset, ...) are not typed here: 01_loss_overlay.py fetches them from wandb."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(ANALYSIS)
for path in (ROOT, ANALYSIS):  # curioslm, and analysis/common
    if path not in sys.path:
        sys.path.insert(0, path)

DATA, FIGURES = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
CONFIG_JSON = os.path.join(DATA, "run_configs.json")  # written by 01_loss_overlay.py

from common import run_ids  # noqa: E402

# Choices: the two headline runs (wandb run paths) in plotting order, with their labels.
MODEL_A, MODEL_B = run_ids.HEADLINE_A, run_ids.HEADLINE_B
LABELS = {MODEL_A: "124M GPT-2 (Model A)", MODEL_B: "113M Gemma-style (Model B)"}
REFERENCE = "gpt2"  # OpenAI's GPT-2 small, loaded through the eval adapter
NOISE_RUNS = (run_ids.LADDER_A4, run_ids.LADDER_A4_SEED2)  # one 1.5B-token Gemma-style run, two seeds
CONFIG_KEYS = ["out_dir", "dataset"]  # what 01 fetches from each run's wandb config
VAL_SETS = ("cosmopedia_v2",)  # held-out sets scored besides each run's own training dataset
BOOTSTRAP = 10_000  # resamples for every bootstrap interval in this study


def run_config(run_path):
    """A run's wandb config values (CONFIG_KEYS plus world_size), as cached by 01_loss_overlay.py."""
    if not os.path.exists(CONFIG_JSON):
        sys.exit(f"{CONFIG_JSON} is missing: run 01_loss_overlay.py first")
    with open(CONFIG_JSON) as f:
        return json.load(f)[run_path]
