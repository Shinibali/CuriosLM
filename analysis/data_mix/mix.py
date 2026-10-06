"""Shared by this study's scripts: the runs compared and the choices they share, declared once. Facts about the runs
(out_dir, training dataset, the data mix) are fetched from wandb once and cached in data/."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(ANALYSIS)
for path in (ROOT, ANALYSIS):  # curioslm, and analysis/common
    if path not in sys.path:
        sys.path.insert(0, path)
from common import run_ids, runs  # noqa: E402

DATA, FIGURES = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
CONFIG_JSON = os.path.join(DATA, "run_configs.json")

# Choices: same architecture (Model B), same token budget, different data. Model A and GPT-2 as references.
BASE, MIX, MODEL_A = run_ids.HEADLINE_B, run_ids.HEADLINE_MIX, run_ids.HEADLINE_A
LABELS = {BASE: "Model B (100% FineWeb-Edu)", MIX: "Model B, 50:50 mix", MODEL_A: "Model A (100% FineWeb-Edu)"}
REFERENCE = "gpt2"  # OpenAI's GPT-2 small, loaded through the eval adapter
BOOTSTRAP = 10_000  # resamples for every bootstrap interval in this study
DISPLAY = {"fineweb_edu": "FineWeb-Edu", "cosmopedia_v2": "Cosmopedia-v2",  # dataset and task names as figures print them
           "hellaswag": "HellaSwag", "arc_easy": "ARC-Easy", "lambada_openai": "LAMBADA"}


def display(dataset):
    return DISPLAY.get(dataset, dataset)


def config(run_path):
    """A run's out_dir, dataset and data_mix from its wandb config (fetched once, then cached)."""
    if not os.path.exists(CONFIG_JSON):
        os.makedirs(DATA, exist_ok=True)
        runs.fetch_configs(list(LABELS), ["out_dir", "dataset", "data_mix"], CONFIG_JSON)
    with open(CONFIG_JSON) as f:
        return json.load(f)[run_path]


def val_sets():
    """The datasets in the mix, in mix order; the first is the one train.py logs as val/loss."""
    return list(config(MIX)["data_mix"])


def val_key(dataset):
    """The wandb key train.py logs a validation set's loss under."""
    return "val/loss" if dataset == val_sets()[0] else f"val_{dataset}/loss"
