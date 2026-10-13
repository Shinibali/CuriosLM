"""Shared by this study's scripts: the models compared, the probe text, paths, and how tokens are labelled.
Each model's out_dir is fetched from its wandb run once and cached; architecture comes from the checkpoint. Attention
probabilities come from the model's own explicit path (GPT.attention_weights); the figures are static
(matplotlib)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(ANALYSIS)
for path in (ROOT, ANALYSIS):  # curioslm, and analysis/common
    if path not in sys.path:
        sys.path.insert(0, path)
from common import models, run_ids, runs  # noqa: E402

DATA, FIGURES = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
CONFIG_JSON = os.path.join(DATA, "run_configs.json")

# Choices: the two headline models, in plotting order.
MODELS = {"Model A": run_ids.HEADLINE_A, "Model B": run_ids.HEADLINE_B}
# The emoji probe: GPT-2's byte-level BPE splits each of these two emoji into a token they share (a space plus the
# first two UTF-8 bytes) and two tokens of its own, so a model can only tell them apart from the later tokens.
PROBE = "Learning is Fun 🤗, Keep exploring 🔭"


def config(run_path):
    """A run's out_dir and training dataset, from its wandb config (fetched once, then cached in data/)."""
    if not os.path.exists(CONFIG_JSON):  # one fetch covers every model in MODELS
        os.makedirs(DATA, exist_ok=True)
        runs.fetch_configs(list(MODELS.values()), ["out_dir", "dataset"], CONFIG_JSON)
    return runs.load_configs([run_path], CONFIG_JSON)[run_path]


def load(name, device):
    """One of MODELS, in eval mode (downloaded from the staging repo if its checkpoint isn't local)."""
    return models.load_model(config(MODELS[name])["out_dir"], device)


def draw_token_attention(ax, matrix, labels, show_rows=True, show_columns=True):
    """One static token-to-token attention heatmap on `ax`: rows are the token looking, columns the tokens it looks
    at; cells after the diagonal, which causal attention never fills, in grey. Returns the image, for a colorbar."""
    import matplotlib.pyplot as plt
    from common import style
    future = np.triu(np.ones(matrix.shape, dtype=bool), k=1)  # key after query
    colormap = plt.get_cmap(style.SEQUENTIAL_CMAP).copy()
    colormap.set_bad(style.GRID)  # masked (future) cells
    image = ax.imshow(np.ma.masked_where(future, matrix), cmap=colormap, vmin=0, vmax=1)
    positions = range(len(labels))
    ax.set_xticks(positions, labels if show_columns else [], rotation=90, fontsize=6)
    ax.set_yticks(positions, labels if show_rows else [], fontsize=6)
    ax.tick_params(length=0)
    ax.grid(False)
    return image


def token_labels(enc, ids):
    """A readable label per token: its text with line breaks shown as ↵ (a bare newline would be an invisible
    label), or its raw bytes when it holds only part of a character (an emoji's later byte-level tokens decode to
    nothing on their own)."""
    labels = []
    for token in ids:
        piece = enc.decode_single_token_bytes(token)
        try:
            labels.append(piece.decode("utf-8").replace("\n", "↵"))
        except UnicodeDecodeError:  # part of a multi-byte character: show the bytes, keeping any leading space
            text = piece.lstrip(b" ")
            labels.append(" " * (len(piece) - len(text)) + "".join(f"\\x{byte:02x}" for byte in text))
    return labels