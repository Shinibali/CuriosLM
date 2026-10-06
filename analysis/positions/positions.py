"""Shared by this study's scripts: which models are examined, where their configs and checkpoints come from, paths.
Each model's out_dir is fetched from its wandb run once and cached; architecture comes from the checkpoint."""
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

# Choices: Model A has learned position embeddings; Model B has RoPE.
LEARNED, ROPE = run_ids.HEADLINE_A, run_ids.HEADLINE_B
# RoPE band speeds, by full turns within the training context, using YaRN's thresholds (Peng et al., 2023), declared
# once in model.py: fast = at least YARN_BETA turns (YaRN leaves these alone), slow = fewer than YARN_ALPHA (YaRN
# rescales these fully), middle = in between (YaRN blends these). Counted in turns, so the same numbers apply at any
# context length; which bands they select depends on the context, the RoPE base and the head size. alpha = 1 is
# principled (never completes a turn); beta = 32 is the paper's empirical choice for Llama models, a tunable
# hyperparameter rather than a law.
from curioslm.model import YARN_ALPHA, YARN_BETA  # noqa: E402
# Choice: stretch Model B's context 4×, 1,024 → 4,096. The YaRN paper's zero-shot results hold to about 2×, so
# at 2× a fine-tune has little to add; 8× is past the zero-shot range for a 124M model, and only ~1% of validation
# documents reach 8k tokens (~3% reach 4k). Zero-shot and fine-tuned YaRN are both measured at this factor.
YARN_FACTOR = 4


def speed_groups(wavelengths, context):
    """{"fast" | "middle" | "slow": band indices}, by how many full turns each band makes within `context` tokens."""
    turns = [context / wavelength for wavelength in wavelengths]
    return {"fast": [band for band, t in enumerate(turns) if t >= YARN_BETA],
            "middle": [band for band, t in enumerate(turns) if YARN_ALPHA <= t < YARN_BETA],
            "slow": [band for band, t in enumerate(turns) if t < YARN_ALPHA]}


def config(run_path):
    """A run's out_dir and training dataset, from its wandb config (fetched once, then cached in data/)."""
    cached = os.path.exists(CONFIG_JSON) and "dataset" in runs.load_configs([run_path], CONFIG_JSON)[run_path]
    if not cached:
        os.makedirs(DATA, exist_ok=True)
        runs.fetch_configs([LEARNED, ROPE], ["out_dir", "dataset"], CONFIG_JSON)
    return runs.load_configs([run_path], CONFIG_JSON)[run_path]


def out_dir(run_path):
    return config(run_path)["out_dir"]


def model_args(out_dir):
    """The architecture saved in a run's checkpoint (downloaded from the staging repo if it isn't local)."""
    from curioslm import checkpoints
    from curioslm.config import IOConfig
    path = os.path.join(ROOT, out_dir, "ckpt.pt")
    if not os.path.exists(path):
        cwd = os.getcwd()
        os.chdir(ROOT)  # download_checkpoint writes <out_dir>/ckpt.pt relative to the working directory
        try:
            checkpoints.download_checkpoint(out_dir, IOConfig().hf_checkpoint_repo_id)
        finally:
            os.chdir(cwd)
    return checkpoints.load(path, map_location="cpu")["model_args"]


def rope_frequencies(args, yarn_factor=1.0):
    """Radians turned per position by each RoPE band, read from the model's own rotation table (the angle at
    position 1), so they match what Model B actually used; with yarn_factor, the table YaRN builds from the training
    length. Returns (frequencies, head size)."""
    import torch
    from curioslm.model import build_rope_cache
    head_size = args["n_embd"] // args["n_head"]
    cos, sin = build_rope_cache(head_size, 2, args["rope_theta"], yarn_factor,
                                args.get("block_size"))  # rows: positions 0 and 1; temperature cancels in atan2
    return torch.atan2(sin[1], cos[1]), head_size  # all below π, so atan2 recovers the angle exactly
