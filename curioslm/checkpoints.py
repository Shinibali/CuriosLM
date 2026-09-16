"""
Checkpoint format: saving, loading, and rebuilding a model from one.
"""

import os
from dataclasses import fields

import torch

from .config import GPTConfig
from .model import GPT

# Derived from the dataclass rather than hardcoded, so adding a knob to GPTConfig
# (n_kv_head for GQA, say) propagates everywhere automatically. This list used to
# be spelled out by hand in three separate places in train.py.
CONFIG_KEYS = tuple(f.name for f in fields(GPTConfig))

# Architecture must match the checkpoint exactly or the weights won't load.
# dropout is excluded: it's a training-time knob you may legitimately change when
# resuming, and it owns no of parameters.
ARCH_KEYS = tuple(k for k in CONFIG_KEYS if k != 'dropout')

_COMPILE_PREFIX = '_orig_mod.'


def strip_compile_prefix(state_dict):
    """torch.compile wraps the model, so checkpoints saved from a compiled model
    carry an '_orig_mod.' prefix on every key. Strip it so the weights load into
    an uncompiled model."""
    for k in list(state_dict):
        if k.startswith(_COMPILE_PREFIX):
            state_dict[k[len(_COMPILE_PREFIX):]] = state_dict.pop(k)
    return state_dict


def save(path, model, optimizer, model_args, iter_num, best_val_loss, config):
    """Write the checkpoint atomically.

    torch.save straight to the destination means a crash mid-write leaves a
    truncated file where the only good checkpoint used to be. Writing to a temp
    file and renaming avoids that: rename is atomic, so the destination is either
    the old checkpoint or the complete new one, never a partial.
    """
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }, f'{path}.tmp')
    os.replace(f'{path}.tmp', path)


def load(path, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


def build_model(checkpoint, overrides=None):
    """Rebuild the model a checkpoint describes and load its weights.

    Architecture comes from the checkpoint; `overrides` lets the caller change
    non-architectural settings (in practice: dropout). Returns (model, model_args)
    so the caller can carry model_args into checkpoints it writes later.
    """
    saved = checkpoint['model_args']
    model_args = {k: saved[k] for k in CONFIG_KEYS if k in saved}
    for k, v in (overrides or {}).items():
        if k in CONFIG_KEYS and k not in ARCH_KEYS:
            model_args[k] = v

    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(strip_compile_prefix(checkpoint['model']))
    return model, model_args
