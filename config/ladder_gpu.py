# The GPT-2 -> Gemma architecture ladder at ~124M params, for GPU runs. Two env vars select a
# run -- the rung, and which dataset to run it on:
#
#   LADDER_RUNG=a2 uv run python train.py config/ladder_gpu.py --device=cuda
#   LADDER_RUNG=a2 LADDER_DATASET=cosmopedia_v2 uv run python train.py config/ladder_gpu.py --device=cuda
#
# A data-mixture study doesn't belong here: it varies a mixture ratio rather than an architecture
# rung, and the loader reads only one dataset at a time today.

import os

dataset = os.environ.get('LADDER_DATASET', 'fineweb_edu')
wandb_project = 'curioslm-ladder'

n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024

# Calibrated on 2x H100 80GB: micro-batch 60 is within 0.2% MFU of the largest that fits, with half the
# memory spare. grad accum is global (train.py divides it by world_size), so every step is
# 60 * 8 * 1024 = 491,520 tokens on any GPU count that divides 8, and 3052 steps is ~1.5B tokens.
batch_size = 60
gradient_accumulation_steps = 8
max_iters = 3052
lr_decay_iters = 3052
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 200
beta2 = 0.95
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = True
wandb_log = True

# 'a0' is empty because GPTConfig's own defaults already are the vanilla baseline; each later
# rung adds only its own delta.
RUNGS = {
    'a0': {},                                                      # vanilla GPT-2
    'a1': {'position_encoding': 'rope'},                           # + RoPE
    'a2': {'position_encoding': 'rope', 'n_kv_head': n_head // 4}, # + GQA, 4:1
    'a3': {'position_encoding': 'rope', 'n_kv_head': n_head // 4,  # + GeGLU, sandwich norm
           'mlp_type': 'geglu', 'norm_placement': 'sandwich'},
    'a4': {'position_encoding': 'rope', 'n_kv_head': n_head // 4,  # + local/global windowed attn
           'mlp_type': 'geglu', 'norm_placement': 'sandwich',
           'attention_window': block_size//2, 'local_global_ratio': 4},
}

rung = os.environ['LADDER_RUNG']  # KeyError if forgotten, rather than a silent default
# dataset is always in the name: the same rung on two datasets must never share an out_dir,
# or one silently overwrites the other's checkpoint
out_dir = f'out-ladder-{dataset}-{rung}'
wandb_run_name = f'ladder-{dataset}-{rung}'
globals().update(RUNGS[rung])  # e.g. KeyError: 'a9' if mistyped
