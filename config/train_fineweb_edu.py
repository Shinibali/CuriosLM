# FineWeb-Edu with the full modernized recipe (RoPE + GQA + GeGLU/sandwich norm + local/global
# attention), at dev/smoke-test scale for iterating on a Mac.

out_dir = 'out-fineweb_edu'
eval_interval = 20
eval_iters = 5
log_interval = 10

wandb_log = False # override via command line if you like
wandb_project = 'curioslm'
wandb_run_name = 'fineweb_edu-model-b'

dataset = 'fineweb_edu'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous BPE tokens

# the fully stacked ladder: every architecture change at once
position_encoding = 'rope'
mlp_type = 'geglu'
norm_placement = 'sandwich'
local_global_ratio = 4

# baby GPT model :) -- shakespeare smoke-test scale, not the real 124M ladder size
n_layer = 8
attention_window = 4
n_head = 4
n_kv_head = n_head // 4 # 4:1 Q:KV; set here because GPTConfig defaults it to None, which a CLI override can't retype
n_embd = n_head * 32 # 128
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 60
lr_decay_iters = 60 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 10 # not super necessary potentially
