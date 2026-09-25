import os
import time
import math

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier

from curioslm import checkpoints, runtime
from curioslm import config as cfg
from curioslm.loaders import dataset as data
from curioslm.config import GPTConfig
from curioslm.model import GPT
from curioslm.loaders.shard_loader import ShardedDataLoader
from curioslm.loaders.hf_shards import download_shards

#-------------------------------------------------------------------
# config: defaults live in config.py, overridable from the CLI via configurator.py:
#   python train.py config/train_shakespeare.py --batch_size=32 --device=mps
#-------------------------------------------------------------------

# annotations only, no values — globals().update() is invisible to type checkers
block_size: int; vocab_size: int; n_layer: int; n_head: int; n_embd: int
n_kv_head: int | None; attention_window: int | None
dropout: float; bias: bool; dataset: str; hf_dataset_repo_id: str
out_dir: str; init_from: str; eval_interval: int; log_interval: int; eval_iters: int
eval_only: bool; always_save_checkpoint: bool; hf_checkpoint_repo_id: str
wandb_log: bool; wandb_project: str; wandb_run_name: str
gradient_accumulation_steps: int; batch_size: int; learning_rate: float; max_iters: int
weight_decay: float; beta1: float; beta2: float; grad_clip: float
decay_lr: bool; warmup_iters: int; lr_decay_iters: int; min_lr: float; backend: str
device: str; dtype: str; compile: bool; seed: int; peak_flops_override: float

globals().update(cfg.flatten(
    cfg.GPTConfig(),    # architecture
    cfg.DataConfig(),   # dataset
    cfg.IOConfig(),     # out_dir, eval cadence, wandb
    cfg.OptimConfig(),  # optimizer, LR schedule, batching, ddp backend
    cfg.RuntimeConfig() # device, precision, compile, seed
))

#-------------------------------------------------------------------
# learning rate decay scheduler (cosine with warmup)
#-------------------------------------------------------------------

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return (1-coeff) * min_lr + coeff * learning_rate

#-------------------------------------------------------------------
# config override
#-------------------------------------------------------------------

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(runtime.package_path('configurator.py')).read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# resolve the '' / -1 sentinels now that overrides have landed
if not device:
    device = runtime.pick_device()
if not dtype:
    dtype = runtime.pick_dtype(runtime.device_type_of(device))
if lr_decay_iters < 0:
    lr_decay_iters = max_iters # keeps --max_iters= alone from silently desyncing the schedule
config.update(device=device, dtype=dtype, lr_decay_iters=lr_decay_iters)

assert warmup_iters < max_iters, f"warmup_iters={warmup_iters} >= max_iters={max_iters}: lr would never decay"

# A config file's derived architecture values (n_kv_head = n_head // 4, attention_window =
# block_size // 2) are frozen when that file execs, before any --key= override lands. Catch the
# combinations that would otherwise be silently wrong rather than loud.
assert n_kv_head is None or n_head % n_kv_head == 0, \
    f"n_head={n_head} not divisible by n_kv_head={n_kv_head}"
assert attention_window is None or attention_window <= block_size, \
    (f"attention_window={attention_window} > block_size={block_size}: every position falls inside "
     "the window, so local layers silently become full causal attention")

#-------------------------------------------------------------------
# various inits, derived attributes, I/O setup
#-------------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # ddp_world_size num of processes will train simultaneously so we can scale down grad accum steps accordingly
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    if init_from == 'scratch':
        checkpoints.check_no_overwrite(os.path.join(out_dir, 'ckpt.pt'), wandb_run_name)

torch.manual_seed(seed + seed_offset)
runtime.enable_tf32()

device_type = runtime.device_type_of(device)
ctx = runtime.autocast_ctx(device_type, dtype)
flops_promised = runtime.peak_flops(device_type, peak_flops_override)
if master_process:
    print(runtime.describe(device, dtype, compile))
    if flops_promised is None:
        print("MFU unavailable: no trusted peak-FLOPS figure for this device (set --peak_flops_override=)")

#-------------------------------------------------------------------
# data loader
#-------------------------------------------------------------------

get_batch = data.make_get_batch(dataset, block_size, batch_size, device, device_type)
if dataset == 'fineweb_edu':
    # ddp_local_rank == 0, not master_process: every node needs its own local copy, and
    # master_process is true on only one process cluster-wide
    if ddp_local_rank == 0:
        download_shards(data.data_dir(dataset), hf_dataset_repo_id)
    # other local ranks wait for rank 0's download instead of racing it
    if ddp:
        barrier()

    train_loader = ShardedDataLoader( dataset, "train", block_size, batch_size, ddp_rank, ddp_world_size, device, device_type)
    val_loader = ShardedDataLoader( dataset, "val", block_size, batch_size, ddp_rank, ddp_world_size, device, device_type)

    get_batch = lambda x: train_loader.next_batch() if x == "train" else val_loader.next_batch()

#-------------------------------------------------------------------
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
#-------------------------------------------------------------------

iter_num = 0
best_val_loss = 1e9

#-------------------------------------------------------------------
# model init
#-------------------------------------------------------------------

# built from checkpoints.CONFIG_KEYS so a new GPTConfig knob never needs repeating here
model_args = {k: globals()[k] for k in checkpoints.CONFIG_KEYS if k in globals()}

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        # not on this node's disk (fresh pod, or lost local state) -- pull it rather than fail
        if ddp_local_rank == 0:
            checkpoints.download_checkpoint(out_dir, hf_checkpoint_repo_id)
        if ddp:
            barrier()
    print(f"Resuming training from {out_dir}")
    checkpoint = checkpoints.load(ckpt_path, map_location=device)
    # architecture is fixed by the checkpoint; dropout stays as asked for on the command line
    model, model_args = checkpoints.build_model(checkpoint, overrides={'dropout': dropout})
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in checkpoints.ARCH_KEYS:
        model_args[k] = getattr(model.config, k)


#-------------------------------------------------------------------
# model compile
#-------------------------------------------------------------------

model.to(device)

scaler = runtime.grad_scaler(device_type, dtype)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

#-------------------------------------------------------------------
# training loop
#-------------------------------------------------------------------

X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0


while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val'].item()
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                checkpoints.save(
                    os.path.join(out_dir, 'ckpt.pt'),
                    model=raw_model, optimizer=optimizer, model_args=model_args,
                    iter_num=iter_num, best_val_loss=best_val_loss, config=config,
                )
                # get it off this pod's ephemeral disk immediately; async so a multi-GB
                # upload never blocks the training loop
                checkpoints.upload_checkpoint_async(out_dir, hf_checkpoint_repo_id)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # only sync gradients on the last micro step; no_sync() just toggles this flag
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # CPU-GPU sync point; undo the grad-accum division to approximate the total loss
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5 and flops_promised: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt, flops_promised)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        mfu_str = f"{running_mfu*100:.2f}%" if running_mfu >= 0 else "n/a"
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {mfu_str}")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()