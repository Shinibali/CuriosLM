"""
Sample from a trained model
"""
import os
import torch

from curioslm import checkpoints, runtime
from curioslm import config as cfg
from curioslm import dataset as data
from curioslm.model import GPT

# -----------------------------------------------------------------------------
# Annotations only — no values. globals().update() below is invisible to type
# checkers, so these tell the IDE which settings this script accepts; the actual
# defaults live in config.py.
start: str; num_samples: int; max_new_tokens: int; temperature: float; top_k: int
device: str; dtype: str; compile: bool; seed: int; peak_flops_override: float

# defaults come from config.py; configurator.py overrides them from the CLI
globals().update(cfg.flatten(
    cfg.SampleConfig(),  # start, num_samples, max_new_tokens, temperature, top_k
    cfg.RuntimeConfig(), # device, precision, compile, seed
))
init_from = 'resume' # sampling defaults to a checkpoint, where IOConfig defaults to 'scratch'
out_dir = cfg.IOConfig().out_dir
compile = False # override the training default: compiling costs ~a minute, sampling is short

exec(open(runtime.package_path('configurator.py')).read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
runtime.enable_tf32()
if not device:
    device = runtime.pick_device()
device_type = runtime.device_type_of(device)
if not dtype:
    dtype = runtime.pick_dtype(device_type)
ctx = runtime.autocast_ctx(device_type, dtype)
print(runtime.describe(device, dtype, compile))

# model
checkpoint = None
if init_from == 'resume':
    # init from a model saved in a specific directory
    checkpoint = checkpoints.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    # dropout off for sampling regardless of what the model was trained with
    model, _ = checkpoints.build_model(checkpoint, overrides={'dropout': 0.0})
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

encode, decode = data.encoder_decoder()

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
