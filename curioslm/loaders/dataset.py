"""
Dataset access: the data/<name>/ layout, tokenizer, and batch sampling. All datasets are
GPT-2 BPE-encoded (tiktoken).
"""

import os

import numpy as np
import tiktoken
import torch

from .. import runtime


EOT = tiktoken.get_encoding('gpt2').eot_token  # GPT-2's <|endoftext|>: data/sharding.py writes it before every document


def data_dir(name):
    return runtime.project_path('data', name)


def encoder_decoder():
    """(encode, decode) using GPT-2 BPE — the only tokenizer this project uses."""
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    return (lambda s: enc.encode(s, allowed_special={'<|endoftext|>'}),
            lambda l: enc.decode(l))


def is_sharded(name):
    """True for datasets stored as prepare.py shards; False for the small single-file ones (train.bin/val.bin)."""
    return not os.path.exists(os.path.join(data_dir(name), 'train.bin'))


def make_get_batch(name, block_size, batch_size, device, device_type):
    """Build a get_batch(split) sampler. Settings are arguments rather than module globals, so
    one process can hold several samplers (different block sizes or datasets) at once."""
    directory = data_dir(name)

    def get_batch(split):
        # Reopen the memmap every call rather than holding one open, to avoid a leak:
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        data = np.memmap(os.path.join(directory, f'{split}.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pinned memory can be copied to the GPU asynchronously
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    return get_batch