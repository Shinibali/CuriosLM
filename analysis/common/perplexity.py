"""Per-token losses, grouped per document, and paired comparisons between models — shared by every perplexity
analysis (held-out sets, perplexity vs context length, loss vs position in the window)."""
import math
import os
import sys

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from curioslm.loaders.shard_loader import ShardedDataLoader  # noqa: E402

EOT = tiktoken.get_encoding("gpt2").eot_token  # the separator data/sharding.py writes before every document


def val_tokens(dataset, n_tokens):
    """The first n_tokens + 1 tokens of a dataset's validation split, located the way training's loader finds it."""
    loader = ShardedDataLoader(dataset, "val", block_size=1, batch_size=1, rank=0, world_size=1, device="cpu",
                               device_type="cpu")  # only its shard lookup is used
    return loader.tokens[:n_tokens + 1].astype(np.int64)  # shards are uint16 on disk; torch wants int64


@torch.no_grad()
def window_losses(model, tokens, device, batch_size=4, window=None):
    """Loss of every token after the first in each non-overlapping window of `window` inputs (default: the model's
    block_size). Returns n_windows * window losses in stream order: out[i] is the loss of tokens[i + 1], and
    out.reshape(-1, window) puts the position within the window on axis 1.

    Full-vocab logits are batch × window × vocab floats: keep batch_size small for long windows on a Mac."""
    window = window or model.config.block_size  # T
    tokens = np.asarray(tokens, dtype=np.int64)
    n_windows = (len(tokens) - 1) // window  # full windows only; -1 because the last token is only a target
    stream = torch.from_numpy(tokens[:n_windows * window + 1])  # +1: the final window's last target
    out = np.empty(n_windows * window, dtype=np.float32)
    for start in range(0, n_windows, batch_size):  # batch_size windows per forward pass
        idx = range(start, min(start + batch_size, n_windows))  # window numbers in this batch (last may be short)
        x = torch.stack([stream[w * window:(w + 1) * window] for w in idx]).to(device)  # (B,T) inputs
        y = torch.stack([stream[w * window + 1:(w + 1) * window + 1] for w in idx]).to(device)  # (B,T) next tokens
        logits, _ = model(x, y)  # targets given: logits at every position (B,T,V)
        losses = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), reduction="none")  # (B*T,)
        out[start * window:(start + len(idx)) * window] = losses.cpu().numpy()  # windows tile the stream in order
    return out


def doc_sums(tokens, losses, eot):
    """Per-document (summed loss, token count) for window_losses' output; documents are separated by `eot`."""
    tokens = np.asarray(tokens)
    # tokens [EOT, a, b, EOT, c, d, e] ↔ losses [ℓa, ℓb, ℓEOT, ℓc, ℓd, ℓe]
    # using eot as separator boolean makes cumsum the running count for the document ID of each scored token
    doc_of = np.cumsum(tokens[1:len(losses) + 1] == eot)  # scored token i+1 belongs to the doc opened by the last EOT
    n_docs = int(doc_of[-1]) + 1  # doc ids run 0..last
    # e.g. doc_of [0,0,1,1,1,1]: summed loss per doc [ℓa+ℓb, ℓEOT+ℓc+ℓd+ℓe], scored tokens per doc [2, 4]
    return (np.bincount(doc_of, weights=losses, minlength=n_docs),  # summed loss per doc
            np.bincount(doc_of, minlength=n_docs))  # scored tokens per doc


def perplexity(sums, counts):
    return math.exp(sums.sum() / counts.sum())  # e^(mean loss per token)


def paired_bootstrap(sums_a, sums_b, counts, n_resamples, seed=0):
    """95% interval for perplexity(B) − perplexity(A), resampling the same documents for both models. Both models
    must have scored the same tokens, so counts are shared."""
    rng = np.random.default_rng(seed)  # fixed seed: reproducible interval
    diffs = []
    for _ in range(n_resamples):
        # draw document numbers at random, with replacement, as many as we have documents
        i = rng.integers(0, len(counts), len(counts))
        diffs.append(perplexity(sums_b[i], counts[i]) - perplexity(sums_a[i], counts[i]))  # same docs: paired
    # sort the diffs and cut off the top and bottom 2.5%: the middle 95% is the confidence interval
    return np.percentile(diffs, [2.5, 97.5])
