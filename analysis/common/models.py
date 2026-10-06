"""Load a CuriosLM checkpoint for analysis and score text with it."""
import os
import re
import sys

import torch
from circuitsvis.tokens import colored_tokens

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from curioslm import checkpoints  # noqa: E402
from curioslm.config import IOConfig  # noqa: E402


def device():
    """The fastest local device: CUDA on a GPU pod, MPS on a Mac, else CPU."""
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(out_dir, device):
    """A run's checkpoint (<out_dir>/ckpt.pt under the repo root) in eval mode, dropout off; pulled from the
    checkpoint staging repo first if it isn't on local disk."""
    ckpt_path = os.path.join(ROOT, out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        cwd = os.getcwd()
        os.chdir(ROOT)  # download_checkpoint writes <out_dir>/ckpt.pt relative to the working directory
        try:
            checkpoints.download_checkpoint(out_dir, IOConfig().hf_checkpoint_repo_id)
        finally:
            os.chdir(cwd)
    model, _ = checkpoints.build_model(checkpoints.load(ckpt_path, map_location="cpu"), overrides={"dropout": 0.0})
    return model.eval().to(device)


def token_losses(model, ids, device):
    """Loss of ids[1:], each given every token before it: one window over the whole sequence."""
    from common.perplexity import window_losses  # one per-token-loss implementation for the project
    return window_losses(model, ids, device, batch_size=1, window=len(ids) - 1).tolist()


def continuation_logprobs(model, context_ids, continuations, device, batch_size=16):
    """Summed log p(continuation | context) for each continuation (token ids), through the eval adapter's
    _loglikelihood_tokens: the project's one log-likelihood implementation. Needs the eval extra."""
    from curioslm.eval_adapter import CurioslmLM  # lazy: only scripts that score continuations need lm-eval
    lm = CurioslmLM(model=model, device=device, batch_size=batch_size)
    requests = [((None, None), context_ids, ids) for ids in continuations]
    # Adding logs multiplies probabilities: log[p(1 | A B C) × p(2 | A B C 1) …] = log p(continuation | context)
    return [logprob for logprob, _ in lm._loglikelihood_tokens(requests)]


def token_texts(enc, ids):
    """Each token's text. A character split across byte-level tokens goes wholly to the first of them."""
    text, offsets = enc.decode_with_offsets(ids)  # offsets[i]: character where token i starts
    return [text[a:b] for a, b in zip(offsets, offsets[1:] + [len(text)])]  # slice between consecutive starts


def colored_tokens_html(tokens, values, **kwargs):
    """CircuitsVis per-token view as an embeddable HTML snippet. CircuitsVis emits a self-closing <div/>, which
    HTML leaves open, so a second view on the same page would nest inside it and be wiped: close it explicitly."""
    view = str(colored_tokens(tokens=tokens, values=values, **kwargs))
    return re.sub(r'(<div id="circuits-vis-[^"]+"[^>]*?)\s*/>', r"\1></div>", view, count=1)


def kv_bytes_formula(config, seq_len, batch, bytes_per_value):
    """Bytes the KV cache holds after seq_len positions: per layer 2 (K and V) × batch × kv heads × head size ×
    stored positions; local layers store at most attention_window − 1 positions."""
    head_size = config.n_embd // config.n_head
    stored = sum(seq_len if config.is_global_layer(layer) else min(seq_len, config.attention_window - 1)
                 for layer in range(config.n_layer))  # positions held, summed over layers
    return stored * 2 * batch * config.num_kv_heads * head_size * bytes_per_value


def forward_flops_formula(config):
    """Forward FLOPs per token by component, from the formula in the blog post, from the config's
    hyperparameters only (biases left out). Per layer: QKV projection 2d(d + 2·n_kv·h) and output projection 2d²
    ("attention projections"); MLP: GELU 2·(d·4d + 4d·d) = 16d², GeGLU 2·(d·2f + f·d) = 6df with f = ⌊8d/3⌋;
    "attention scores" 4d·k̄, with k̄ = keys a query sees averaged over the window: (T + 1)/2 global,
    (W(W + 1)/2 + (T − W)·W)/T local. Plus the "output layer" 2dV once."""
    d, T, V, L = config.n_embd, config.block_size, config.vocab_size, config.n_layer
    h, n_kv = d // config.n_head, config.num_kv_heads
    mlp = 16 * d * d if config.use_gelu_activation else 6 * d * int(2 / 3 * 4 * d)  # GeGLU width as in model.py
    scores = 0.0
    for layer in range(L):
        if config.is_global_layer(layer):
            k_bar = (T + 1) / 2
        else:
            W = config.attention_window
            k_bar = (W * (W + 1) / 2 + (T - W) * W) / T  # ramp up to W keys, then W keys for the rest
        scores += 4 * d * k_bar
    return {"attention projections": L * (2 * d * (d + 2 * n_kv * h) + 2 * d * d), "attention scores": scores,
            "MLP": L * mlp, "output layer": 2 * d * V}
