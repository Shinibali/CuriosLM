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
    return "mps" if torch.backends.mps.is_available() else "cpu"


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
