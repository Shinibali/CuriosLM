"""Load a CuriosLM checkpoint for analysis and score text with it."""
import os
import re
import sys

import torch
import torch.nn.functional as F
from circuitsvis.tokens import colored_tokens

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from curioslm import checkpoints  # noqa: E402


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(ckpt_path, device):
    """The checkpoint's model in eval mode, dropout off."""
    model, _ = checkpoints.build_model(checkpoints.load(ckpt_path, map_location="cpu"), overrides={"dropout": 0.0})
    return model.eval().to(device)


@torch.no_grad()
def token_losses(model, ids, device):
    """Loss of ids[1:], each given every token before it."""
    x = torch.tensor(ids[:-1], device=device)[None]  # (1,T) inputs
    y = torch.tensor(ids[1:], device=device)[None]  # (1,T) targets: inputs shifted by one
    logits, _ = model(x, y)  # passing targets returns logits at every position (1,T,V)
    return F.cross_entropy(logits[0].float(), y[0], reduction="none").cpu().tolist()  # (T,): −log p per token


def token_texts(enc, ids):
    """Each token's text. A character split across byte-level tokens goes wholly to the first of them."""
    text, offsets = enc.decode_with_offsets(ids)  # offsets[i]: character where token i starts
    return [text[a:b] for a, b in zip(offsets, offsets[1:] + [len(text)])]  # slice between consecutive starts


def colored_tokens_html(tokens, values, **kwargs):
    """CircuitsVis per-token view as an embeddable HTML snippet. CircuitsVis emits a self-closing <div/>, which
    HTML leaves open, so a second view on the same page would nest inside it and be wiped: close it explicitly."""
    view = str(colored_tokens(tokens=tokens, values=values, **kwargs))
    return re.sub(r'(<div id="circuits-vis-[^"]+"[^>]*?)\s*/>', r"\1></div>", view, count=1)
