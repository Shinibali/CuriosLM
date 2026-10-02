"""CuriosLM as an lm-evaluation-harness model: benchmarks score our checkpoints with lm-eval's own task definitions
and scoring conventions, so the numbers are comparable to published ones.

    uv run --extra eval python evaluate.py out-ladder-fineweb_edu-a4 --tasks hellaswag --limit 50
"""
import os

import tiktoken
import torch
import torch.nn.functional as F

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from . import checkpoints
from .config import IOConfig
from .model import GPT


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(out_dir):
    """A checkpoint's model, dropout off. `out_dir` is a run directory (its ckpt.pt is pulled from the staging repo
    if missing locally) or "gpt2", OpenAI's GPT-2 small, as a reference."""
    if out_dir == "gpt2":
        return GPT.from_pretrained("gpt2", dict(dropout=0.0))
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        checkpoints.download_checkpoint(out_dir, IOConfig().hf_checkpoint_repo_id)
    model, _ = checkpoints.build_model(checkpoints.load(ckpt_path, map_location="cpu"), overrides={"dropout": 0.0})
    return model


@register_model("curioslm")
class CurioslmLM(TemplateLM):
    def __init__(self, out_dir=None, model=None, device=None, batch_size=16):
        """Pass `out_dir` (as lm-eval's model_args do) or an already-built `model` (as the tests do)."""
        super().__init__()
        assert (out_dir is None) != (model is None), "give exactly one of out_dir or model"
        self._device = torch.device(device or default_device())
        self.model = (model if model is not None else load_model(out_dir)).eval().to(self._device)  # float32
        self.enc = tiktoken.get_encoding("gpt2")
        self.batch_size = int(batch_size)
        self.max_length = self.model.config.block_size  # the longest input one forward pass takes

    @property
    def eot_token_id(self):
        return self.enc.eot_token  # also lm-eval's prefix for an empty context, as in training's document separator

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        return self.enc.encode_ordinary(string)  # a literal "<|endoftext|>" in text stays plain text

    def tok_decode(self, tokens, **kwargs):
        return self.enc.decode(tokens)

    @torch.no_grad() # scoring only
    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        """Score continuations given contexts. The one method the benchmarks run through.

        requests: list of ((context_str, continuation_str), context_ids, continuation_ids); the strings are for
            caching and may be None, the token ids are what to score. context_ids is never empty.
        Returns: one (logprob, is_greedy) per request, **in request order**:
            logprob   = sum over continuation tokens of log p(token | every token before it), a float
            is_greedy = True iff every continuation token was the model's single most likely next token
        """
        results = []

        for start in range(0, len(requests), self.batch_size): # chunks of batch size
                           
            batch = requests[start: start + self.batch_size]
            inputs, conts = [], []

            for _, context_ids, continuation_ids in batch:
                assert 0 < len(continuation_ids) <= self.max_length,  "continuation must fit in one window"
                full = (context_ids + continuation_ids)[-(self.max_length+1):]  # truncate: keep the last max_length + 1

                inputs.append(full[:-1]) # the last token is a target, not input
                conts.append(continuation_ids)

            width = max(len(inp) for inp in inputs)
            x = torch.zeros(len(batch), width, dtype=torch.long) # (B,T) right padded zeros
            for i, inp in enumerate(inputs):
                x[i, :len(inp)] = torch.tensor(inp)
            x = x.to(self._device)

            logits, _ = self.model(x, torch.zeros_like(x)) # dummy targets: logits at every position (B,T,V)
            logp = F.log_softmax(logits.float(), dim=-1)  # (B,T,V): log p of every next token, at every row

            for i, (inp, cont) in enumerate(zip(inputs, conts)):
                n = len(cont)
                pos = torch.arange(len(inp) - n, len(inp), device=self._device)  # rows predicting the n continuation tokens

                targets = torch.tensor(cont, device=self._device)  # (n,) the tokens that actually came
                token_logp = logp[i, pos, targets] # (n,) pairs row pos[j] with token targets[j]

                greedy = bool((logp[i, pos].argmax(dim=-1) == targets).all()) # was every token the top choice?
                
                # Adding logs is multiplying probabilities: adds those n values: log p(continuation i | context i)
                results.append((token_logp.sum().item(), greedy)) 

        return results # total of (log_prob, greedy) for all samples

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        """Whole-document log-likelihood, for perplexity tasks like WikiText: the document is cut into
        non-overlapping windows of at most max_length tokens, each scored by _loglikelihood_tokens."""
        out = []
        for (string,) in [req.args for req in requests]:
            windows = [make_disjoint_window(w) for w in get_rolling_token_windows(
                token_list=self.tok_encode(string), prefix_token=self.prefix_token_id,
                max_seq_len=self.max_length, context_len=1)]  # each window: (context ids, ids to score)
            scored = self._loglikelihood_tokens([((None, None), ctx, cont) for ctx, cont in windows],
                                                disable_tqdm=disable_tqdm)
            out.append(sum(logprob for logprob, _ in scored))  # windows tile the document, so log-probs add
        return out

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("generation tasks aren't part of this evaluation")
