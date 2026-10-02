"""_loglikelihood_tokens against a slow, obviously-correct reference: score each continuation token with its own
forward pass over the tokens before it (the chain rule, one term at a time)."""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("lm_eval")  # needs the eval extra: uv sync --extra eval

from curioslm.config import GPTConfig  # noqa: E402
from curioslm.eval_adapter import CurioslmLM  # noqa: E402
from curioslm.model import GPT  # noqa: E402

BLOCK, VOCAB = 16, 97


@pytest.fixture
def lm():
    torch.manual_seed(0)
    model = GPT(GPTConfig(n_embd=32, n_head=4, n_layer=2, block_size=BLOCK, vocab_size=VOCAB, dropout=0.0))
    return CurioslmLM(model=model, device="cpu", batch_size=2)


@torch.no_grad()
def reference(model, context, continuation):
    """(logprob, is_greedy) the slow way. Inputs longer than the model allows keep their last BLOCK + 1 tokens,
    the adapter's truncation contract."""
    assert 0 < len(continuation) <= BLOCK, "a continuation must fit in one window, or its first tokens see nothing"
    full = context + continuation
    start = max(0, len(full) - (BLOCK + 1))  # where the kept window begins; < len(context) since continuation <= BLOCK
    total, greedy = 0.0, True
    for p in range(len(context), len(full)):  # position of each continuation token
        seen = full[start:p]  # everything before it, inside the window
        logits, _ = model(torch.tensor([seen]))  # no targets: logits for the next token only, (1,1,V)
        logp = F.log_softmax(logits[0, -1], dim=-1)
        total += logp[full[p]].item()
        greedy &= int(logp.argmax()) == full[p]
    return total, greedy


def request(context, continuation):
    return ((None, None), context, continuation)


def check(lm, pairs):
    got = lm._loglikelihood_tokens([request(c, x) for c, x in pairs])
    assert len(got) == len(pairs)
    for (context, continuation), (logprob, greedy) in zip(pairs, got):
        want_logprob, want_greedy = reference(lm.model, context, continuation)
        assert logprob == pytest.approx(want_logprob, abs=1e-4)
        assert greedy == want_greedy


def test_single_token_continuation(lm):
    check(lm, [([5, 17, 3], [42])])


def test_multi_token_continuation_sums_each_token(lm):
    check(lm, [([5, 17, 3, 8], [42, 9, 77])])


def test_batch_of_different_lengths_keeps_order(lm):
    """batch_size=2 with 5 requests of different lengths: padding and any reordering must not leak into results."""
    check(lm, [([1, 2], [3]), ([4, 5, 6, 7, 8, 9], [10, 11]), ([12], [13, 14, 15]), ([16, 17, 18], [19]),
               ([20, 21, 22, 23], [24, 25, 26, 27])])


def test_too_long_input_keeps_the_last_block_plus_one_tokens(lm):
    check(lm, [(list(range(1, 30)), [31, 32, 33])])


def test_is_greedy_true_for_the_models_own_choices(lm):
    """Build a continuation from the model's argmax, so is_greedy must be True."""
    context, continuation = [5, 17, 3], []
    for _ in range(3):
        with torch.no_grad():
            logits, _ = lm.model(torch.tensor([context + continuation]))
        continuation.append(int(logits[0, -1].argmax()))
    (_, greedy), = lm._loglikelihood_tokens([request(context, continuation)])
    assert greedy is True
