"""Undo the "baby talk" rewrite of the step-320 page, so the model can be scored on a clean twin of it.

The page was mangled by a few fixed rules: l -> w, th -> d, word-final th -> f (deaf, 20f) and qu -> qw. Edit
distance to a dictionary can't invert that: `kiwwed` is two edits from both `killed` and `kissed`, and `dose` is
already a word, so it would stay. Instead each word expands into every way of undoing the rules (each w may be a
real w or an l, each d a real d or a th, ...), and the language model picks the candidate it finds likeliest after
the already-corrected text before it. That is how `Of dose` becomes `Of those` while `dose` stays available.

Running it writes the clean page to data/page_clean.txt and every change, with the runner-up and the log-prob
margin, to data/unmangle_changes.csv, so each decision can be checked by eye.

    uv run --extra analysis --extra eval python analysis/step320_spike/02_unmangle.py
"""
import csv
import itertools
import json
import os

import numpy as np
import tiktoken

import page  # also puts analysis/ on sys.path
from common import models  # noqa: E402

CONTEXT = 256  # tokens of corrected text the model sees before each word


def candidates(word):
    """Every spelling the rules could have mangled into `word`, including `word` itself."""
    word = word.replace("qw", "qu").replace("Qw", "Qu")  # "qw" never occurs in English: undo unconditionally
    options = []
    for i, ch in enumerate(word):
        if ch in "wW":
            options.append((ch, "l" if ch == "w" else "L"))  # keep the w, or undo l -> w
        elif ch in "dD":
            options.append((ch, "th" if ch == "d" else "Th"))  # keep the d, or undo th -> d
        elif ch == "f" and i == len(word) - 1 and i > 0:  # word-final f (deaf, 20f)
            options.append((ch, "th"))  # keep the f, or undo th -> f
        else:
            options.append((ch,))  # no rule touches this character
    return sorted({"".join(p) for p in itertools.product(*options)})  # every combination: 2^(rule sites) spellings


def unmangle(model, enc, text, device):
    """Returns (clean text, list of changes). Words are fixed left to right, each seeing the fixed text before it."""
    out, changes, prefix_ids, last = [], [], [enc.eot_token], 0  # prefix starts at a document boundary
    for m in page.WORD.finditer(text):
        gap, word = text[last:m.start()], m.group()  # gap: spaces/punctuation since the previous word
        cands = candidates(word)
        if len(cands) == 1:  # no w, d or final f: nothing to decide
            best = cands[0]
        else:
            continuations = [enc.encode_ordinary(gap + c) for c in cands]  # with its gap: " those"
            logps = models.continuation_logprobs(model, prefix_ids[-CONTEXT:], continuations, device)
            ranked = sorted(zip(logps, cands), reverse=True)  # most likely spelling first
            best = ranked[0][1]
            if best != word:
                # margin = log p(best) − log p(runner-up) = ln [ p(best) / p(runner-up) ]
                changes.append({"position": m.start(), "mangled": word, "clean": best,
                                "runner_up": ranked[1][1], "margin": round(ranked[0][0] - ranked[1][0], 2)})  # nats
        out.append(gap + best)
        prefix_ids = prefix_ids + enc.encode_ordinary(gap + best)  # later words see the corrected text
        last = m.end()
    out.append(text[last:])  # trailing text after the last word
    return "".join(out), changes


def locate_page(model, device):
    """Replay the loader to the spike's logged batch and pick its worst-scoring sequence: that's the page.
    Recorded in page_location.json for page.load_sequence()."""
    shard, offset, batch = page.logged_batch()
    mean_losses = [sum(models.token_losses(model, seq.tolist(), device)) / (len(seq) - 1) for seq in batch]  # each row
    worst = max(range(len(mean_losses)), key=mean_losses.__getitem__)  # argmax: the highest-loss sequence
    loc = {"shard": os.path.relpath(shard, page.ROOT),  # relative to the repo: portable, no local paths
           "offset": offset, "sequence": worst, "block_size": batch.shape[1] - 1,  # batch rows hold block_size + 1
           "mean_loss": round(mean_losses[worst], 4), "batch_median_loss": round(float(np.median(mean_losses)), 4)}
    with open(page.LOCATION_JSON, "w") as f:
        json.dump(loc, f, indent=2)
    print(f"page: {os.path.basename(shard)}, offset {offset:,}, sequence {worst} of {len(batch)} "
          f"(mean loss {loc['mean_loss']} vs batch median {loc['batch_median_loss']})")


def main():
    device = models.device()
    enc = tiktoken.get_encoding("gpt2")
    model = models.load_model(page.run_config()["out_dir"], device)
    os.makedirs(page.DATA, exist_ok=True)
    locate_page(model, device)
    text = enc.decode(page.load_sequence())
    clean, changes = unmangle(model, enc, text, device)
    with open(page.CLEAN_TXT, "w") as f:
        f.write(clean)
    with open(page.CHANGES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["position", "mangled", "clean", "runner_up", "margin"])
        w.writeheader()
        w.writerows(changes)
    print(f"{len(changes)} words changed; wrote {page.CLEAN_TXT} and {page.CHANGES_CSV}")


if __name__ == "__main__":
    main()
