"""Figure: where the step-320 page's extra loss lands, word by word.

One 1,024-token sequence in the step-320 batch is a page rewritten in "baby talk". This takes its opening
sentences and their un-mangled twin (from 02_unmangle.py), scores both from a document boundary, and sums each
word's token losses. The twin changes only the mangled words, so comparing word by word shows what each mangled
spelling costs and how it tokenizes. Also writes an interactive per-token view (CircuitsVis) of both passages.

    uv run --extra analysis python analysis/step320_spike/03_token_losses.py
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import tiktoken
from matplotlib.transforms import blended_transform_factory

import page  # also puts analysis/ on sys.path
from common import models, style  # noqa: E402

OUT = os.path.join(page.FIGURES, "word_losses.png")
HTML = os.path.join(page.FIGURES, "token_losses.html")
TOKENS_CSV = os.path.join(page.DATA, "token_losses.csv")
WORDS_CSV = os.path.join(page.DATA, "word_losses.csv")
EXCERPT_START = "Of "  # the excerpt runs from the page's first full sentence to the end of its paragraph
LOSS_MAX = 12.0        # colour ceiling (nats) for the per-token view; the few tokens above it take the darkest shade


def excerpt(text):
    start = text.index(EXCERPT_START)
    return text[start:text.index("\n", start)]


def word_losses(text, tokens, losses):
    """[(word, summed loss, its tokens)] in text order. A token belongs to the word holding its first non-space
    character; punctuation-only tokens belong to no word."""
    spans = [(m.start(), m.end(), m.group()) for m in page.WORD.finditer(text)]
    words = [[w, 0.0, []] for _, _, w in spans]
    pos = 0
    for tok, loss in zip(tokens, losses):
        first = pos + len(tok) - len(tok.lstrip(" "))  # index of the token's first non-space character
        for i, (a, b, _) in enumerate(spans):
            if a <= first < b:  # that character falls inside word i
                words[i][1] += loss
                words[i][2].append(tok.strip())
                break
        pos += len(tok)  # tokens tile the text, so lengths give each one's start
    return words


def main():
    device = models.device()
    enc = tiktoken.get_encoding("gpt2")
    model = models.load_model(page.CKPT, device)
    seq = page.load_sequence()
    if not os.path.exists(page.CLEAN_TXT):
        sys.exit(f"{page.CLEAN_TXT} is missing: run 02_unmangle.py first")
    with open(page.CLEAN_TXT) as f:
        clean_page = f.read()
    page_mean = sum(models.token_losses(model, seq, device)) / (len(seq) - 1)  # whole page, in its training context

    # the excerpt and its twin, each scored from a document boundary
    passages = {}
    for name, text in [("mangled", excerpt(enc.decode(seq))), ("clean", excerpt(clean_page))]:
        ids = [enc.eot_token] + enc.encode_ordinary(text)  # start at a document boundary, as in training
        tokens, losses = models.token_texts(enc, ids)[1:], models.token_losses(model, ids, device)  # aligned: no EOT
        passages[name] = (tokens, losses, word_losses(text, tokens, losses))
    mangled_words, clean_words = passages["mangled"][2], passages["clean"][2]
    assert len(mangled_words) == len(clean_words), "the twin must change spellings only, never the word count"
    pairs = list(zip(clean_words, mangled_words))  # same word count, so position i is the same word
    changed = [(c, m) for c, m in pairs if c[0] != m[0]]
    extra_total = sum(m[1] - c[1] for c, m in pairs)  # all words: includes spill-over onto unchanged ones
    extra_changed = sum(m[1] - c[1] for c, m in changed)

    os.makedirs(page.DATA, exist_ok=True)
    os.makedirs(page.FIGURES, exist_ok=True)
    with open(TOKENS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["passage", "position", "token", "loss"])
        for name, (tokens, losses, _) in passages.items():
            w.writerows((name, i, t, f"{l:.4f}") for i, (t, l) in enumerate(zip(tokens, losses)))
    with open(WORDS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clean", "clean_loss", "clean_tokens", "mangled", "mangled_loss", "mangled_tokens"])
        w.writerows((c[0], f"{c[1]:.3f}", "|".join(c[2]), m[0], f"{m[1]:.3f}", "|".join(m[2])) for c, m in pairs)

    # interactive per-token view: hover a token for its exact loss
    sections = []
    for name, title in [("mangled", "As it appeared in the batch"), ("clean", "Un-mangled twin")]:
        view = models.colored_tokens_html(passages[name][0], passages[name][1], min_value=0, max_value=LOSS_MAX,
                                          positive_color=style.SERIES[0])
        sections.append(f"<h3>{title}</h3>\n{view}")
    with open(HTML, "w") as f:
        f.write(f"<p>Per-token loss (nats) of a trained 113M Gemma-style model on the opening of the step-320 "
                f"page. Darker = higher; hover for values. Whole page: mean loss {page_mean:.2f}.</p>\n"
                + "\n".join(sections))

    # figure: one row per distinct changed word (first occurrence), clean vs mangled loss
    rows = list({(c[0], m[0]): (c, m) for c, m in reversed(changed)}.values())[::-1]  # dedupe, keep first occurrence
    style.apply()
    fig, ax = plt.subplots(figsize=(6.5, 0.34 * len(rows) + 0.6))
    clean_color, mangled_color = style.SERIES[0], style.SERIES[1]
    for row, (c, m) in enumerate(rows):
        ax.plot([c[1], m[1]], [row, row], color=style.GRID, linewidth=2.5, zorder=1, solid_capstyle="round")
        ax.scatter([c[1]], [row], s=48, color=clean_color, zorder=3, label="Real spelling" if row == 0 else None)
        ax.scatter([m[1]], [row], s=48, color=mangled_color, zorder=3, label="Baby-talk spelling" if row == 0 else None)
    # Table-like layout: the real word's tokens label each row on the left, the baby-talk tokens sit in a
    # column on the right, and the plot between them carries only the dots.
    mono = dict(family="DejaVu Sans Mono", fontsize=8.5, color=style.INK)
    ax.set_yticks(range(len(rows)), [" · ".join(c[2]) for c, _ in rows], **mono)
    ax.tick_params(axis="y", length=0, pad=6)
    right_column = blended_transform_factory(ax.transAxes, ax.transData)
    for row, (_, m) in enumerate(rows):
        ax.text(1.02, row, " · ".join(m[2]), transform=right_column, va="center", **mono)
    ax.set_ylim(len(rows) - 0.5, -0.5)  # text order, top to bottom, no padding rows
    ax.set_xlim(0, max(m[1] for _, m in rows) * 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Loss on the word (nats, summed over its tokens)")
    # title, subtitle and legend stacked just above the axes, in points, so the saved figure crops tight
    header = [(54, "What each baby-talk word costs the model", dict(fontsize=12, fontweight="bold")),
              (36, f"Baby-talk words carry {extra_changed:.0f} of the excerpt's {extra_total:.0f} extra nats; "
                   "the rest spills onto the words after them.", dict(fontsize=9, color=style.INK_SECONDARY)),
              (22, "Row labels: how GPT-2's tokenizer splits the real word (left) and the baby-talk word (right).",
               dict(fontsize=9, color=style.INK_SECONDARY))]
    for offset, text, kw in header:
        ax.annotate(text, (0, 1), xycoords="axes fraction", xytext=(0, offset), textcoords="offset points",
                    va="bottom", **kw)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5, borderaxespad=0.2,
              handletextpad=0.2, columnspacing=1.2)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.12)

    print(f"wrote {OUT}, {HTML}")
    print(f"  page mean loss {page_mean:.3f}; extra loss {extra_total:.1f} nats, {extra_changed:.1f} on "
          f"{len(changed)} changed words")


if __name__ == "__main__":
    main()
