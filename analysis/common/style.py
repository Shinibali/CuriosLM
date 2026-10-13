"""Shared figure style for the post: one palette and one set of mark rules for every chart."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Categorical, in fixed order (never cycled). Validated for colour-vision deficiency on adjacent pairs; the 3rd and
# 4th sit below 3:1 contrast on SURFACE, so charts using them must carry direct labels.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# Model identity: the same model gets the same colour in every figure, and blue and orange mean nothing else.
MODEL_A, MODEL_B, GPT2 = SERIES[0], SERIES[1], SERIES[2]
MODEL_B_MIX = "#c2417a"  # Model B's architecture on the 50:50 data mix (rose)
MODEL_A_DARK = "#104281"  # a second Model A run (e.g. the 9.8B-token one beside the 1.5B), same hue, darker

# Everything that isn't a model gets colours of its own.
COMPONENTS = {"attention projections": "#5b3fa8", "attention scores": "#b3a1e6",  # attention: one purple family
              "MLP": "#127e7e", "output layer": "#857e72"}  # teal, dark warm grey (distinct from the background)
ACCENT = "#127e7e"  # a single highlighted series that isn't a model (teal)
OTHER = ["#5b3fa8", "#127e7e", "#c28a00"]  # non-model categories in fixed order: purple, teal, gold
GOOD, BAD = "#127e7e", "#c28a00"  # better / worse (teal, gold)
SEQUENTIAL_CMAP = "Purples"  # heatmaps of non-model quantities

# Sequential, one hue light -> dark, for magnitudes such as loss.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,  # grid behind the data
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 1.6, "legend.frameon": False,
    })


def mark_uniform_loss(ax, label=True):
    """A dashed line at ln(vocabulary size): the loss of a uniform guess over GPT-2's tokens, where an untrained model
    starts (slightly above it, since random weights aren't perfectly uniform). The size comes from the tokenizer."""
    import math
    import tiktoken
    vocabulary = tiktoken.get_encoding("gpt2").n_vocab  # 50,257
    loss = math.log(vocabulary)
    ax.axhline(loss, color=INK_SECONDARY, linewidth=0.9, linestyle="--", zorder=1)
    if label:
        ax.text(1, loss, f"uniform guess over {vocabulary:,} tokens: ln {vocabulary:,} = {loss:.2f}  ", ha="right",
                va="bottom", fontsize=8.5, color=INK_SECONDARY, transform=ax.get_yaxis_transform())
    return loss


def spread_labels(ys, min_gap):
    """Nudge direct-label positions apart so near-identical lines don't print on top of each other."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])  # label indices, lowest y first
    placed = list(ys)
    for a, b in zip(order, order[1:]):
        placed[b] = max(placed[b], placed[a] + min_gap)  # push each label up to clear the one below
    return placed
