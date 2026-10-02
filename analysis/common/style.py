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

# Sequential, one hue light -> dark, for magnitudes such as loss.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 1.6, "legend.frameon": False,
    })


def spread_labels(ys, min_gap):
    """Nudge direct-label positions apart so near-identical lines don't print on top of each other."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])  # label indices, lowest y first
    placed = list(ys)
    for a, b in zip(order, order[1:]):
        placed[b] = max(placed[b], placed[a] + min_gap)  # push each label up to clear the one below
    return placed
