"""Matplotlib style for this repo's figures. The rules are in README.md beside this file.

    from steering_reasoning import plot_style
    plot_style.use()            # experiment figures
    plot_style.use("paper")     # paper figures: the paper wheel and thinner marks

Applies base.mplstyle and registers the two heatmap colormaps: `paper_teal`
(unipolar) and `verm_paper_teal` (bipolar). `paper_teal` becomes the default
image colormap.
"""
import math
import os

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER, INK, TEAL, VERMILLION = "#fefdfc", "#111111", "#3a7d85", "#b8442b"
MUTED, FAINT, RULE = "#4a4a4a", "#9a9a9a", "#e5e5e5"
NEUTRAL = "#7d7d7d"  # a series that is neither good nor bad: control, reference, chance

# Series with no valence take a wheel in order; vermillion is never in it.
# experiments: teal, ochre, sage, slate, plum
CATEGORICAL = ["#3a7d85", "#9c6c24", "#5a6f33", "#356a7e", "#76566f"]
# paper: teal, yellow, green, cool grey, purple (close in lightness, differ by hue)
PAPER_CATEGORICAL = ["#3a7d85", "#b3952f", "#5a7a3a", "#6b7a88", "#78628e"]

_PAPER_RC = {
    "axes.prop_cycle": cycler(color=PAPER_CATEGORICAL),
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "lines.markeredgewidth": 0.5,
}


def _register():
    for name, colors in (("paper_teal", [PAPER, TEAL]),
                         ("verm_paper_teal", [VERMILLION, PAPER, TEAL])):
        if name not in plt.colormaps:
            cmap = LinearSegmentedColormap.from_list(name, colors)
            matplotlib.colormaps.register(cmap, name=name)
            matplotlib.colormaps.register(cmap.reversed(), name=name + "_r")


def use(variant="experiments"):
    """Apply the style. `variant` is "experiments" (default) or "paper"."""
    if variant not in ("experiments", "paper"):
        raise ValueError(f"unknown variant {variant!r}")
    plt.style.use(os.path.join(HERE, "base.mplstyle"))
    _register()
    plt.rcParams["image.cmap"] = "paper_teal"
    if variant == "paper":
        plt.rcParams.update(_PAPER_RC)


def fmt_value(v):
    """About three significant digits, at most two decimals: 0.85, 1.74, 25.7, 132."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    v = float(v)
    a = abs(v)
    if a >= 100 or v.is_integer():
        return "%d" % round(v)
    dec = 1 if a >= 10 else 2
    return ("%.*f" % (dec, v)).rstrip("0").rstrip(".")


def nice_ceiling(x):
    """The next round number at or above x: 1.8 -> 2, 0.85 -> 1, 25.7 -> 30, 132 -> 150."""
    x = abs(float(x))
    if x == 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(x))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * mag >= x * (1 - 1e-9):
            return m * mag
    return 10 * mag


def heatmap(ax, data, bipolar=None, values=True, cmap=None, vmin=None, vmax=None, fontsize=7):
    """A heatmap by the rules: paper->teal for unipolar data, vermillion->paper->teal
    centred on zero for bipolar, ink edges on every cell, and the numbers in the
    cells. Returns the image."""
    data = np.asarray(data, dtype=float)
    if bipolar is None:
        bipolar = np.nanmin(data) < 0 < np.nanmax(data)
    if cmap is None:
        cmap = "verm_paper_teal" if bipolar else "paper_teal"
    if bipolar and vmin is None and vmax is None:
        a = nice_ceiling(np.nanmax(np.abs(data)))       # symmetric, with headroom
        vmin, vmax = -a, a
    elif not bipolar:
        if vmin is None:
            vmin = min(0.0, float(np.nanmin(data)))
        if vmax is None:
            vmax = nice_ceiling(np.nanmax(data))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ink_edges(im, ax)
    if values:
        norm = im.norm
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                t = norm(v)
                dark = (abs(t - 0.5) > 0.32) if bipolar else (t > 0.6)
                ax.text(j, i, fmt_value(v), ha="center", va="center", fontsize=fontsize,
                        family="monospace", color=PAPER if dark else INK)
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    return im


def ink_edges(im, ax):
    """Solid ink border around every heatmap cell (imshow draws none itself)."""
    h, w = im.get_array().shape[:2]
    for i in range(h):
        for j in range(w):
            # clip_on=False: otherwise the outer cells lose half their stroke at the
            # axes boundary and the frame comes out lighter than the inner lines
            ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK,
                                           linewidth=1.0, clip_on=False, zorder=3))


def mono(*axes):
    """Kept for older scripts; the style is mono throughout, so this is a no-op safeguard."""
    for ax in axes:
        for lab in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]:
            lab.set_family("monospace")
        leg = ax.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_family("monospace")
