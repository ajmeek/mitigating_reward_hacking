"""The repo figure style (steering_reasoning.plot_style, "paper" variant) with Computer Modern Sans for the paper.

    import paper_style; paper_style.use()
"""
import matplotlib.pyplot as plt

from steering_reasoning import plot_style  # the repo's vendored figure style (src/steering_reasoning/plot_style)

PAPER, INK, TEAL, VERMILLION = plot_style.PAPER, plot_style.INK, plot_style.TEAL, plot_style.VERMILLION
NEUTRAL = plot_style.NEUTRAL          # a series that is neither good nor bad: control, reference, chance
MUTED, FAINT, RULE = "#4a4a4a", "#9a9a9a", "#e5e5e5"
CATEGORICAL = plot_style.PAPER_CATEGORICAL  # the paper wheel for series with no valence: teal, yellow, green, grey, purple

def _mix(c1, c2, t):
    """Linear blend of two hex colors, t=0 -> c1, t=1 -> c2."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def teal_ramp(n):
    """n teals from light to dark for ordered treatments (dose levels); the middle is
    teal-muted. Use VERMILLION for the untreated baseline alongside these."""
    if n == 1:
        return [TEAL]
    stops = [_mix(TEAL, PAPER, 0.38), TEAL, _mix(TEAL, INK, 0.45)]
    import numpy as np
    xs = np.linspace(0, 2, n)
    out = []
    for x in xs:
        i = min(int(x), 1)
        out.append(_mix(stops[i], stops[i + 1], x - i))
    return out


# Widths in inches for a two-column paper.
COL, FULL = 3.3, 6.75


def use():
    plot_style.use("paper")
    # CMU Sans Serif from TeX Live's cm-unicode when it is not installed system-wide (Austin's box).
    import glob
    from matplotlib import font_manager
    for f in glob.glob("/usr/share/texlive/texmf-dist/fonts/opentype/public/cm-unicode/cmunss*.otf"):
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif", "Computer Modern Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,  # CMU Sans Serif has no U+2212 glyph
        "axes.formatter.use_mathtext": False,
    })


def save(fig, stem, where=None):
    """Write <stem>.pdf and .png beside the calling script (or into `where`); return the png path."""
    import inspect
    import os
    if where is None:
        where = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(where, f"{stem}.{ext}"))
    return os.path.join(where, f"{stem}.png")
