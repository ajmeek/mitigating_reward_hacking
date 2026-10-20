"""Held-out judge segments in the (e, u_perp) plane at layer 35, the same plane as
action_plane.py: x = e (canonical bf16 approach-to-action axis, see
src/steering_reasoning/canonical/INDEX.md), y = u_perp, canonical u with its e component
projected out. Each point is one segment's mean activation on the bf16 per-token capture,
centred on that capture's mean (probe-refit-bf16/artifacts/mu_bf16_L35.safetensors). Val + test
rollouts only. -u (the direction the clamp pushes toward) drawn as an arrow; the same canonical
u and e define the coordinates, so the arrow and the points share one basis.
Labels: x "approach to action, e", y "hack vs. decline, u_perp" (the caption defines u_perp);
legend hack / decline / other, the same order as action_plane.py.
Reads hack-action-anchor/artifacts/segment_projections_eu_bf16.csv
(hack-action-anchor/scripts/segment_projections_eu.py)."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_style  # noqa: E402

paper_style.use()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); HA = f"{ROOT}/experiments/hack-action-anchor/artifacts"
d = pd.read_csv(f"{HA}/segment_projections_eu_bf16.csv"); d = d[d.split.isin(["val", "test"])]
sys.path.insert(0, f"{ROOT}/src")
from steering_reasoning.canonical import load  # noqa: E402  (canonical u and e, see src/steering_reasoning/canonical/INDEX.md)
u, e = load("u"), load("e"); ue = float(u @ e); u_xy = np.array([ue, (1 - ue ** 2) ** 0.5])

fig, ax = plt.subplots(figsize=(paper_style.COL + 0.4, paper_style.COL + 0.2))
O = d[d.cls == "other"]
h = {"other": ax.scatter(O.e, O.u_perp, s=3, color=paper_style.FAINT, alpha=0.35, lw=0, rasterized=True, zorder=1, label="other")}
for cls, color in [("decline", paper_style.TEAL), ("hack", paper_style.VERMILLION)]:
    g = d[d.cls == cls]; h[cls] = ax.scatter(g.e, g.u_perp, s=7, color=color, alpha=0.8, lw=0, zorder=3, label=cls)
R = 12
ax.annotate("", xy=tuple(-R * u_xy), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=paper_style.INK, lw=0.9, shrinkA=0, shrinkB=0), zorder=4)   # -u: the direction the clamp pushes toward
ax.text(*(-R * 1.1 * u_xy), r"$-u$", ha="center", va="center", fontsize=9, color=paper_style.INK, zorder=4)
ax.axhline(0, color=paper_style.RULE, lw=0.8, zorder=0); ax.axvline(0, color=paper_style.RULE, lw=0.8, zorder=0)
ax.set_xlabel(r"approach to action, $e$"); ax.set_ylabel(r"hack vs. decline, $u_{\perp e}$")
ax.set_aspect("equal", adjustable="datalim")   # keep the axes box fixed so it matches action_plane_hackedit.py
# legend markers all the same size and opacity, whatever the plotted point size (same order as action_plane.py)
from matplotlib.lines import Line2D  # noqa: E402
lc = {"hack": paper_style.VERMILLION, "decline": paper_style.TEAL, "other": paper_style.FAINT}
leg = ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4, mew=0, color=lc[k], label=k) for k in ("hack", "decline", "other")],
                loc="upper left", handlelength=1.0)
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.13, top=0.97)   # same axes box as action_plane_ewma.py
print(d.cls.value_counts().to_dict())
print(paper_style.save(fig, "segment_scatter"))
