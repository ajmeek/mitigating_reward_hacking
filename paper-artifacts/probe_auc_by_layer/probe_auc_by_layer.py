"""Val AUROC by layer for the DiffMean direction u = norm(a_hat - b_hat) (a: hack pole,
b: decline pole) on two contrasts: hack vs decline segments, and hack vs every other segment.
bf16 fit, one u per layer on the train rollouts, scored on val. The sweep covers all 64
layers; the figure shows every 4th from 3 (3, 7, ..., 63), the grid the steering layer was
selected on, where L35 is the argmax of hack vs decline.
Reads probe-refit-bf16/artifacts/sweep_bf16.json.
Colors: the two contrasts have no valence, so they take the wheel in order."""
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_style  # noqa: E402

paper_style.use()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
val = json.load(open(f"{ROOT}/experiments/probe-refit-bf16/artifacts/sweep_bf16.json"))["val"]
layers = list(range(3, 64, 4))
CHOSEN = 35
assert max(layers, key=lambda L: val[str(L)]["u_hva"]) == CHOSEN

C = paper_style.CATEGORICAL
series = [("u_hva", C[0], "hack vs. decline"), ("u_hack_vs_rest", C[1], "hack vs. other")]
fig, ax = plt.subplots(figsize=(paper_style.COL, 0.72 * paper_style.COL))
ax.axhline(0.5, color=paper_style.FAINT, lw=0.7, ls=(0, (3, 3)), zorder=1)
for key, color, label in series:
    ax.plot(layers, [val[str(L)][key] for L in layers], color=color, lw=0.9, marker="o", ms=2.5,
            markeredgecolor=paper_style.INK, markeredgewidth=0.5, label=label, zorder=3)
# the steering layer: the hack vs decline dot at L35 is a star
ax.plot([CHOSEN], [val[str(CHOSEN)]["u_hva"]], ls="none", marker="*", ms=7, color=C[0],
        markeredgecolor=paper_style.INK, markeredgewidth=0.5, zorder=4)
ax.set_xlim(0, 66); ax.set_xticks([15, 31, 47, 63])
ax.set_xlabel("Layer")
ax.set_ylim(0.25, 1.05); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("AUROC")
ax.tick_params(length=2.5, width=0.6)
for sp in ("left", "bottom"):
    ax.spines[sp].set_linewidth(0.7)
ax.legend(loc="lower right", handlelength=1.4, borderaxespad=0.2)
fig.tight_layout()
print(paper_style.save(fig, "probe_auc_by_layer"))
