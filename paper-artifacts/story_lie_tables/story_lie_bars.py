"""Pooled lie and truth rate by addition scale, grouped bars with 90% Wilson intervals (ink), six story prompts.
Reads experiments/story-dilemma-steering/artifacts/codes/grades_glm.jsonl; writes story_lie_bars.{pdf,png} beside this script.
    uv run --with matplotlib --with numpy python paper-artifacts/story_lie_tables/story_lie_bars.py"""
import json, math, os, sys, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_style  # noqa: E402
paper_style.use()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
A = ["addn16","addn12","addn8","addn4","base","addp4","addp8","addp12","addp16"]; X = [-16,-12,-8,-4,0,4,8,12,16]
P = {"resume","package","late","rye","dent","reddot"}
g = [json.loads(l) for l in open(f"{ROOT}/experiments/story-dilemma-steering/artifacts/codes/grades_glm.jsonl")]
g = [o for o in g if "answer" in o and o["prompt"] in P]
def wilson(k, n, z=1.645):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
fig, ax = plt.subplots(figsize=(5.2, 2.6))
w = 1.15
for off, lab, col in ((-w / 2, "lie", paper_style.VERMILLION), (w / 2, "honest", paper_style.TEAL)):
    ks = [sum(o["answer"] == ("truth" if lab == "honest" else lab) for o in g if o["arm"] == a) for a in A]; ns = [sum(o["arm"] == a for o in g) for a in A]
    ys = [k / n for k, n in zip(ks, ns)]; ci = [wilson(k, n) for k, n in zip(ks, ns)]
    ax.bar([x + off for x in X], ys, width=w, color=col, alpha=0.8, edgecolor=paper_style.INK, linewidth=0.5, label=lab)
    ax.errorbar([x + off for x in X], ys, yerr=[[y - lo for y, (lo, hi) in zip(ys, ci)], [hi - y for y, (lo, hi) in zip(ys, ci)]],
                fmt="none", ecolor=paper_style.INK, elinewidth=0.6, capsize=0)
ax.set_xticks(X); ax.set_xticklabels([f"{x:+d}" if x else "0" for x in X]); ax.set_xlim(-19.5, 19.5); ax.set_ylim(0, 1)
ax.set_xlabel(r"steering coefficient $s$  ($h_t \leftarrow h_t + s\,u$)"); ax.set_ylabel("rate of stories")
ax.legend(frameon=False, loc="upper left", ncol=2)
fig.tight_layout(); paper_style.save(fig, "story_lie_bars")
print("n per dose:", [sum(o["arm"] == a for o in g) for a in A])
