"""Table 7 candidate with intervals, colour and Spearman rows, from the story-dilemma GLM grades
(experiments/story-dilemma-steering/artifacts/codes/grades_glm.jsonl; six stories, nine addition scales, n = 20 per cell).
Two stacked blocks (lie rate, honesty rate), one column per story: each cell is the rate with its 90% Wilson interval and a
background tint proportional to the rate (vermillion for lies, teal for honesty; needs the colortbl package). The last row
of each block is Spearman's rho between the steering coefficient s and that story's rate over the nine doses (no interval in
the table; story_spearman.json keeps an approximate Fisher-z / Bonett-Wright one per story). The json also holds the mean rho
across the six (independent) stories with a 90% t-interval over the stories (df = 5), for the Figure 6 caption.
Writes story_by_story_ci.tex and story_spearman.json beside this script. Does not touch story_by_story.tex.
    uv run python paper-artifacts/story_lie_tables/story_by_story_ci.py
"""
import collections, json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
G = os.path.join(ROOT, "experiments/story-dilemma-steering/artifacts/codes/grades_glm.jsonl")
A = ["addn16", "addn12", "addn8", "addn4", "base", "addp4", "addp8", "addp12", "addp16"]
S = np.array([-16, -12, -8, -4, 0, 4, 8, 12, 16], float)
H = [r"$-16$", r"$-12$", r"$-8$", r"$-4$", "0", "$+4$", "$+8$", "$+12$", "$+16$"]
P = ["resume", "package", "late", "rye", "dent", "reddot"]
SHORT = {"resume": "r\\'esum\\'e", "package": "stereo", "late": "play", "rye": "rye", "dent": "scrape", "reddot": "sweater"}
Z = 1.645
g = [json.loads(l) for l in open(G)]; g = [o for o in g if "answer" in o and o["prompt"] in P]
cell = collections.defaultdict(list)
for o in g: cell[(o["prompt"], o["arm"])].append(o["answer"])
# indicator arrays per (story, arm): outcomes[p][a] = {lie: 0/1 array, truth: 0/1 array}
IND = {p: {a: {lab: np.array([v == lab for v in cell[(p, a)]], float) for lab in ("lie", "truth")} for a in A} for p in P}
N = {p: {a: len(cell[(p, a)]) for a in A} for p in P}

def wilson(k, n, z=Z):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

def rankavg(x):
    """Average ranks with ties (what scipy.stats.rankdata does by default)."""
    x = np.asarray(x); order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r

def spearman(x, y):
    rx, ry = rankavg(x), rankavg(y)
    if rx.std() == 0 or ry.std() == 0: return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

rates = {p: {lab: np.array([IND[p][a][lab].mean() for a in A]) for lab in ("lie", "truth")} for p in P}
rho = {lab: {p: spearman(S, rates[p][lab]) for p in P} for lab in ("lie", "truth")}
def fisher_ci(r, n, z=Z):
    """Approximate 90% interval for Spearman's rho from n pairs: Fisher's z with the Bonett & Wright (2000) standard error
    sqrt((1 + rho^2 / 2) / (n - 3)), which is wider than the Pearson one. The nine doses are fixed and the cells are small
    (n = 20), so this is a rough interval; the mean over stories below is the better-founded summary."""
    if math.isnan(r): return float("nan"), float("nan")
    r = min(max(r, -0.999999), 0.999999); zf = math.atanh(r); se = math.sqrt((1 + r * r / 2) / (n - 3))
    return math.tanh(zf - z * se), math.tanh(zf + z * se)
T90_5 = 2.015   # t_{0.95, 5}: the six stories are independent, so the mean rho gets a t-interval over them
out = {"n_per_cell": sorted({N[p][a] for p in P for a in A}), "n_doses": len(A),
       "per_story_interval": "approximate 90% Fisher-z interval over the nine doses, Bonett-Wright standard error", "mean_interval": "90% t-interval over the six stories (df = 5)"}
for lab in ("lie", "truth"):
    rs = np.array([rho[lab][p] for p in P])
    out[lab] = {"per_story": {p: {"rho": rho[lab][p], "ci90": fisher_ci(rho[lab][p], len(A))} for p in P},
                "mean_rho": float(rs.mean()), "mean_ci90": (float(rs.mean() - T90_5 * rs.std(ddof=1) / math.sqrt(len(P))), float(rs.mean() + T90_5 * rs.std(ddof=1) / math.sqrt(len(P))))}
json.dump(out, open(os.path.join(HERE, "story_spearman.json"), "w"), indent=1)

# ---- table ----------------------------------------------------------------------------------------------------
TINT = {"lie": ("lietint", "B8442B"), "truth": ("honesttint", "3A7D85")}
MAXPCT = 55   # background strength at rate 1.0 (percent of the tint colour)
def cellstr(lab, p, a):
    k = int(IND[p][a][lab].sum()); n = N[p][a]; r = k / n; lo, hi = wilson(k, n)
    return r"\cellcolor{%s!%d}%.2f {\scriptsize[%.2f, %.2f]}" % (TINT[lab][0], round(MAXPCT * r), r, lo, hi)
def num(v): return ("$-$" if v < 0 else "") + "%.2f" % abs(v)   # text digits with a proper minus sign, no math spacing
k = len(P); C = r"w{r}{1.75cm}"
L = ["% generated by paper-artifacts/story_lie_tables/story_by_story_ci.py; needs \\usepackage{colortbl}",
     r"\definecolor{%s}{HTML}{%s}%%" % TINT["lie"], r"\definecolor{%s}{HTML}{%s}%%" % TINT["truth"]]
for lab, title in (("lie", "lie rate"), ("truth", "honesty rate")):
    L += [r"\begin{tabular}{w{r}{0.6cm} " + C * k + "}", r"\toprule",
          r" & \multicolumn{" + str(k) + r"}{c}{" + title + r"} \\", r"\cmidrule(lr){2-" + str(k + 1) + "}",
          "$s$ & " + " & ".join(r"{\scriptsize " + SHORT[p] + "}" for p in P) + r" \\", r"\midrule"]
    for a, h in zip(A, H):
        L.append(h + " & " + " & ".join(cellstr(lab, p, a) for p in P) + r" \\")
    L += [r"\midrule", r"$\rho$ & " + " & ".join(num(rho[lab][p]) for p in P) + r" \\", r"\bottomrule", r"\end{tabular}"]
    if lab == "lie": L.append(r"\\[0.8em]")
open(os.path.join(HERE, "story_by_story_ci.tex"), "w").write("\n".join(L) + "\n")
for lab in ("lie", "truth"):
    m, (lo, hi) = out[lab]["mean_rho"], out[lab]["mean_ci90"]
    print(f"{lab:5s} mean rho over {k} stories = {m:+.2f} [{lo:+.2f}, {hi:+.2f}]; per story " +
          ", ".join(f"{p} {rho[lab][p]:+.2f} [{out[lab]['per_story'][p]['ci90'][0]:+.2f}, {out[lab]['per_story'][p]['ci90'][1]:+.2f}]" for p in P))
print(f"n per cell {out['n_per_cell']}; wrote story_by_story_ci.tex and story_spearman.json to {HERE}")
