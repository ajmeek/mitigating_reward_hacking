"""Tabulate the GLM (or any) grades: rate of lie / evade / truth by prompt x dose, paired tests, trend.
usage: python scripts/analyze.py [grades.jsonl]"""
import json, math, os, random, sys, collections
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
f = sys.argv[1] if len(sys.argv) > 1 else "artifacts/codes/grades_glm.jsonl"
g = [json.loads(l) for l in open(f)]; g = [o for o in g if "answer" in o]
A = ["addn16","addn12","addn8","addn4","base","addp4","addp8","addp12","addp16"]; D = {a: (0 if a == "base" else (1 if a[3] == "p" else -1) * int(a[4:])) for a in A}
L = {a: ("base" if a == "base" else f"{D[a]:+d}u") for a in A}
P = ["resume","package","late","rye","dent","reddot"]   # caddy (all versions) and homework excluded: floor prompts
g = [o for o in g if o["prompt"] in P]
cell = {(o["prompt"], o["arm"], o["sample"]): o["answer"] for o in g}
n_cell = collections.Counter((p, a) for p, a, s in cell)
print(len(g), "graded;", collections.Counter(o["answer"] for o in g))
def rate(p, a, lab):
    v = [x for (pp, aa, s), x in cell.items() if pp == p and aa == a]; return (sum(x == lab for x in v) / len(v), len(v)) if v else (float("nan"), 0)
for lab in ("lie", "evade", "truth", "none"):
    print(f"\n{lab.upper()} rate\n| prompt | " + " | ".join(L[a] for a in A) + " |\n|---" * 10 + "|")
    for p in P: print(f"| {p} | " + " | ".join((f"{rate(p,a,lab)[0]:.2f}" if rate(p,a,lab)[1] else "–") for a in A) + " |")
    pooled = []
    for a in A:
        v = [x for (pp, aa, s), x in cell.items() if aa == a]; pooled.append(sum(x == lab for x in v) / len(v) if v else float("nan"))
    print("| **pooled** | " + " | ".join(f"**{x:.2f}**" for x in pooled) + " |")
print("\nn per (prompt, arm): min", min(n_cell.values()), "max", max(n_cell.values()))
# paired lie vs base and trend
def paired(a):
    d = [(cell[(p,a,s)] == "lie") - (cell[(p,"base",s)] == "lie") for (p, aa, s) in cell if aa == "base" and (p, a, s) in cell]
    if not d: return None
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d); m = pos + neg
    pv = min(1, 2 * sum(math.comb(m, k) for k in range(min(pos, neg) + 1)) / 2**m) if m else 1
    return sum(d) / len(d), pos, neg, len(d), pv
print("\nLIE paired vs base:")
for a in A:
    if a != "base" and paired(a): m, pos, neg, n, pv = paired(a); print(f"  {L[a]:5s} {m:+.3f}  flips more/less {pos}/{neg} of {n}  p={pv:.3g}")
units = sorted({(p, s) for (p, a, s) in cell})
def slope(us):
    xs, ys = [], []
    for p, s in us:
        for a in A:
            if (p, a, s) in cell: xs.append(D[a] / 4); ys.append(cell[(p, a, s)] == "lie")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys); return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
rng = random.Random(0); bs = sorted(slope([rng.choice(units) for _ in units]) for _ in range(1000))
print(f"lie trend: {slope(units):+.4f} per 4u  90% CI [{bs[50]:+.4f}, {bs[949]:+.4f}]")
