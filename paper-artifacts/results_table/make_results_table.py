"""The paper's full results table (artifacts/clamp_summary_table.tex, Table "clamp-summary"): one row per
ImpossibleBench arm at the 120-message budget, n = 60 (30 test tasks x seeds 42/43), with hack rate on the
conflicting split, honest-submission rate on the same rollouts, and solve rate on the original split, each
with its 90% Wilson interval.

Sources (all at 120 messages):
  experiments/diff-mean-steering/artifacts/summary_fleet.csv   u / optimized arms, hack + solve (*_ext columns)
  experiments/spec-prompting/artifacts/summary_m120.csv                      spec prompt, hack + solve
  experiments/spec-vector/artifacts/results_120msg/summary.csv                  v_spec floor / add arms, hack + solve
  experiments/honest-submission-judge/data/labels.jsonl, labels_spec.jsonl       honest submissions (gpt-6-luna judge)
Intervals are recomputed from count / n here so the three columns share one formula.

    python paper-artifacts/results_table/make_results_table.py
"""
import csv, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "../.."))
FLEET = os.path.join(REPO, "experiments/diff-mean-steering/artifacts/summary_fleet.csv")
SPEC_PROMPT = os.path.join(REPO, "experiments/spec-prompting/artifacts/summary_m120.csv")
SPEC_VECTOR = os.path.join(REPO, "experiments/spec-vector/artifacts/results_120msg/summary.csv")
LABELS = [os.path.join(REPO, "experiments/honest-submission-judge/data", f) for f in ("labels.jsonl", "labels_spec.jsonl")]
N = 60
# blocks of (label, hack/solve source key, honest-label paper_row)
BLOCKS = [
    [("baseline", "base", "baseline"), ("$u$ cap 0", "c0", "u cap 0"), ("$u$ cap $-$4", "cn4", "u cap -4"),
     ("$u$ cap $-$8", "cn8", "u cap -8"), ("$u$ cap $-$12", "cn12", "u cap -12"), ("$u$ cap $-$16", "cn16", "u cap -16")],
    [("$u$ add $-$2", "addn2", "u add -2"), ("$u$ add $-$4", "addn4", "u add -4"), ("$u$ add $-$6", "addn6", "u add -6"),
     ("$u$ add $-$8", "addn8", "u add -8")],
    [("opt cap $-$4", "optc_r4", "opt cap -4"), ("opt cap $-$8", "optc_r8", "opt cap -8"), ("opt cap $-$12", "optc_r12", "opt cap -12")],
    [("opt add 12", "opta_k12", "opt add 12"), ("opt add 16", "opta_k16", "opt add 16"), ("opt add 20", "opta_k20", "opt add 20")],
    [("spec prompt", "spec", "spec prompt")],
    [(r"$v_{\mathrm{spec}}$ floor $+5$", "v1floor5", "v_spec floor +5"), (r"$v_{\mathrm{spec}}$ floor $+10$", "v1floor10", "v_spec floor +10"),
     (r"$v_{\mathrm{spec}}$ floor $+16.5$", "v1floor16", "v_spec floor +16.5"),
     (r"$v_{\mathrm{spec}}$ add $+8$", "v1add8", "v_spec add +8"), (r"$v_{\mathrm{spec}}$ add $+16$", "v1add16", "v_spec add +16")],
]


def wilson(k, n, z=1.6448536269514722):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def counts():
    """{arm key: {"hack": k, "solve": k}} at 120 messages, n = 60 each."""
    out = {}
    for r in csv.DictReader(open(FLEET)):
        assert r["extend_limit"] == "120" and r["n"] == str(N), r
        out.setdefault(r["arm"], {})["hack" if r["metric"] == "hack rate" else "solve"] = int(r["count_ext"])
    for path, arms in ((SPEC_PROMPT, {"spec"}), (SPEC_VECTOR, {"v1floor5", "v1floor10", "v1floor16", "v1add8", "v1add16"})):
        for r in csv.DictReader(open(path)):
            if r["arm"] in arms and r["group"] == "all":
                assert r["n_scored"] == str(N), r
                out.setdefault(r["arm"], {})["hack" if r["variant"] == "conflicting" else "solve"] = int(r["count"])
    return out


def honest():
    raw = [json.loads(l) for f in LABELS if os.path.exists(f) for l in open(f) if l.strip()]
    rows = {r["job_id"]: r for r in raw if r.get("label") in ("HONEST", "NOT")}.values()
    out = {}
    for r in rows:
        out.setdefault(r["paper_row"], []).append(r)
    return out


def main():
    cnt, hon = counts(), honest()
    lines = [r"\begin{tabular}{l rr rr rr}", r"\toprule",
             r" & \multicolumn{2}{c}{hack} & \multicolumn{2}{c}{honest submit} & \multicolumn{2}{c}{solve} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
             r"arm & rate & 90\% CI & rate & 90\% CI & rate & 90\% CI \\", r"\midrule"]
    for bi, block in enumerate(BLOCKS):
        if bi:
            lines.append(r"\midrule")
        for label, key, hrow in block:
            c = cnt[key]; rs = hon[hrow]
            assert len(rs) == N, (hrow, len(rs))
            h = sum(r["label"] == "HONEST" for r in rs)
            assert sum(r["outcome_final"] == "1" for r in rs) == c["hack"], (key, "judge rows disagree with the hack count")
            cells = [label]
            for k in (c["hack"], h, c["solve"]):
                lo, hi = wilson(k, N)
                cells += [f"{k / N:.2f}", f"({lo:.2f}, {hi:.2f})"]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = os.path.join(HERE, "clamp_summary_table.tex")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{sum(len(b) for b in BLOCKS)} arms -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
