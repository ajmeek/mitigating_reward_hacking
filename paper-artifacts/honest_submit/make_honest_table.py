"""Honest-submission table (paper: artifacts/honest_submit_summary_table.tex) from the judge's labels.

Reads experiments/honest-submission-judge/data/labels.jsonl (16 main-sweep arms, results of record) and, when
present, labels_spec.jsonl (the seven model-spec arms), counts HONEST per arm, and writes
honest_submit_summary_table.tex (rate + 90% Wilson interval, the paper's layout) and honest_counts.csv
(n, honest, hacked, neither per arm). Arm order and labels are fixed below; an arm with no labels is skipped.

    python paper-artifacts/honest_submit/make_honest_table.py
"""
import csv, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "../.."))
LABELS = [os.path.join(REPO, "experiments/honest-submission-judge/data", f) for f in ("labels.jsonl", "labels_spec.jsonl")]
# blocks of (paper_row in labels.jsonl, label in the table)
BLOCKS = [
    [("baseline", "baseline"), ("u cap 0", "$u$ cap 0"), ("u cap -4", "$u$ cap $-$4"), ("u cap -8", "$u$ cap $-$8"),
     ("u cap -12", "$u$ cap $-$12"), ("u cap -16", "$u$ cap $-$16")],
    [("u add -2", "$u$ add $-$2"), ("u add -4", "$u$ add $-$4"), ("u add -6", "$u$ add $-$6"), ("u add -8", "$u$ add $-$8")],
    [("opt cap -4", "opt cap $-$4"), ("opt cap -8", "opt cap $-$8"), ("opt cap -12", "opt cap $-$12")],
    [("opt add 12", "opt add 12"), ("opt add 16", "opt add 16"), ("opt add 20", "opt add 20")],
    [("spec baseline", "baseline (spec runs)"), ("spec prompt", "spec prompt"),
     ("v_spec floor +5", r"$v_{\mathrm{spec}}$ floor $+5$"), ("v_spec floor +10", r"$v_{\mathrm{spec}}$ floor $+10$"),
     ("v_spec floor +16.5", r"$v_{\mathrm{spec}}$ floor $+16.5$"),
     ("v_spec add +8", r"$v_{\mathrm{spec}}$ add $+8$"), ("v_spec add +16", r"$v_{\mathrm{spec}}$ add $+16$")],
]


def wilson(k, n, z=1.6448536269514722):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    missing = [f for f in LABELS if not os.path.exists(f)]
    assert not missing, f"judge label files missing (both are tracked): {missing}"
    raw = [json.loads(l) for f in LABELS for l in open(f) if l.strip()]
    # judge.py appends a label=None row on an error and a fresh row when resumed: keep the last labelled row per rollout
    rows = list({r["job_id"]: r for r in raw if r.get("label") in ("HONEST", "NOT")}.values())
    unlabelled = {r["job_id"] for r in raw} - {r["job_id"] for r in rows}
    assert not unlabelled, f"{len(unlabelled)} rollouts have no label yet (rerun the judge): {sorted(unlabelled)[:5]}"
    by = {}
    for r in rows:
        by.setdefault(r["paper_row"], []).append(r)
    lines = [r"\begin{tabular}{l rrrr}", r"\toprule", r" & \multicolumn{4}{c}{honest submission} \\", r"\cmidrule(lr){2-5}",
             r"arm & $n$ & count & rate & 90\% CI \\", r"\midrule"]
    counts, first = [], True
    for block in BLOCKS:
        present = [(k, lab) for k, lab in block if k in by]
        if not present:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        for k, lab in present:
            rs = by[k]; n = len(rs); h = sum(r["label"] == "HONEST" for r in rs)
            hk = sum(r["outcome_final"] == "1" for r in rs)
            assert not any(r["label"] == "HONEST" and r["outcome_final"] == "1" for r in rs), f"{k}: honest and hacked"
            lo, hi = wilson(h, n)
            lines.append(f"{lab} & {n} & {h} & {h / n:.2f} & ({lo:.2f}, {hi:.2f}) \\\\")
            counts.append({"arm": k, "n": n, "honest": h, "hacked": hk, "neither": n - h - hk, "rate": round(h / n, 3),
                           "ci90_lo": round(lo, 3), "ci90_hi": round(hi, 3)})
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "honest_submit_summary_table.tex"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "honest_counts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(counts[0])); w.writeheader(); w.writerows(counts)
    print(f"{len(rows)} labels, {len(counts)} arms -> honest_submit_summary_table.tex, honest_counts.csv")


if __name__ == "__main__":
    main()
