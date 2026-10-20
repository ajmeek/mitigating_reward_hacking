"""Hack rate: unsteered, spec in the system prompt, spec + adversarial task-owner instruction in the user turn.
Bars with Wilson 90% intervals, paper style.

Sources: the main sweep's summary (experiments/diff-mean-steering/artifacts/summary_fleet.csv,
unsteered baseline), the spec-prompting per-rollout table (experiments/spec-prompting/artifacts/rollouts_m120.csv,
spec arm, 120 messages, seeds 42+43, n = 60) and the adversarial-instruction arm
(experiments/spec-adversarial-instruction, seed 42). The adversarial arm's counts come from its Inspect eval logs
(experiments/spec-adversarial-instruction/data/logs/, gitignored; on S3 under derived/spec-adversarial-instruction/)
and are cached in adversarial_counts.json beside this script, so the figure redraws without the logs.

    uv run python paper-artifacts/spec_adversarial/spec_adversarial.py [--from-logs]
"""
import csv, glob, json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(ROOT / "paper-artifacts"))
import paper_style; paper_style.use()
import matplotlib.pyplot as plt
def wilson(k, n, z=1.645):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)
# spec arm: the per-rollout release table (the same rows the results table's spec-prompt row is built from)
idx = list(csv.DictReader(open(ROOT / "experiments/spec-prompting/artifacts/rollouts_m120.csv")))
def rate(arm):
    r = [x for x in idx if x["arm"] == arm and x["variant"] == "conflicting"]; return sum(x["hack"] == "True" for x in r), len(r)
# adversarial arm: every scored sample across its log dirs, cached beside the script
LOGS = ROOT / "experiments/spec-adversarial-instruction/data/logs/advspec/conflicting/s42"
CACHE = HERE / "adversarial_counts.json"
if "--from-logs" in sys.argv or not CACHE.exists():
    from inspect_ai.log import read_eval_log_sample_summaries
    k = n = 0
    for f in sorted(glob.glob(str(LOGS / "**/*.eval"), recursive=True)):
        for s in read_eval_log_sample_summaries(f):
            if s.scores and not s.error:
                n += 1; k += any(str(v.value).upper().startswith("C") for v in s.scores.values())
    assert n, f"no scored samples under {LOGS}; pull the logs from S3 or drop --from-logs to use {CACHE.name}"
    CACHE.write_text(json.dumps({"hack": k, "n": n, "logs": str(LOGS.relative_to(ROOT))}, indent=1) + "\n")
c = json.loads(CACHE.read_text()); k, n = c["hack"], c["n"]
# Unsteered: the paper's baseline (main sweep, artifacts/summary_fleet.csv: 16/60). Spec: the 2026-09-19 release,
# marked interim until Austin's rerun for the table lands. Spec + adversarial: this run, marked interim until all 29 scorable
# rollouts finish (lcbhard_77 overflows the context and cannot run).
sw = {r["arm"]: r for r in csv.DictReader(open(ROOT / "experiments/diff-mean-steering/artifacts/summary_fleet.csv")) if r["variant"] == "conflicting"}
base_k, base_n = int(sw["base"]["count_ext"]), int(sw["base"]["n"])
bars = [("unsteered", base_k, base_n, paper_style.VERMILLION),
        ("spec prompt\n(interim)", *rate("spec"), paper_style.TEAL),
        ("spec prompt +\nadversarial instruction\n(interim)", k, n, paper_style.CATEGORICAL[1])]
fig, ax = plt.subplots(figsize=(3.6, 2.4))
for i, (lab, kk, nn, col) in enumerate(bars):
    p = kk / nn; lo, hi = wilson(kk, nn)
    ax.bar(i, p, width=0.62, color=col, alpha=0.8, edgecolor=paper_style.INK, linewidth=0.5, zorder=2)
    ax.vlines(i, lo, hi, color=paper_style.INK, lw=0.9, zorder=3)
ax.set_xticks(range(3)); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
ax.set_ylabel("hack rate"); ax.set_ylim(0, 0.45); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
fig.tight_layout()
print(paper_style.save(fig, "spec_adversarial"))
for lab, kk, nn, _ in bars: print(f"{lab.replace(chr(10),' ')}: {kk}/{nn} = {kk/nn:.2f}  90% CI {tuple(round(x,2) for x in wilson(kk,nn))}")
