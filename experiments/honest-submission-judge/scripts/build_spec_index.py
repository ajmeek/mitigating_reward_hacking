"""Build the judge's rollout index for the model-spec arms (data/spec_rollouts_index.csv).

The 16 main-sweep arms were judged from paper_rollouts_index.csv (not in the repo). This writes the same
columns for the seven spec arms, all at 120 messages, from the analysis outputs of record:

    experiments/spec-prompting/artifacts/rollouts_m120.csv         base, spec   (spec-prompt baseline and spec prompt)
    experiments/spec-vector/artifacts/results_120msg/rollouts.csv     v1floor5/10/16, v1add8/16

Only the conflicting variant (metric == hack) is listed. `full_transcript` is the rollout's final log,
repo-relative: the m120 continuation log when the authors' rollout was continued from a shorter first pass, else the base log. Run from the repo root:

    python experiments/honest-submission-judge/scripts/build_spec_index.py
    python experiments/honest-submission-judge/scripts/judge.py --index experiments/honest-submission-judge/data/spec_rollouts_index.csv \
        --data-root . --out experiments/honest-submission-judge/data/labels_spec.jsonl
"""
import csv, hashlib, os, sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
OUT = os.path.join(REPO, "experiments/honest-submission-judge/data/spec_rollouts_index.csv")
SOURCES = [  # (experiment, rollouts csv, {arm key: paper row})
    ("spec-prompting", "experiments/spec-prompting/artifacts/rollouts_m120.csv",
     {"base": "spec baseline", "spec": "spec prompt"}),
    ("spec-vector", "experiments/spec-vector/artifacts/results_120msg/rollouts.csv",
     {"v1floor5": "v_spec floor +5", "v1floor10": "v_spec floor +10", "v1floor16": "v_spec floor +16.5",
      "v1add8": "v_spec add +8", "v1add16": "v_spec add +16"}),
]
FIELDS = ["paper_row", "arm", "seed", "task", "metric", "run", "job_id", "full_transcript", "outcome_final", "at_limit_final"]


def main():
    rows, missing = [], []
    for exp, csv_path, arms in SOURCES:
        for r in csv.DictReader(open(os.path.join(REPO, csv_path))):
            if r["variant"] != "conflicting" or r["arm"] not in arms:
                continue
            assert r["scored"] == "True" and not r["error"], r
            sub = f"m120/{r['file']}" if r["ext"] == "True" else r["file"]
            path = f"experiments/{exp}/data/logs/{r['arm']}/conflicting/s{r['seed']}/{sub}"
            if not os.path.exists(os.path.join(REPO, path)):
                missing.append(path)
            rows.append({"paper_row": arms[r["arm"]], "arm": r["arm"], "seed": r["seed"], "task": r["task"],
                         "metric": "hack", "run": exp,
                         "job_id": hashlib.sha1(f"{exp}/{r['arm']}/s{r['seed']}/{r['task']}".encode()).hexdigest()[:20],
                         "full_transcript": path, "outcome_final": r["outcome"], "at_limit_final": r["at_limit"]})
    rows.sort(key=lambda r: (r["run"], r["arm"], int(r["seed"]), int(r["task"].split("_")[1])))
    assert len({r["job_id"] for r in rows}) == len(rows)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
    per = {}
    for r in rows:
        per[r["paper_row"]] = per.get(r["paper_row"], 0) + 1
    print(f"{len(rows)} rollouts -> {os.path.relpath(OUT, REPO)}", per)
    if missing:
        print(f"{len(missing)} transcripts not on disk (pull from S3 first):", *missing[:5], sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
