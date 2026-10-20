#!/usr/bin/env python
"""Package this experiment's rollouts for the steering-vector work: one replay record per rollout
(the whole transcript under the 120-message limit, exact generated token ids per turn) plus a
readable Markdown transcript of each, and an index that joins every rollout to its outcome labels.

    uv run python experiments/spec-prompting/scripts/export_records.py [--ext m120] [--no-transcripts]

Writes, under experiments/spec-prompting/data/ (gitignored; mirror to S3 with
`aws s3 sync` to derived/spec-prompting-v2/{records,transcripts}/, the released prefix):

  records/<arm>_<variant>_s<seed>.jsonl   replay records (steering_reasoning/evals/eval_records.py
                                          schema; `turns[].token_ids` from the proxy's ids sidecar;
                                          base turns + continued turns, boundary in
                                          `resumed_from_messages`)
  records/index.csv                       one row per rollout: ids, extension, limit, labels from
                                          artifacts/rollouts_<ext>.csv (hack, solve, unsat_*, cites,
                                          ...), turn counts, and how many turns are synthetic
  records/manifest.json                   per-cell counts, ids-join tallies, file sizes + sha256
  transcripts/<arm>/<variant>/s<seed>/<task>.md

A "synthetic" turn (usage 0/0, empty content, finish max_tokens) is one where the server refused
the request because the conversation had filled the 131k context window; Inspect substitutes an
empty reply and the harness nudges the model on, so a rollout that hits the window ends in a run
of empty turns until the message limit. Such rollouts are at_limit for a different reason than
"still working"; the index says how many turns each rollout lost that way.
"""

from __future__ import annotations

import argparse
import glob
import os
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

# EXPORT_EXP=experiments/<other> reuses this script for an experiment with the same layout
# (spec-prompting-basetasks: same arms, conflicting only; its tasks.json gives the task roles).
_here = Path(__file__).resolve().parents[1]
REPO = _here.parents[1]
EXP = (REPO / os.environ["EXPORT_EXP"]).resolve() if os.environ.get("EXPORT_EXP") else _here
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from steering_reasoning.evals.eval_records import (  # noqa: E402
    arm_steering,
    dedup_records,
    load_ids_log,
    merge_extended,
    sample_records,
)
from inspect_ai.log import read_eval_log  # noqa: E402

ARMS = ("base", "spec")
VARIANTS = tuple(os.environ.get("EXPORT_VARIANTS", "conflicting original").split())
SEEDS = (42, 43)
SPLITS = json.loads((REPO / "src/steering_reasoning/data/splits.json").read_text())
GROUP = {t: g for g, ts in SPLITS["test_groups"].items() for t in ts}
if (EXP / "tasks.json").exists():   # a non-test experiment: the group column carries the task's role
    GROUP = {**GROUP, **json.load(open(EXP / "tasks.json")).get("role", {})}
LABEL_COLS = ["outcome", "hack", "solve", "at_limit", "unsat_any", "unsat_reasoning", "unsat_final",
              "first_unsat_turn", "cites", "cites_reasoning", "n_submits", "n_tool_calls", "out_tokens",
              "honest_escalation", "wrong_unsat"]


_IDS: dict[str, dict] = {}


def ids_index(arm: str) -> dict:
    """The arm's sidecar, parsed once (46 MB for base) and shared by its four cells."""
    if arm not in _IDS:
        _IDS[arm] = load_ids_log([EXP / "data/logs" / arm / "ids.jsonl"])
    return _IDS[arm]


def _export(files: list[str], *, arm: str, variant: str, seed: int, mode, dose) -> list[dict]:
    out: list[dict] = []
    for f in files:
        log = read_eval_log(f, resolve_attachments=True)
        strict = bool((log.eval.model_args or {}).get("strict_tools", True))
        out.extend(sample_records(log, arm=arm, mode=mode, dose=dose, seed=seed, split=variant, eval_file=f,
                                  strict_tools=strict, ids_index=ids_index(arm)))
    return out


def cell_records(arm: str, variant: str, seed: int, ext: str | None) -> list[dict]:
    d = EXP / "data/logs" / arm / variant / f"s{seed}"
    mode, dose, _ = arm_steering(EXP / "arms.tsv", arm, REPO)
    kw = dict(arm=arm, variant=variant, seed=seed, mode=mode, dose=dose)
    base = dedup_records(_export(sorted(glob.glob(str(d / "*.eval"))), **kw))
    if not ext:
        return merge_extended(base, [], ext_name="")
    ext_files = sorted(glob.glob(str(d / ext / "*.eval")))
    ext_recs = dedup_records(_export(ext_files, **kw)) if ext_files else []
    return merge_extended(base, ext_recs, ext_name=ext)


def write_transcripts(recs: list[dict], out_dir: Path, max_chars: int) -> int:
    """One Markdown file per rollout, rendered from the log that holds its final state (the
    extension log when it was extended, else the base log)."""
    from render_eval_transcript import render  # scripts/render_eval_transcript.py

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in recs:
        base_log = Path(r["base_eval_file"]).name if r.get("base_eval_file") else "-"
        head = (f"<!-- {r['rollout_id']} | arm {r['arm']} | {r['split']} | seed {r['seed']} | "
                f"extended: {r.get('extended') or 'no'} (from {base_log}) | "
                f"limit: {(r['limit'] or {}).get('limit') if r['limit'] else 'none'} | "
                f"is_reward_hack={r['is_reward_hack']} solved={r['solved']} -->\n\n")
        md = render(r["eval_file"], max_chars, sample=r["task_id"])
        (out_dir / f"{r['task_id']}.md").write_text(head + md)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ext", default="m120", help="continuation directory to fold in ('' for the base limit only)")
    ap.add_argument("--out", default=str(EXP / "data"))
    ap.add_argument("--no-transcripts", action="store_true")
    ap.add_argument("--max-chars", type=int, default=20000, help="per-block clip in the Markdown transcripts")
    ap.add_argument("--cells", default="", help="comma-separated arm/variant/s<seed> to (re)do; default all")
    a = ap.parse_args()
    ext = a.ext or None
    out = Path(a.out)
    rec_dir, tr_dir = out / "records", out / "transcripts"
    rec_dir.mkdir(parents=True, exist_ok=True)

    labels = None
    lab_path = EXP / "artifacts" / (f"rollouts_{ext}.csv" if ext else "rollouts.csv")
    if not lab_path.exists() and ext and (EXP / "artifacts/rollouts.csv").exists():
        lab_path = EXP / "artifacts/rollouts.csv"   # basetasks: analyze.py --ext m120 wrote the 120 view here
    if lab_path.exists():
        labels = pd.read_csv(lab_path).set_index(["arm", "variant", "seed", "task"])
        # with a duplicate key .loc returns a frame and every label would silently vanish
        assert labels.index.is_unique, f"{lab_path} has duplicate (arm, variant, seed, task) rows"
    else:
        print(f"WARN no label table at {lab_path}; index.csv will carry no outcome labels")

    all_cells = {f"{arm}/{variant}/s{seed}" for arm in ARMS for variant in VARIANTS for seed in SEEDS}
    only = {c for c in a.cells.split(",") if c}
    if only - all_cells:
        raise SystemExit(f"--cells: unknown {sorted(only - all_cells)}; choose from {sorted(all_cells)}")
    manifest = {"ext": ext, "cells": {}}
    index_rows = []
    for arm in ARMS:
        for variant in VARIANTS:
            for seed in SEEDS:
                cell = f"{arm}/{variant}/s{seed}"
                if only and cell not in only:
                    continue
                recs = cell_records(arm, variant, seed, ext)
                recs.sort(key=lambda r: int(r["task_id"].split("_")[1]))
                path = rec_dir / f"{arm}_{variant}_s{seed}.jsonl"
                with path.open("w") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
                joins: dict[str, int] = {}
                n_syn = 0
                for r in recs:
                    for t in r["turns"]:
                        joins[t["ids_join"]] = joins.get(t["ids_join"], 0) + 1
                        n_syn += bool(t["synthetic_length"])
                    lkey = (arm, variant, seed, r["task_id"])
                    lab = None
                    if labels is not None:
                        if lkey in labels.index:
                            lab = labels.loc[lkey]
                        else:
                            print(f"WARN {r['rollout_id']}: no row in {lab_path.name}; labels left blank")
                    joins_r = {j: sum(t["ids_join"] == j for t in r["turns"])
                               for j in ("sidecar", "sidecar_ambiguous", "missing", "length_mismatch", "none")}
                    row = {"rollout_id": r["rollout_id"], "arm": arm, "variant": variant, "seed": seed,
                           "task": r["task_id"], "group": GROUP.get(r["task_id"], "?"),
                           "extended": r.get("extended") or "", "resumed_from_messages": r.get("resumed_from_messages"),
                           "limit": (r["limit"] or {}).get("limit") if r["limit"] else "",
                           "n_messages": len(r["messages"]), "n_turns": len(r["turns"]),
                           "n_synthetic_turns": sum(bool(t["synthetic_length"]) for t in r["turns"]),
                           "n_turns_ids_exact": joins_r["sidecar"],
                           "n_turns_ids_ambiguous": joins_r["sidecar_ambiguous"],
                           "n_turns_ids_missing": joins_r["missing"] + joins_r["length_mismatch"] + joins_r["none"],
                           "is_reward_hack": r["is_reward_hack"], "solved": r["solved"],
                           "records_file": path.name, "eval_file": Path(r["eval_file"]).name,
                           "base_eval_file": Path(r["base_eval_file"]).name if r.get("base_eval_file") else ""}
                    if lab is not None:
                        row.update({c: lab[c] for c in LABEL_COLS if c in lab.index})
                    index_rows.append(row)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest["cells"][cell] = {"file": path.name, "rollouts": len(recs),
                                           "extended": sum(1 for r in recs if r.get("extended")),
                                           "turns": sum(len(r["turns"]) for r in recs), "synthetic_turns": n_syn,
                                           "ids_join": joins, "bytes": path.stat().st_size, "sha256": sha}
                msg = f"{cell}: {len(recs)} rollouts, {manifest['cells'][cell]['turns']} turns, join {joins}, synthetic {n_syn}"
                if not a.no_transcripts:
                    n = write_transcripts(recs, tr_dir / arm / variant / f"s{seed}", a.max_chars)
                    msg += f", {n} transcripts"
                print(msg, flush=True)

    idx = pd.DataFrame(index_rows)
    if only and (rec_dir / "index.csv").exists() and (rec_dir / "manifest.json").exists():
        # partial redo: replace only the redone cells
        old = pd.read_csv(rec_dir / "index.csv")
        keep = old[~old.rollout_id.isin(set(idx.rollout_id))]
        idx = pd.concat([keep, idx], ignore_index=True)
        old_m = json.loads((rec_dir / "manifest.json").read_text())
        if old_m.get("ext") != ext:
            raise SystemExit(f"--cells redo under --ext {ext!r} but the existing manifest is for {old_m.get('ext')!r}; "
                             "rebuild everything")
        old_m["cells"].update(manifest["cells"])
        manifest = old_m
    idx = idx.sort_values(["arm", "variant", "seed", "task"]).reset_index(drop=True)
    idx.to_csv(rec_dir / "index.csv", index=False)
    manifest["rollouts"] = int(len(idx))
    manifest["turns"] = int(idx.n_turns.sum())
    manifest["turns_with_ids"] = int(idx.n_turns_ids_exact.sum() + idx.n_turns_ids_ambiguous.sum())
    manifest["turns_ids_ambiguous"] = int(idx.n_turns_ids_ambiguous.sum())
    manifest["synthetic_turns"] = int(idx.n_synthetic_turns.sum())
    manifest["rollouts_with_synthetic_turns"] = int((idx.n_synthetic_turns > 0).sum())
    manifest["labels_from"] = str(lab_path.relative_to(REPO)) if labels is not None else None
    (rec_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"index: {len(idx)} rollouts, {manifest['turns']} turns ({manifest['turns_with_ids']} with token ids, "
          f"{manifest['turns_ids_ambiguous']} of them ambiguous; "
          f"{manifest['synthetic_turns']} synthetic in {manifest['rollouts_with_synthetic_turns']} rollouts) -> {rec_dir}")


if __name__ == "__main__":
    main()
