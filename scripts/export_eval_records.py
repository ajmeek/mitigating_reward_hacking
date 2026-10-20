#!/usr/bin/env python
"""Export replay records (records.jsonl) from steering-arm Inspect logs.

Root env (inspect-ai). Each --logs entry is a local .eval file, a local directory, or an
s3:// prefix laid out as `.../<ARM>/<split>/s<seed>/`; arm, split and seed are parsed from the
path (override with --arm/--split/--seed) and the seed is asserted against the log. The arm's
steering (mode + dose) comes from its experiment's arms.tsv — the table the runner used —
found at experiments/<exp>/arms.tsv when the logs live under experiments/<exp>/data/logs/,
else given with --arms-tsv, else --mode/--dose. Pass the arm proxy's --ids-log sidecar(s) so
every turn carries its exact generated token ids and the steered spans. One JSON line per
rollout (a sample that was restarted into the same directory keeps only its scored record);
see steering_reasoning/evals/eval_records.py for the schema.

    uv run python scripts/export_eval_records.py \\
        --logs experiments/<exp>/data/logs/cn8/conflicting/s42/ --ids-log experiments/<exp>/data/logs/cn8/ids.jsonl \\
        --select lcbhard_4,lcbhard_33 --out records_cn8_s42.jsonl

With --ext m120 the cell's continued rollouts (`s<seed>/m120/*.eval`; only the authors' logs have
these, from a shorter first pass) are folded onto their base records. Runs made at 120 messages
directly need no --ext.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from steering_reasoning.evals.eval_records import (  # noqa: E402
    DEFAULT_MAX_TOKENS_EFFECTIVE,
    arm_steering,
    dedup_records,
    export_records,
    merge_extended,
    parse_arm_path,
)

EXP_RE = re.compile(r"^(?P<exp>.*?/experiments/[^/]+)/data/logs/")


def find_arms_tsv(spec: str) -> Path | None:
    m = EXP_RE.match(str(Path(spec).resolve()) if not spec.startswith("s3://") else spec)
    if not m:
        return None
    p = Path(m["exp"]) / "arms.tsv"
    return p if p.exists() else None


def resolve_logs(spec: str, cache_dir: Path) -> list[Path]:
    if spec.startswith("s3://"):
        rel = spec[5:].split("/", 1)[1].rstrip("/")
        local = cache_dir / rel
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(["aws", "s3", "sync", spec, str(local), "--exclude", "*", "--include", "*.eval",
                        "--quiet"], check=True)
        return sorted(local.glob("*.eval"))
    p = Path(spec)
    if p.is_dir():
        return sorted(p.glob("*.eval"))
    if p.suffix == ".eval" and p.exists():
        return [p]
    raise SystemExit(f"--logs {spec}: not a .eval file, directory, or s3:// prefix")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--select", default="", help="comma-separated task ids (default: all samples)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm"); ap.add_argument("--split"); ap.add_argument("--seed", type=int)
    ap.add_argument("--arms-tsv", help="the experiment's arms.tsv (default: experiments/<exp>/arms.tsv next to the logs)")
    ap.add_argument("--mode", choices=("none", "clamp", "add", "ablate"), help="the arm's steering mode, for logs without an arms.tsv")
    ap.add_argument("--dose", type=float, help="the arm's dose (clamp threshold / add scale), for logs without an arms.tsv")
    ap.add_argument("--max-tokens-effective", type=int, default=DEFAULT_MAX_TOKENS_EFFECTIVE,
                    help="the proxy's max_tokens cap at generation time (n5 arms: 16000)")
    ap.add_argument("--cache-dir", default=str(REPO / "data" / "scratch" / "eval-logs"))
    ap.add_argument("--ids-log", nargs="*", default=[],
                    help="the arm proxy's --ids-log sidecar file(s) (run_arm.sh: $E/data/logs/<ARM>/ids.jsonl; "
                         "fetch it yourself for an s3:// --logs, only *.eval are synced): joins exact generated "
                         "token ids, the server's prompt-ids hash and the steered spans onto each turn")
    ap.add_argument("--ext", metavar="m<limit>",
                    help="also read <logs>/<ext>/*.eval — the cell's rollouts continued under a larger message "
                         "limit (only the authors' logs have these) — and fold each onto its base record, so a record "
                         "is the whole rollout (base turns + continued turns; `resumed_from_messages` marks the "
                         "boundary, `extended` names the continuation)")
    args = ap.parse_args()
    if args.ext and not re.fullmatch(r"m\d+", args.ext):
        raise SystemExit(f"--ext must look like m120, got {args.ext!r}")

    select = [t for t in args.select.split(",") if t] or None
    records = []
    for spec in args.logs:
        try:
            meta = parse_arm_path(spec)
        except ValueError:
            if not (args.arm and args.split and args.seed is not None):
                raise SystemExit(f"--logs {spec}: path has no <arm>/<split>/s<seed>; pass --arm, "
                                 "--split and --seed")
            meta = parse_arm_path(f"/{args.arm}/{args.split}/s{args.seed}/")
        if args.arm: meta["arm"] = args.arm
        if args.split: meta["split"] = args.split
        if args.seed is not None: meta["seed"] = args.seed
        if args.dose is not None and args.mode is None:
            raise SystemExit("--dose needs --mode")
        if args.mode is not None:
            meta["mode"], meta["dose"] = (None if args.mode == "none" else args.mode), args.dose
        else:
            table = Path(args.arms_tsv) if args.arms_tsv else find_arms_tsv(spec)
            if table is None:
                raise SystemExit(f"--logs {spec}: no arms.tsv found for these logs; pass --arms-tsv or --mode/--dose")
            meta["mode"], meta["dose"], _ = arm_steering(table, meta["arm"], REPO)
        files = resolve_logs(spec, Path(args.cache_dir))
        kw = dict(arm=meta["arm"], mode=meta["mode"], dose=meta["dose"], seed=meta["seed"], split=meta["split"],
                  select=select, max_tokens_effective=args.max_tokens_effective, ids_logs=args.ids_log or None)
        raw = export_records(files, **kw)
        recs = dedup_records(raw)
        dropped = len(raw) - len(recs)
        note = f", {dropped} restarted duplicate(s) dropped" if dropped else ""
        if args.ext:
            ext_spec = spec.rstrip("/") + "/" + args.ext
            if ext_spec.startswith("s3://") or Path(ext_spec).is_dir():
                ext_files = resolve_logs(ext_spec, Path(args.cache_dir))
                ext_raw = export_records(ext_files, **kw)
                recs = merge_extended(recs, dedup_records(ext_raw), ext_name=args.ext)
                n_ext = sum(1 for r in recs if r["extended"])
                note += f"; {len(ext_files)} {args.ext} log(s) → {n_ext} rollout(s) extended"
            else:
                recs = merge_extended(recs, [], ext_name=args.ext)
                note += f"; no {args.ext}/ directory, nothing extended"
        else:
            recs = merge_extended(recs, [], ext_name="")   # same fields either way: extended=None
        joins = {}
        for r in recs:
            for t in r["turns"]:
                joins[t.get("ids_join", "none")] = joins.get(t.get("ids_join", "none"), 0) + 1
        print(f"{spec}: {len(files)} log(s) → {len(recs)} rollout(s) [{meta['arm']} {meta['mode']} dose={meta['dose']} "
              f"{meta['split']} s{meta['seed']}]{note}; ids join: {joins}")
        records.extend(recs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n_turns = sum(len(r["turns"]) for r in records)
    print(f"wrote {len(records)} records, {n_turns} turns → {out}")


if __name__ == "__main__":
    main()
