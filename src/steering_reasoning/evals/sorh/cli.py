"""Command line for the SoRH eval: `python -m steering_reasoning.evals.sorh <subcommand> --help`.

  fetch           download the dataset rows to a JSON file (+ .meta.json with the HF sha)
  dev-set         draw (or --check) the dose-sweep dev set -> ids.json
  eval-set        draw (or --check) the eval set from the rows and a dev ids.json -> ids.json
  arms            print the rows of an arms.tsv (name port upstream mode artifact dose)
  proxy-args      print the `python -m steering_reasoning.proxy` arguments for one arm of an arms.tsv
  check-polarity  refuse a vector/mode/dose that would steer the wrong way (one artifact, or every row of a table)
  check-delivery  prove each steering proxy changes the temperature-0 output vs the base proxy
  check-servers   prove two passthrough proxies (two servers) are byte-identical unsteered
  run             run the Inspect task for one arm against a proxy (resumable with --sample-ids / --ids-file)
  remaining       ids without a FINISHED sample under an arm's log dir (what the launcher resumes with)
  status          one TSV line per arm from a raw zip scan (no inspect_ai needed)
  grade           blind judge over <logs>/<arm>/ -> grades + labels.json (resumable; --mock, --estimate, --join)
  summarize       logs + labels -> per_item.csv, arm_summary.csv, summary.md
  select          pick the dose of record per vector from a sweep summary -> selection.md
  final           tables of record for an eval set -> summary.csv, summary.md

Paths are taken as given (relative to the cwd). Only `run`, `remaining` and `grade` import inspect_ai;
`grade` also needs the openai SDK unless --mock / --estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import steering_reasoning
from steering_reasoning.evals.config import GEN_DEFAULTS, MAX_TOKENS

from . import analyze, checks, dataset, grade as grade_mod, logs as logs_mod, task as task_mod
from .arms import ArmSpec, read_arms_tsv

REPO_ROOT = Path(steering_reasoning.__file__).resolve().parents[2]   # <repo>/src/steering_reasoning/__init__.py


def _root(a) -> str:
    """The directory relative artifact paths resolve under: --root if given, else the cwd when the
    table's artifacts exist there, else the repo root (arms tables are written repo-relative)."""
    if getattr(a, "root", None):
        return a.root
    table = getattr(a, "arms_tsv", None)
    if table:
        rel = [x.artifact for x in read_arms_tsv(table) if x.artifact and not Path(x.artifact).is_absolute()]
        if rel and not all(Path(p).exists() for p in rel) and all((REPO_ROOT / p).exists() for p in rel):
            return str(REPO_ROOT)
    return "."


def _ids_from(args) -> list[str]:
    ids = []
    if getattr(args, "ids_file", None):
        ids += Path(args.ids_file).read_text().split()
    if getattr(args, "ids", None):
        ids += [i for i in args.ids if i]
    if getattr(args, "sample_ids", None):
        ids += [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    return ids


def _arms_table(args) -> dict[str, ArmSpec] | None:
    t = getattr(args, "arms_tsv", None)
    if not t:
        return None
    return {a.name: a for a in read_arms_tsv(t)}


def _vector_order(table: dict[str, ArmSpec] | None) -> list[str] | None:
    if not table:
        return None
    return list(OrderedDict((a.vector, None) for a in table.values()))


def _arm_names(args, default_from_table: bool = True) -> list[str]:
    names = [a for a in (args.arms or "").replace(",", " ").split() if a] if isinstance(args.arms, str) else list(args.arms or [])
    if not names and default_from_table:
        table = _arms_table(args)
        if table:
            names = list(table)
    return names


# ----------------------------------------------------------------------------- dataset
def cmd_fetch(a) -> int:
    rows = dataset.fetch_rows(a.out, revision=a.revision)
    print(f"wrote {a.out}: {len(rows)} rows")
    return 0


def _write_or_check(text: str, out: str, check: bool, label: str) -> int:
    if check:
        on_disk = Path(out).read_text(encoding="utf-8") if Path(out).exists() else None
        same = on_disk is not None and on_disk.rstrip("\n") == text.rstrip("\n")   # a missing final newline is not a difference
        print(f"{out}: {'MATCHES' if same else 'DIFFERS FROM'} a fresh draw ({label})")
        return 0 if same else 1
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    print(f"wrote {out} ({label})")
    return 0


def cmd_dev_set(a) -> int:
    rows = dataset.load_rows(a.rows, strict=not a.no_strict)
    dev = dataset.draw_dev_set(rows, seed=a.seed, n_writing=a.n_writing, n_coding=a.n_coding)
    return _write_or_check(json.dumps(dev, indent=1) + "\n", a.out, a.check,
                           f"{len(dev['ids'])} ids: {len(dev['writing_ids'])} writing + {len(dev['coding_ids'])} coding, seed {a.seed}")


def cmd_eval_set(a) -> int:
    rows = dataset.load_rows(a.rows, strict=not a.no_strict)
    dev = dataset.load_ids(a.dev)
    ev = dataset.draw_eval_set(rows, dev, seed=a.seed, n_writing=a.n_writing)
    return _write_or_check(json.dumps(ev, indent=1) + "\n", a.out, a.check,
                           f"{ev['n']} ids: {len(ev['coding_ids'])} coding + {len(ev['writing_ids'])} writing, "
                           f"{len(ev['heldout_ids'])} held out, seed {a.seed}")


# ----------------------------------------------------------------------------- arms
def cmd_arms(a) -> int:
    for arm in read_arms_tsv(a.arms_tsv):
        print("\t".join([arm.name, str(arm.port or "-"), arm.upstream or "-", arm.mode, arm.artifact or "-",
                         "-" if arm.dose is None else str(arm.dose)]))
    return 0


def cmd_proxy_args(a) -> int:
    table = {x.name: x for x in read_arms_tsv(a.arms_tsv)}
    if a.arm not in table:
        print(f"arm {a.arm!r} not in {a.arms_tsv} (has {sorted(table)})", file=sys.stderr)
        return 2
    print(" ".join(table[a.arm].proxy_args(cap=a.cap, cache_salt=not a.no_cache_salt, root=_root(a))))
    return 0


# ----------------------------------------------------------------------------- checks
def cmd_check_polarity(a) -> int:
    targets = []
    only = set(_arm_names(a, default_from_table=False))
    if a.arms_tsv:
        root = _root(a)
        for arm in read_arms_tsv(a.arms_tsv):
            if arm.steered and arm.mode != "ablate" and (not only or arm.name in only):
                targets.append((arm.name, arm.resolved(root).artifact, arm.mode, arm.dose))
    if a.artifact:
        if not a.mode:
            print("--artifact needs --mode (and --dose for add/clamp)", file=sys.stderr)
            return 2
        if a.mode in ("add", "clamp") and a.dose is None:
            print(f"--mode {a.mode} needs --dose", file=sys.stderr)
            return 2
        targets.append((Path(a.artifact).stem, a.artifact, a.mode, a.dose))
    if not targets:
        print("nothing to check: give --artifact/--mode/--dose or --arms-tsv (with steered rows)", file=sys.stderr)
        return 2
    rc = 0
    for name, artifact, mode, dose in targets:
        try:
            meta = checks.load_sidecar(artifact)
            warns = checks.check_polarity(meta, mode, dose)
        except checks.PolarityError as e:
            print(f"REFUSED {name}: {e}")
            rc = 1
            continue
        except (OSError, ValueError) as e:
            print(f"REFUSED {name}: cannot read sidecar: {e}")
            rc = 1
            continue
        for w in warns:
            print(f"WARNING {name}: {w}")
        print(f"POLARITY OK: {name} mode={mode} dose={dose} (sign: {checks.polarity(meta) or 'unknown'})", flush=True)
    return rc


def _name_ports(a, keep) -> list[tuple[str, int]]:
    """name:port pairs from --arms, or from the table's rows that `keep(spec)` accepts (restricted to
    --arms names when both are given)."""
    if a.arms_tsv:
        table = read_arms_tsv(a.arms_tsv)
        only = set(a.arms or [])
        return [(x.name, x.port) for x in table if (not only or x.name in only) and x.port and keep(x)]
    return checks.parse_arm_ports(a.arms or [])


def cmd_check_delivery(a) -> int:
    # base plus every steered arm; a second passthrough row (base2) would fail base != X by design
    pairs = _name_ports(a, lambda x: x.mode != "none" or x.name == a.base)
    ok = checks.check_delivery(pairs, model=a.model, max_tokens=a.max_tokens, base=a.base)
    return 0 if ok else 1


def cmd_check_servers(a) -> int:
    user = checks.FALLBACK_USER
    if a.rows and a.row >= 0:
        rows = dataset.load_rows(a.rows, strict=False)
        if 0 <= a.row < len(rows) and rows[a.row].get("user"):
            user = rows[a.row]["user"]
    pairs = _name_ports(a, lambda x: x.mode == "none")     # only passthrough proxies can be byte-identical
    if len(pairs) < 2:
        print("check-servers needs at least two passthrough (mode none) proxies", file=sys.stderr)
        return 2
    ok = checks.check_servers(pairs, user=user, model=a.model, max_tokens=a.max_tokens)
    return 0 if ok else 1


# ----------------------------------------------------------------------------- run / progress
def cmd_run(a) -> int:
    ids = _ids_from(a) or None
    logs = task_mod.run_eval(a.rows, a.log_dir, base_url=a.base_url, model=a.model, sample_ids=ids, limit=a.limit,
                             max_connections=a.max_connections, temperature=a.temperature, top_p=a.top_p, seed=a.seed,
                             max_tokens=a.max_tokens, fail_on_error=a.fail_on_error, retry_on_error=a.retry_on_error,
                             log_buffer=a.log_buffer, max_retries=a.max_retries, strict_rows=not a.no_strict)
    task_mod.print_summary(logs)
    return 0


def cmd_remaining(a) -> int:
    ids = _ids_from(a)
    print(" ".join(logs_mod.remaining_ids(a.log_dir, ids)))
    return 0


def cmd_status(a) -> int:
    names = _arm_names(a) or [n for n, *_ in analyze.resolve_arms(a.logs)]
    if not names:
        print("no arms: pass --arms or --arms-tsv, or a logs dir holding <arm>/*.eval", file=sys.stderr)
        return 2
    for arm in names:
        r = logs_mod.scan_dir(os.path.join(a.logs, arm))
        print(f"{arm}\t{r['finished']}\t{r['errored_only']}\t{r['unreadable']}\t{r['files']}\t{r['cap_hits']}")
    return 0


# ----------------------------------------------------------------------------- grade
def cmd_grade(a) -> int:
    out_dir = a.out_dir
    labels_out = a.labels_out or os.path.join(out_dir, "labels.json")
    if a.join:
        grade_mod.join_labels(out_dir, labels_out, a.rel_to)
        return 0
    arms = _arm_names(a)
    if not arms:
        print("no arms: pass --arms a,b,... or --arms-tsv", file=sys.stderr)
        return 2
    if not a.rows:
        print("grade needs --rows (unless --join)", file=sys.stderr)
        return 2
    rows = dataset.load_rows(a.rows, strict=not a.no_strict)
    prices = None
    if a.prices:
        try:
            i, c, o = (float(x) for x in a.prices.split(","))
        except ValueError:
            print("--prices takes three numbers: input,cached_input,output USD per million tokens", file=sys.stderr)
            return 2
        prices = dict(input=i, cached_input=c, output=o)
    cfg = grade_mod.JudgeConfig(model=a.model, effort=a.effort, provider=a.provider, region=a.region, prices=prices,
                                max_completion_tokens=a.max_completion_tokens,
                                timeout=a.timeout, max_retries=a.max_retries, backoff_base=a.backoff_base,
                                backoff_cap=a.backoff_cap, concurrency=a.concurrency, report_every=a.report_every, mock=a.mock)
    return grade_mod.grade(rows, a.logs, arms, out_dir, labels_out, cfg=cfg, salt=a.salt, seed=a.seed, limit=a.limit,
                           estimate_only=a.estimate, force=a.force, key_file=a.key_file, dump_requests=a.dump_requests,
                           rel_to=a.rel_to, rows_label=a.rows, logs_label=a.logs)


# ----------------------------------------------------------------------------- analysis
def cmd_summarize(a) -> int:
    rows = dataset.load_rows(a.rows, strict=not a.no_strict)
    table = _arms_table(a)
    names = _arm_names(a, default_from_table=False) or None
    res = analyze.build_summary(a.logs, a.labels, rows, names, a.base, cap=a.cap, arms=table)
    paths = analyze.write_summary(res, a.out, logs_label=a.logs, labels_label=a.labels or "", title=a.title)
    for r in res["summary"]:
        if r["subset"] == "all":
            print({k: v for k, v in r.items() if v != ""})
    print("wrote " + ", ".join(paths.values()))
    return 0


def cmd_select(a) -> int:
    warnings: list[str] = []
    table = _arms_table(a)
    per_item = None
    if a.summary and os.path.exists(a.summary):
        source = a.summary
        try:
            cells = analyze.load_summary_csv(a.summary, table)
        except analyze.SummaryFormatError as e:
            print(f"ERROR {e}", file=sys.stderr)
            return 4
        pi = a.per_item or (a.summary[:-len("arm_summary.csv")] + "per_item.csv" if a.summary.endswith("arm_summary.csv") else None)
        if pi and os.path.exists(pi):
            per_item = analyze.read_per_item(pi)
    elif a.logs and os.path.isdir(a.logs):
        source = f"{a.logs} + {a.labels}"
        if not a.rows:
            print("select --logs needs --rows", file=sys.stderr)
            return 2
        if not (a.labels and os.path.exists(a.labels)):
            warnings.append(f"labels file {a.labels} not found: nothing is graded yet")
        rows = dataset.load_rows(a.rows, strict=not a.no_strict)
        names = _arm_names(a, default_from_table=False) or None
        res = analyze.build_summary(a.logs, a.labels, rows, names, a.base, cap=a.cap, arms=table, verbose=False)
        cells = [analyze.norm_row(r, table) for r in res["summary"] if r["subset"] == "all"]
        per_item = res["per_item"]
        print(f"built the sweep summary in memory from {source} ({len(cells)} cells)")
    else:
        print("No sweep to select from: give --summary (an arm_summary.csv from `summarize`) or --logs/--labels/--rows.",
              file=sys.stderr)
        return 2
    if not cells:
        print(f"ERROR {source}: no cell rows (subset == all)", file=sys.stderr)
        return 2
    if not any(c["n_graded"] for c in cells):
        print(f"No graded rows in {source}: the sweep is generated but not graded. Nothing written (exit 3).")
        return 3
    if a.common_rows:
        if not per_item:
            print("ERROR --common-rows needs the per_item CSV next to the summary (or --logs)", file=sys.stderr)
            return 2
        cells = analyze.common_rows_summary(per_item, [c["cell"] for c in cells], a.base, warnings, table)
    selection, curves, base_row = analyze.select(cells, a.base, a.expect_n, a.rule, a.degen_cap, warnings, _vector_order(table))
    md = analyze.render_selection(selection, curves, base_row, source, warnings, a.rule, a.degen_cap, a.common_rows, a.title)
    print(md)
    written = []
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(selection, indent=1))
        written.append(a.out_json)
    Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_md).write_text(md)
    written.append(a.out_md)
    print("wrote " + " and ".join(written))
    return 0


def cmd_final(a) -> int:
    per_item = analyze.read_per_item(a.per_item)
    ids = dataset.load_ids(a.ids)
    arms = _arm_names(a) or None          # --arms, else the arms table's order, else base then first-seen
    out, subsets = analyze.final_tables(per_item, ids, arms, a.base)
    if not out:
        print("no graded rows for the ids in the set", file=sys.stderr)
        return 1
    os.makedirs(a.out_dir, exist_ok=True)
    analyze.write_csv(os.path.join(a.out_dir, "summary.csv"), out)
    Path(os.path.join(a.out_dir, "summary.md")).write_text(analyze.render_final(out, subsets, a.title))
    analyze.print_final(out, subsets)
    print(f"wrote {os.path.join(a.out_dir, 'summary.csv')} and {os.path.join(a.out_dir, 'summary.md')}")
    return 0


# ----------------------------------------------------------------------------- parser
def _add_rows(p, required=True):
    p.add_argument("--rows", required=required, help="the dataset rows JSON (sorh fetch)")
    p.add_argument("--no-strict", action="store_true", help="accept a rows file that is not the 1,073-row dump")


def _add_arms(p, help_="comma or space separated arm names"):
    p.add_argument("--arms", default="", help=help_)
    p.add_argument("--arms-tsv", default=None, help="arms table (name mode dose port [artifact [upstream]])")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m steering_reasoning.evals.sorh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download the dataset rows")
    p.set_defaults(fn=cmd_fetch)
    p.add_argument("--out", required=True)
    p.add_argument("--revision", default=None, help="HF commit sha to pin")

    p = sub.add_parser("dev-set", help="draw the dose-sweep dev set")
    p.set_defaults(fn=cmd_dev_set)
    _add_rows(p)
    p.add_argument("--out", required=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--seed", type=int, default=dataset.DEV_SEED)
    p.add_argument("--n-writing", type=int, default=dataset.DEV_N_WRITING)
    p.add_argument("--n-coding", type=int, default=dataset.DEV_N_CODING)

    p = sub.add_parser("eval-set", help="draw the eval set (all coding + dev writing + a sample of the rest)")
    p.set_defaults(fn=cmd_eval_set)
    _add_rows(p)
    p.add_argument("--dev", required=True, help="the dev ids.json")
    p.add_argument("--out", required=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--seed", type=int, default=dataset.EVAL_SEED)
    p.add_argument("--n-writing", type=int, default=dataset.EVAL_N_WRITING)

    p = sub.add_parser("arms", help="print an arms table")
    p.set_defaults(fn=cmd_arms)
    p.add_argument("--arms-tsv", required=True)

    p = sub.add_parser("proxy-args", help="proxy arguments for one arm")
    p.set_defaults(fn=cmd_proxy_args)
    p.add_argument("--arms-tsv", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--cap", type=int, default=MAX_TOKENS)
    p.add_argument("--root", default=None, help="resolve relative artifacts under this dir (default: cwd, else the repo root)")
    p.add_argument("--no-cache-salt", action="store_true")

    p = sub.add_parser("check-polarity", help="vector/mode/dose sanity")
    p.set_defaults(fn=cmd_check_polarity)
    p.add_argument("--artifact")
    p.add_argument("--mode", choices=["add", "clamp", "ablate"])
    p.add_argument("--dose", type=float)
    p.add_argument("--arms-tsv")
    p.add_argument("--arms", default="", help="only these rows of the table")
    p.add_argument("--root", default=None, help="resolve relative artifacts under this dir (default: cwd, else the repo root)")

    p = sub.add_parser("check-delivery", help="base==base, base!=steered")
    p.set_defaults(fn=cmd_check_delivery)
    p.add_argument("--arms", nargs="*", help="name:port ... (or arm names with --arms-tsv)")
    p.add_argument("--arms-tsv")
    p.add_argument("--model", default=checks.DEFAULT_SERVED_MODEL)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--base", default="base")

    p = sub.add_parser("check-servers", help="two passthrough proxies agree")
    p.set_defaults(fn=cmd_check_servers)
    p.add_argument("--arms", nargs="*")
    p.add_argument("--arms-tsv")
    p.add_argument("--model", default=checks.DEFAULT_SERVED_MODEL)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--rows", default=None)
    p.add_argument("--row", type=int, default=0, help="dataset row for the probe (-1 = built-in)")

    p = sub.add_parser("run", help="run the task for one arm")
    p.set_defaults(fn=cmd_run)
    _add_rows(p)
    p.add_argument("--log-dir", required=True, help="the arm's log directory <logs>/<arm>/")
    p.add_argument("--base-url", default=task_mod.DEFAULT_BASE_URL)
    p.add_argument("--model", default=task_mod.DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample-ids", default=None, help="comma-separated ids")
    p.add_argument("--ids-file", default=None, help="whitespace-separated ids")
    p.add_argument("--max-connections", type=int, default=10)
    p.add_argument("--temperature", type=float, default=GEN_DEFAULTS["temperature"])
    p.add_argument("--top-p", type=float, default=GEN_DEFAULTS["top_p"])
    p.add_argument("--seed", type=int, default=GEN_DEFAULTS["seed"])
    p.add_argument("--max-tokens", type=int, default=None, help="unset: the proxy stamps its cap")
    p.add_argument("--fail-on-error", type=float, default=0.0)
    p.add_argument("--retry-on-error", type=int, default=2)
    p.add_argument("--log-buffer", type=int, default=5)
    p.add_argument("--max-retries", type=int, default=30, help="Inspect's per-request retry bound (a dead proxy fails the sample instead of retrying forever)")

    p = sub.add_parser("remaining", help="ids without a finished sample")
    p.set_defaults(fn=cmd_remaining)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--ids-file")
    p.add_argument("--ids", nargs="*")

    p = sub.add_parser("status", help="per-arm progress TSV")
    p.set_defaults(fn=cmd_status)
    p.add_argument("--logs", required=True)
    _add_arms(p)

    p = sub.add_parser("grade", help="blind judge")
    p.set_defaults(fn=cmd_grade)
    p.add_argument("--logs", required=True)
    _add_rows(p, required=False)
    _add_arms(p)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--labels-out", default=None)
    p.add_argument("--provider", choices=list(grade_mod.PROVIDERS), default=grade_mod.DEFAULT_PROVIDER,
                   help="openai = any OpenAI-compatible chat-completions endpoint (JUDGE_BASE_URL), bedrock = Amazon Bedrock's "
                        "Converse API (the paper's SoRH grades: GLM 5 as zai.glm-5); default JUDGE_PROVIDER from .env. Other endpoints may "
                        "need --effort none (reasoning_effort is OpenAI-only) and must support JSON-schema response_format")
    p.add_argument("--model", default=None, help="default: JUDGE_MODEL from .env, else zai.glm-5 (bedrock) / gpt-5.5 (openai)")
    p.add_argument("--region", default=grade_mod.DEFAULT_REGION, help="bedrock region (JUDGE_REGION; the API key is scoped to one)")
    p.add_argument("--prices", default=None, help="input,cached_input,output USD per million tokens (default: gpt-5.5 list prices "
                                                 "for openai/gpt-5.5; otherwise unpriced, cost.json says priced=false)")
    p.add_argument("--effort", default=grade_mod.DEFAULT_EFFORT, help="OpenAI reasoning_effort ('none' to omit the parameter for other endpoints; ignored by bedrock)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--salt", default=grade_mod.DEFAULT_SALT)
    p.add_argument("--max-completion-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--backoff-base", type=float, default=2.0)
    p.add_argument("--backoff-cap", type=float, default=60.0)
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--dump-requests", type=int, default=3)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--join", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--key-file", default=None, help="a file with JUDGE_API_KEY=... (default: the repo-root .env; the environment variable wins)")
    p.add_argument("--rel-to", default=None, help="record log paths relative to this dir")

    p = sub.add_parser("summarize", help="logs + labels -> tables")
    p.set_defaults(fn=cmd_summarize)
    p.add_argument("--logs", required=True)
    _add_rows(p)
    p.add_argument("--labels", default=None)
    p.add_argument("--out", required=True)
    _add_arms(p, "report order; default: base then every log dir")
    p.add_argument("--base", default="base")
    p.add_argument("--cap", type=int, default=analyze.DEFAULT_CAP,
                                                             help="output_tokens at or above this = cap hit (runs of record: 16000)")
    p.add_argument("--title", default="SoRH summary")

    p = sub.add_parser("select", help="dose of record per vector")
    p.set_defaults(fn=cmd_select)
    p.add_argument("--summary", default=None, help="arm_summary.csv from summarize")
    p.add_argument("--per-item", default=None)
    p.add_argument("--logs", default=None)
    p.add_argument("--labels", default=None)
    _add_rows(p, required=False)
    _add_arms(p)
    p.add_argument("--base", default="base")
    p.add_argument("--cap", type=int, default=analyze.DEFAULT_CAP)
    p.add_argument("--expect-n", type=int, default=None)
    p.add_argument("--common-rows", action="store_true")
    p.add_argument("--rule", choices=list(analyze.RULES), default="hc")
    p.add_argument("--degen-cap", type=float, default=analyze.DEFAULT_DEGEN_CAP)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", default=None)
    p.add_argument("--title", default="SoRH dose sweep: dose selection")

    p = sub.add_parser("final", help="tables of record for an eval set")
    p.set_defaults(fn=cmd_final)
    p.add_argument("--per-item", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--out-dir", required=True)
    _add_arms(p, "report order; default: the arms table's order, else base then first-seen")
    p.add_argument("--base", default="base")
    p.add_argument("--title", default="Final tables")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, OSError) as e:      # bad table, unreadable file, wrong rows file: a one-line error, exit 2
        print(f"error: {e}", file=sys.stderr)
        return 2
