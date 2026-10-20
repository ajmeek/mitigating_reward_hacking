"""Regenerate every figure and table the paper includes and collect them under the paper's file names.

    uv run python paper-artifacts/make_paper_artifacts.py                 # everything -> paper-artifacts/out/
    uv run python paper-artifacts/make_paper_artifacts.py --list          # what would run, and what each needs
    uv run python paper-artifacts/make_paper_artifacts.py --only fig4_oneplane_bands.pdf --only clamp_summary_table.tex
    uv run python paper-artifacts/make_paper_artifacts.py --paper-repo ../paper   # copy into <paper>/artifacts/ as well
    uv run python paper-artifacts/make_paper_artifacts.py --check --paper-repo ../paper   # compare, copy nothing

Each STEP runs one generator script from its own directory with the same interpreter as this one, then
copies the files it wrote into the output directory under the names the paper's LaTeX \\input{}s or
\\includegraphics{}es (artifacts/<name> in the paper repo). Two prep steps rebuild derived inputs
(sorh_all_arms.csv, honest_counts.csv) and run only when a step that reads them (`feeds`) is wanted.

Inputs: everything a step reads is tracked under experiments/*/artifacts/, experiments/*/data/ (gitignore
exceptions) or paper-artifacts/, except the paths listed in a step's `needs`, which are gitignored data
(pulled from S3 with the command the step names), and the Python modules in `needs_modules` that the
default `uv sync` does not install. A step whose `needs` are absent is skipped with that message;
everything else still runs. One item in the paper has no generator in this repo (NO_GENERATOR below)
and one is a hand-written tracked file (COPY_ONLY); the summary lists them so nothing is silently missing.

Generators write their outputs beside themselves (tracked, so the working tree changes on every run,
--check included). --check compares the regenerated .tex files with the paper repo's copies (byte-for-byte
after the leading `%` comment lines, which the paper's copies carry as provenance) and reports which PDFs
were regenerated (PDFs embed a creation date, so they are not compared). A copy into the paper repo keeps
the provenance comment block the paper's copy already has.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EJ = "experiments/hack-action-anchor/data/edit_judgments.json"   # tracked (gitignore exception), listed for clarity

# One entry per generator run. `script` is relative to this directory unless it starts with "experiments/".
# `outputs` maps the file the script writes (relative to the script's directory) to the paper's file name.
# A prep step has no outputs and `feeds`: the paper files whose steps read what it writes.
STEPS = [
    dict(script="sorh_specvec/sorh_data.py", outputs={}, feeds=["sorh_oneplane_bands.pdf", "sorh_summary_table.tex"],
         note="rebuilds sorh_all_arms.csv from the two SoRH experiments"),
    dict(script="honest_submit/make_honest_table.py", outputs={}, feeds=["fig4_oneplane_bands.pdf", "clamp_summary_table.tex"],
         note="rebuilds honest_counts.csv from the honest-submission judge labels"),
    dict(script="dose_response/dose_response_rows.py", outputs={"fig4_oneplane_bands.pdf": "fig4_oneplane_bands.pdf"},
         note="main dose-response figure: honest-submission, hack and solve rates per arm with 90% Wilson bands (--all draws five more variants)"),
    dict(script="sorh_specvec/sorh_rows.py", outputs={"sorh_oneplane_bands.pdf": "sorh_oneplane_bands.pdf"},
         note="School of Reward Hacks figure in the same layout (--all draws five more variants)"),
    dict(script="sorh_specvec/make_sorh_table.py", outputs={"sorh_summary_table.tex": "sorh_summary_table.tex"}),
    dict(script="results_table/make_results_table.py", outputs={"clamp_summary_table.tex": "clamp_summary_table.tex"}),
    dict(script="story_lie_tables/story_lie_bars.py", outputs={"story_lie_bars.pdf": "story_lie_bars.pdf"}),
    dict(script="story_lie_tables/story_by_story_ci.py", outputs={"story_by_story_ci.tex": "story_by_story_ci.tex"},
         note="per-story lie / honesty table with 90% Wilson intervals, rate-tinted cells (needs colortbl) and a Spearman rho row; also writes story_spearman.json for the Figure 6 caption"),
    dict(script="story_lie_tables/story_prompts.py",
         outputs={f"story_prompt_{k}.tex": f"story_prompt_{k}.tex" for k in ("resume", "package", "late", "rye", "dent", "reddot")}),
    dict(script="action_plane/segment_scatter.py", outputs={"segment_scatter.pdf": "segment_scatter.pdf"}),
    dict(script="action_plane/action_plane_ewma.py", outputs={"action_plane_ewma.pdf": "action_plane_turn_paths.pdf"}, needs=[EJ]),
    dict(script="cap_vs_floor/cap_vs_floor.py", outputs={"cap_vs_floor.pdf": "cap_vs_floor.pdf"}),
    dict(script="probe_auc_by_layer/probe_auc_by_layer.py", outputs={"probe_auc_by_layer.pdf": "probe_auc_by_layer.pdf"}),
    dict(script="jlens_pole_table/jlens_pole_table.py", outputs={"jlens_pole_table.tex": "jlens_pole_table.tex"}),
    dict(script="jlens_main_table/jlens_main_table.py", outputs={"jlens_main_table_2block.tex": "jlens_main_table_2block.tex"},
         note="formats the tracked jlens_main_readouts.json; jlens_main_readouts.py recomputes it (torch + transformers, 3.3 GB lens)"),
    dict(script="prehack_compact/prehack_compact.py",
         outputs={"prehack_ramps.pdf": "prehack_ramps.pdf", "prehack_window_auc.pdf": "prehack_window_auc.pdf"}, needs=[EJ]),
    dict(script="spec_adversarial/spec_adversarial.py", outputs={"spec_adversarial.pdf": "spec_adversarial.pdf"},
         note="adversarial-arm counts from adversarial_counts.json (cached from the eval logs; --from-logs recomputes)"),
    dict(script="transcript_u_trace/transcript_u_trace.py", outputs={"transcript_u_trace.pdf": "transcript_u_trace.pdf"},
         needs=["data/captures/qwen35_27b_bf16_token_L35/lcbhard_22.safetensors",
                "data/derived/token_series.parquet", EJ],
         needs_modules=["torch"],
         pull="the layer-35 capture of all 42 rollouts (scripts/pull_bf16_token_L35.py --src <your step-14 dump>) and "
              "scripts/extract_token_series.py for data/derived/token_series.parquet; the bf16 capture needs torch: run the driver as "
              "`uv run --with torch python paper-artifacts/make_paper_artifacts.py --only transcript_u_trace.pdf`"),
    dict(script="experiments/intervention-magnitude/scripts/final_figure.py", args=["--x", "measured", "--from-table"],
         env={"PAPER_STYLE": "1"},
         outputs={"../artifacts/off_manifold_final_measured.pdf": "off_manifold_final_measured.pdf"},
         note="redrawn from the tracked off_manifold_table_measured.csv; drop --from-table to recompute from the capture statistics"),
]
COPY_ONLY = {"story_lie_tables/story_prompts_blurbs.tex": "story_prompts_blurbs.tex"}   # hand-written, tracked here
NO_GENERATOR = {
    "fig1_floor.pdf": "Figure 1, drawn by hand (fig1_floor.svg in the paper repo)",
}
ALL_NAMES = {n for st in STEPS for n in st["outputs"].values()} | set(COPY_ONLY.values()) | set(NO_GENERATOR)


def step_paths(st):
    s = st["script"]
    script = (ROOT / s) if s.startswith("experiments/") else (HERE / s)
    return script, script.parent


def wanted(st, only):
    """Run this step? Without --only every step runs; a prep step runs when one of the files it feeds is wanted."""
    if not only: return True
    names = st["outputs"].values() if st["outputs"] else st.get("feeds", [])
    return any(n in only for n in names)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(HERE / "out"), help="where the paper-named copies go (default paper-artifacts/out/)")
    ap.add_argument("--paper-repo", help="paper repo checkout; copies also go to <paper-repo>/artifacts/ (or are compared with --check)")
    ap.add_argument("--only", action="append", default=[], help="paper file name(s) to build; repeatable")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    ap.add_argument("--check", action="store_true", help="regenerate and compare with --paper-repo, copy nothing")
    ap.add_argument("--keep-going", action="store_true", help="continue past a failing step")
    A = ap.parse_args()
    unknown = sorted(set(A.only) - ALL_NAMES)
    if unknown:
        ap.error(f"unknown --only name(s) {unknown}; known: {sorted(ALL_NAMES)}")
    if A.check and not A.paper_repo:
        ap.error("--check needs --paper-repo (there is nothing else to compare with)")
    paper_art = Path(A.paper_repo).expanduser().resolve() / "artifacts" if A.paper_repo else None
    if paper_art is not None and not paper_art.is_dir():
        ap.error(f"{paper_art} is not a directory")
    out = Path(A.out).resolve()

    if A.list:
        for st in STEPS:
            print(f"{st['script']}" + (" " + " ".join(st.get("args", [])) if st.get("args") else ""))
            for src, dst in st["outputs"].items(): print(f"    -> {dst}" + (f"  (from {src})" if Path(src).name != dst else ""))
            if st.get("feeds"): print(f"    prep for {', '.join(st['feeds'])}")
            for n in st.get("needs", []): print(f"    needs {n}" + ("" if (ROOT / n).exists() else "   [missing]"))
            for m in st.get("needs_modules", []): print(f"    needs module {m}" + ("" if has_module(m) else "   [missing]"))
            if st.get("note"): print(f"    {st['note']}")
        for src, dst in COPY_ONLY.items(): print(f"copy {src} -> {dst}")
        for dst, why in NO_GENERATOR.items(): print(f"no generator: {dst}  ({why})")
        return 0

    if not A.check: out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), MPLBACKEND="Agg")
    report, failed = [], False
    for st in STEPS:
        if not wanted(st, A.only): continue
        script, cwd = step_paths(st)
        rows_for = list(st["outputs"].values()) or [f"({st['script']})"]   # a prep step reports under its script name
        missing = [n for n in st.get("needs", []) if not (ROOT / n).exists()]
        missing += [f"module {m}" for m in st.get("needs_modules", []) if not has_module(m)]
        if missing:
            for dst in rows_for:
                report.append((dst, "SKIPPED", f"missing {', '.join(missing)}" + (f"; pull: {st['pull']}" if st.get("pull") else "")))
            continue
        cmd = [sys.executable, str(script), *st.get("args", [])]
        print(f"\n### {st['script']} {' '.join(st.get('args', []))}", flush=True)
        r = subprocess.run(cmd, cwd=cwd, env={**env, **st.get("env", {})}, text=True, capture_output=True)
        sys.stdout.write(r.stdout[-2000:])
        if r.returncode:
            sys.stdout.write(r.stderr[-3000:])
            failed = True
            for dst in rows_for: report.append((dst, "FAILED", f"{st['script']} exit {r.returncode}"))
            if not A.keep_going: break
            continue
        if not st["outputs"]:
            report.append((rows_for[0], "OK", "prep step" + (f": {st['note']}" if st.get("note") else "")))
        for src, dst in st["outputs"].items():
            p = (cwd / src).resolve()
            if not p.exists():
                report.append((dst, "FAILED", f"{st['script']} did not write {src}")); failed = True; continue
            report.append((dst, "OK", place(p, dst, out, paper_art, A.check)))
    for src, dst in COPY_ONLY.items():
        if A.only and dst not in A.only: continue
        report.append((dst, "OK", place(HERE / src, dst, out, paper_art, A.check) + " (hand-written, copied)"))
    for dst, why in NO_GENERATOR.items():
        if A.only and dst not in A.only: continue
        report.append((dst, "NO GENERATOR", why))

    if not report:
        print("nothing to do"); return 0
    w = max(len(r[0]) for r in report)
    print("\n" + "=" * 100)
    for dst, status, detail in report: print(f"{dst:<{w}}  {status:<13} {detail}")
    n = lambda s: sum(st == s for _, st, _ in report)  # noqa: E731
    print(f"\n{n('OK')} ok, {n('SKIPPED')} skipped, {n('FAILED')} failed, {n('NO GENERATOR')} without a generator"
          + ("" if A.check else f"; copies in {out}"))
    return 1 if failed else 0


def has_module(name):
    return importlib.util.find_spec(name) is not None


def leading_comments(path):
    """The `%` comment lines a .tex file starts with, and the rest."""
    lines = path.read_text().splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("%"): i += 1
    return "".join(lines[:i]), "".join(lines[i:])


def same_tex(a, b):
    """Equal after dropping the leading `%` comment lines (the paper's copies carry provenance comments)."""
    return leading_comments(a)[1] == leading_comments(b)[1]


def place(src, dst, out, paper_art, check):
    """Copy src to out/dst (and the paper repo); with check, compare instead. Returns a one-line note."""
    notes = []
    ref = paper_art / dst if paper_art is not None else None
    if ref is not None:
        if not ref.exists(): notes.append(f"not in {paper_art}")
        elif src.suffix == ".tex": notes.append("identical to the paper's copy" if same_tex(src, ref) else "DIFFERS from the paper's copy")
        else: notes.append("binary (not compared)")
    if not check:
        shutil.copyfile(src, out / dst)
        if ref is not None and ref.exists():
            if src.suffix == ".tex":   # keep the provenance comment block the paper's copy carries
                head_ref, _ = leading_comments(ref); head_new, body_new = leading_comments(src)
                ref.write_text((head_new or head_ref) + body_new)
            else:
                shutil.copyfile(src, ref)
            notes.append("copied to the paper repo")
    return "; ".join(notes) if notes else "written"


if __name__ == "__main__":
    sys.exit(main())
