# paper-artifacts

Figures and tables for the write-up. Each subfolder is one group of artifacts: the
script(s) that make them sit beside the PDF (for LaTeX) and PNG (for review) they
produce. Scripts read from `experiments/*/artifacts/`, never from raw data. Vectors are the canonical bf16 u, s, e
(`src/steering_reasoning/canonical/`, see the repo CLAUDE.md).

Style: `paper_style.py` applies the repo's figure style, `steering_reasoning.plot_style` in its
"paper" variant (rules in `src/steering_reasoning/plot_style/README.md`: paper and ink, teal-muted
first series, vermillion for the baseline or bad case, no gridlines, left/bottom spines only) and then
swaps the face to Computer Modern Sans to match the paper body text. Conventions on top of that:

- Color carries valence. A series that is the treated or good case is teal
  (`paper_style.TEAL`), the baseline or bad case is vermillion, and a control or
  reference that is neither is grey (`paper_style.NEUTRAL`). Series that are merely
  different things with no valence take the wheel `paper_style.CATEGORICAL` in order:
  teal, yellow, green, cool grey, purple. Teal is shared between the two uses on purpose.
  Vermillion is never used for a no-valence series.
- Lines with dots: a thin line (0.9 pt) under dots filled in the series color with a
  0.5 pt ink outline.
- Error bars are always ink-black 90% intervals (Wilson for rates).
- Ordered treatments (clamp doses) take `paper_style.teal_ramp(n)`, light to dark;
  the untreated baseline is vermillion.
- No fractions or counts annotated on points.
- `paper_style.save(fig, stem)` writes `<stem>.pdf` and `<stem>.png` beside the script.

Run every item the paper includes with the driver (from the repo root, after `uv sync`):

```
uv run python paper-artifacts/make_paper_artifacts.py              # -> paper-artifacts/out/<paper file name>
uv run python paper-artifacts/make_paper_artifacts.py --list       # steps, outputs, gitignored inputs each needs
uv run python paper-artifacts/make_paper_artifacts.py --check --paper-repo <paper checkout>   # compare, copy nothing
uv run python paper-artifacts/make_paper_artifacts.py --paper-repo <paper checkout>           # copy into its artifacts/
```

The driver's `STEPS` table is the map from paper file name to generator; it runs each script from its own
directory, renames outputs that the paper includes under another name (`action_plane_ewma.pdf` →
`action_plane_turn_paths.pdf`), skips a step whose gitignored inputs are absent and says what to pull,
and lists the one paper item that has no generator in this repo (`fig1_floor.pdf`, drawn by hand). A single script can still be run
on its own: `uv run python paper-artifacts/<group>/<script>.py`.

Style: the figure style is vendored in the repo (`steering_reasoning.plot_style`, merged 2026-10-13),
so a clean clone renders every figure as the paper has it; CMU Sans is picked up from a TeX Live path
and otherwise falls back to Helvetica or DejaVu.
Every script resolves the repo root from its own location.

## Status (2026-10-13, arXiv version)

The paper's LaTeX reads copies of these outputs from the paper repo's `artifacts/` folder;
`make_paper_artifacts.py` writes them under the paper's names (and copies them with `--paper-repo`).
Groups are split into those behind the paper and those that are not. The one item the paper uses
that has **no generator in this repo** (`fig1_floor`) is noted in the root README.

### In the paper

| group / script | paper item | source |
|---|---|---|
| `cap_vs_floor/` | cap and floor thresholds figure | `experiments/spec-vector/artifacts/cap_floor/{heldout_token_hist,token_hist}.csv`, `thresholds.json` |
| `transcript_u_trace/` | monitor trace over one rollout | `data/captures/qwen35_27b_bf16_token_L35/` (gitignored; needs torch) + `data/derived/token_series.parquet` (gitignored; `scripts/extract_token_series.py`) + `experiments/hack-action-anchor/data/edit_judgments.json` (tracked); per-token h·u over one held-out rollout (lcbhard_22), EWMA and a zoom on the hack turn; takes a task id to draw others |
| `sorh_specvec/` (`sorh_rows.py`, `make_sorh_table.py`, `sorh_data.py`) | School of Reward Hacks figure and table | `experiments/spec-vector-sorh/artifacts/`, `experiments/opt-vector-sorh/artifacts/` via `sorh_all_arms.csv` |
| `story_lie_tables/` (`story_lie_bars.py`, `story_by_story_ci.py`) | short-story lying figure, per-story table, prompt blurbs | `experiments/story-dilemma-steering/artifacts/codes/grades_glm.jsonl` |
| `probe_auc_by_layer/` | layer-selection figure | `experiments/probe-refit-bf16/artifacts/sweep_bf16.json` (bf16 all-layer sweep, val split) |
| `action_plane/segment_scatter.py` | approach-axis figure, left | `experiments/hack-action-anchor/artifacts/segment_projections_eu_bf16.csv` |
| `action_plane/action_plane_ewma.py` | approach-axis figure, right (copied to the paper as `action_plane_turn_paths.pdf`) | `experiments/hack-action-anchor/artifacts/turn_paths_last512_bf16.npz` + edit judge (mean path over the last 128 tokens before a tool call for hack edits, honest submits and other turns; causal EWMA, half-life 1.5 log2 d, two passes; held-out, bf16) |
| `dose_response/dose_response_rows.py` | main dose-response figure (`fig4_oneplane_bands`): honest-submission and hack rates on the impossible tests and solve rate on the original tests per arm, joined within each steering family, 90% Wilson bands; `--all` also draws a three-row layout and error-bar / no-interval styles | `experiments/diff-mean-steering/artifacts/summary_fleet.csv`, `experiments/spec-prompting/artifacts/summary_m120.csv`, `experiments/spec-vector/artifacts/results_120msg/summary.csv`, `honest_submit/honest_counts.csv` |
| `sorh_specvec/sorh_rows.py` | School of Reward Hacks figure (`sorh_oneplane_bands`) in the same layout: honest and complete / hack / degenerate shares per arm; `--all` draws the other variants | `sorh_all_arms.csv` |
| `results_table/` | full results table (`clamp_summary_table.tex`, appendix: hack, honest submit, solve per arm) | `summary_fleet.csv`, `spec-prompting/artifacts/summary_m120.csv`, `spec-vector/artifacts/results_120msg/summary.csv`, `experiments/honest-submission-judge/data/labels*.jsonl` |
| `honest_submit/` | honest-submission counts (`honest_counts.csv`: hacked / honest / neither per arm; standalone table no longer in the paper) | `experiments/honest-submission-judge/data/labels.jsonl` (+ `labels_spec.jsonl`) |
| `jlens_main_table/` | main-body J-lens table (`jlens_main_table_2block.tex`) | `jlens_main_readouts.json` beside the script, computed by `jlens_main_readouts.py` from the canonical `u`, `spec-vector/artifacts/v1_cites_L35`, `opt-vector/artifacts/opt_27b_L35_v3c{,_cl}` through `steering_reasoning.lens` (needs torch, transformers, the Neuronpedia lens) |
| `jlens_pole_table/` | appendix J-lens pole table | `experiments/anti-hack-labeling/artifacts/jlens_s_L35.json`, `experiments/hack-action-anchor/artifacts/jlens_u_perp_e_L35.md` |
| `prehack_compact/` | pre-hack appendix (`prehack_ramps.pdf`, `prehack_window_auc.pdf`) | `experiments/prehack-u-profiles/artifacts/{turn_onset_ramps,hazard_mean_vs_max_gap0}.csv`, `experiments/hack-action-anchor/artifacts/turn_paths_last512_bf16.npz` + edit judge via `experiments/prehack-u-profiles/scripts/edits.py` |
| `spec_adversarial/` | adversarial-instruction appendix | `experiments/spec-prompting/artifacts/rollouts_m120.csv` (spec arm), `experiments/diff-mean-steering/artifacts/summary_fleet.csv` (baseline), and the adversarial arm's counts cached in `adversarial_counts.json` beside the script (recomputed from the gitignored eval logs under `experiments/spec-adversarial-instruction/data/logs/` with `--from-logs`) |
| (not here) `experiments/intervention-magnitude/scripts/final_figure.py` | off-manifold appendix (`off_manifold_final_measured.pdf`) | with `--from-table` (what the driver runs) the tracked `artifacts/off_manifold_table_measured.csv`; without it `artifacts/{stats,whitened}_L35.npz` (gitignored; regenerate with that experiment's `capture_stats.py`, `whiten_pass.py`), `data/proj_{opt,spec}_L35.npz`, `artifacts/measured_push.csv` and the sweep's `summary_fleet.csv`. The figure is written into that experiment's `artifacts/` |
| `story_lie_tables/story_prompts.py` | the six story prompts in the appendix (`story_prompt_<key>.tex`) | `experiments/story-dilemma-steering/prompts.json` + the length suffix `generate_stories.py` appends; `story_prompts_blurbs.tex` beside it is hand-written |
| `story_lie_tables/story_by_story_ci.py` | per-story lie / honesty table (`story_by_story_ci.tex`): 90% Wilson intervals, cells tinted by rate (needs `colortbl` in the paper), Spearman's rho between $s$ and each story's rate; `story_spearman.json` adds the mean rho over the six stories with a 90% t-interval for the Figure 6 caption | the same grades as `story_lie_bars.py` |
