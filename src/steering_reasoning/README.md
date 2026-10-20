# steering_reasoning — module map

Shared library code graduated from experiment scripts. Heavy deps (torch, safetensors,
numpy, zstandard) are imported lazily inside functions, so the package imports cleanly
from the root eval env; the analysis env is `uv sync` at the repo root (dev group has
numpy/pandas/matplotlib), with `uv run --with torch ...` for the scripts that need torch.

Top-level modules:

- `activations.py` — read `dump_activations.py` output dirs: `load_dump(dump_dir,
  layers=, extra_tensors=)` returns (manifest, rows); `join_judge_labels(rows)` attaches
  MisTrace judge labels to captured segment metadata (loads MisTrace via
  `data/mistrace.py`).
  Also `safetensors_header(path)` and `memmap_tensor(path, name)` for memmapping one
  tensor of a shard too large to load (the fit stages reduce it in row chunks).
- `stats.py` — `wilson(k, n, z=1.645)` (the paper's 90% interval, clamped) and
  `auroc_ranks(pos, neg)` (numpy midrank AUROC); the summarize and fit stages import these
  instead of carrying local copies.
- `s3stream.py` — `s3_sizes(prefix)`, `s3_cp(src, dst)` and `Prefetch(prefix, cache_dir,
  sizes, budget)`: stream multi-GB per-rollout files from S3 one at a time under a disk
  budget, prefetching the next while the current one is reduced (`scripts/fit_directions_sweep.py`).
- `labels.py` — anti-hack label artifacts: `JUNK_KINDS`,
  `load_anti_hack_labels(labels_path, kinds_path)` -> (DECLINE, EXCLUDED) key-sets, and
  `classify_segments(rollout, decline, excluded, dedup_text=)` ->
  (hack_idx, decline_idx, other_idx). `dedup_text=True` reproduces phase2_sweep's
  within-rollout text dedup; `False` reproduces ramp_u_L35's no-dedup variant.
- `directions.py` — `fit_diffmean_ab(rows, layer_index, train_ids, decline, excluded)`
  fits the per-rollout DiffMean hack (a) and anti-hack (b) directions and their
  difference u at one captured layer; `load_u(path)` loads a single-layer safetensors
  artifact as a unit vector. Also the shared helpers `auroc(pos, neg)`, `unit(v)`,
  `project(x, v)`, `trace_diffmean(rows)` and `span_diffmean(gold_rows)` that the
- `temporal.py` — onset scaffolding: `turn_table(rollout, layer_index)` groups segments
  into turns by message_index, `first_hack_turn(rollout)`, `zscore_within(vals)`.
- `figstyle.py` — shared matplotlib styling.
- `lens.py` — vllm-lens interop helpers.

Subpackages:

- `canonical/` — **the** vectors (u, s, e, mu at L35) and pointers to the canonical
  captures, labels and splits, with provenance: see `canonical/INDEX.md`. Start here
  before hunting through `experiments/`.

- `data/` — dataset loading (MisTrace) and the frozen train/val/test splits
  (`splits.json`).
- `evals/` — inspect-ai task, solver, and ImpossibleBench compatibility layer (eval env
  only).
  `evals/sorh/` is the School of Reward Hacks eval (`python -m steering_reasoning.evals.sorh --help`): dataset rows and id-set draws,
  arms tables (`ArmSpec`, `read_arms_tsv`), the single-turn Inspect task and runner, pre-run checks
  (polarity, delivery, servers), the blind judge, and the analysis (per-item/per-arm tables, dose
  selection, final tables); `python -m steering_reasoning.evals.sorh --help`. Only `run`,
  `remaining` and `grade` need inspect_ai.
- `steering/` — steering-vector extraction and artifact handling. Most modules require
  the serving env (vllm + vllm-lens); the exception is `steering/clamp.py`
  (`build_clamp_payload(artifact, threshold, mode=, scale=)` serializing the
  vllm_xargs["apply_steering_vectors"] JSON, and `strip_think(s)`), which is
  env-agnostic and re-exported from `steering/__init__.py`.

Experiment scripts under `experiments/*/scripts/` that predate a module keep their local
copies as frozen records; live tooling imports from here.
The pipeline stages a reproduction runs (labels, capture, fit, calibrate, summarize) are
entry points under `scripts/` (see the root README, "Reproducing the paper"); the experiment
folders they came from keep a stub that dispatches to the new location.
