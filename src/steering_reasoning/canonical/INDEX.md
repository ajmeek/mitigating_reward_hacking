# Canonical objects for the steering setup

One place that says which vector, capture, label set and split is *the* one. Everything the paper
reports is built from these. The vectors are copied here (`vectors_L35.safetensors`) so analysis code
can `from steering_reasoning.canonical import load`; large data stays where it was produced and is
pointed to by `paths` in `__init__.py`.

## Model and layer

`Qwen/Qwen3.5-27B` served in bf16 (vLLM + vllm-lens). Every direction is fit on bf16 captures: the
all-103 per-token capture (`scripts/capture_all_tokens.py`) and its segment means, and the per-token
capture of the 42 hacked rollouts at layers 31 and 35 (`scripts/dump_activations.py`). Layer **35**.

## Vectors (`vectors_L35.safetensors`, keys → provenance)

| key | what | recipe | source of record |
|---|---|---|---|
| `u` | **the steering vector** | u = norm(â − b̂); a = mean(hack segs) − mean(rest), b = mean(decline segs) − mean(rest), per-rollout DiffMean on train rollouts, bf16 segment means | `experiments/probe-refit-bf16/artifacts/directions_bf16_all_layers.safetensors` row 35 (`scripts/fit_directions_sweep.py`) |
| `s` | shared hack/anti-hack axis | s = norm(â + b̂) from the same a, b; exactly orthogonal to u | same file, rows a and b at 35 |
| `e` | approach-to-action axis | DiffMean of tokens 20–60 before the end of a reasoning turn vs >200 before, band dropped, position-matched across turns in 12 log bins, turns with no hack/decline segments, per-rollout weighting, train rollouts | `experiments/wrapup-axis-paperstyle/artifacts/e_paper_bf16_L35.safetensors:e_paper_any_call` (`scripts/fit_e.py --layer 35`) |

`cos(a, b)` at L35 is stored in the metadata (`COS_AB`), so `a_b_hat()` reconstructs the two poles
exactly. `u_perp_e()` gives the y-axis of the paper's plane figures (14° from u). There is no stored
mean activation: every figure centres the segments it plots on the mean of the capture it loads
(`experiments/probe-refit-bf16/artifacts/mu_bf16_L35.safetensors`, from `scripts/compute_mu.py`).

Derived numbers to quote: cos(s, u) = 0.00, cos(e, u) = −0.24, cos(e, s) = 0.60, cos(a, b) = 0.35.

Not canonical: the within-turn e variants and the logistic termination probe (`paths["e_variants"]`). For a direction at
another layer use the all-layer file (`paths["directions_all_layers_bf16"]`).

## Steering

The steering artifact for the u arms is `u_L35_steering.safetensors` (+ `.json`). Cap along u,
decode-only, thresholds in raw (uncentred) projection units; the ordinary-reasoning mean sits near −10
on u, and the figures subtract the capture mean. The paper's sweep (`experiments/diff-mean-steering/`,
`artifacts/summary_fleet.csv`) runs cap 0, −4, −8, −12, −16, addition −2, −4, −6, −8 and the baseline on
the 30 test tasks, both variants, seeds 42 and 43, at a 120-message budget. Any other artifact (an
optimized or spec vector) is an explicit row in an experiment's `arms.tsv` under its own name.

## Data

| object | location | notes |
|---|---|---|
| **splits** | `src/steering_reasoning/data/splits.json` | `impossible.{train 24, val 8, test 30}` and `original.{test 30}` (same 30 task ids, paired), `test_groups` A/B/C |
| per-token capture, 42 hacked rollouts, L31 + L35 | `data/captures/qwen35_27b_bf16_token_L31_35/` (layer-35 slice: `data/captures/qwen35_27b_bf16_token_L35/`) | `scripts/dump_activations.py`; format in `data/README.md` |
| all-103 capture (per token, all 64 layers) | `scripts/capture_all_tokens.py` output | includes the 61 clean rollouts; `segmeans/` holds the per-layer segment means the fit streams |
| hack labels (sentence-level judge) | MisTrace `span_labels_json`, joined by `activations.join_judge_labels` | |
| decline (anti-hack) labels | `experiments/anti-hack-labeling/artifacts/labels_full.json` + `decline_unique.json` (kinds) | `labels.load_anti_hack_labels` |
| first HACK edit per rollout (action anchor) | `experiments/hack-action-anchor/artifacts/action_anchor_table.csv` | |
| per-token u projections + labels, all 42 rollouts | `data/derived/token_series.parquet` | `scripts/extract_token_series.py` |

## Readouts

J-lens: `lens.py` (`jacobian(35)`, `top_tokens`); lens weights from `neuronpedia/jacobian-lens`
(HF hub, cached), unembedding and final-norm gain from the `Qwen/Qwen3.5-27B` shards
(`lens.load_unembedding`). The J-lens is the readout the paper reports; the logit lens is comparison only.
