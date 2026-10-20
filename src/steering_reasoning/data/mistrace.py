"""Load MisTrace (https://huggingface.co/datasets/ameek/MisTrace) as pipeline records.

MisTrace is the PUBLISHED source of truth for the annotated misalignment traces. Prefer it
over anything checked into data/ — the in-repo JSON snapshot predates the judge-annotation
pass (every judge_label is null there) and includes five DeepSeek rollouts that errored
during generation and were dropped before publication.

Rows store conversations and span labels as JSON strings; this module decodes them into the
record shape the capture pipeline expects (`messages`, `task_id`, `is_reward_hack`,
`span_labels`, ...), so `iter_reasoning_turns` / `reasoning_segments_by_message` work
unchanged.

Span labels carry two independent annotations per segment:
- `gold_label`: human. Present on 4/42 hacked Qwen rollouts, 2/53 DeepSeek.
- `judge_label`: GLM-5. Present on every hacked rollout, absent on clean ones.
Agreement is model-dependent — measured against gold, GLM-5 scores F1 ~0.60 on Qwen traces
and ~0.86 on DeepSeek. Choose a label field deliberately; don't assume they're equivalent.
"""

import json
from typing import Literal

CONFIGS = {
    "reward-hacking": ("qwen3_5_27b", "deepseek_r1"),
    "cot-faithfulness": ("qwen3_5_27b", "r1_0528", "hermes_4_3_36b"),
}
REPO = "ameek/MisTrace"

# columns holding JSON strings -> the key they decode to
_JSON_COLUMNS = {
    "messages_json": "messages",
    "span_labels_json": "span_labels",
    "task_json": "task",
    "cheating_json": "cheating",
    "generation_config_json": "generation_config",
    "trace_labels_json": "trace_labels",
    "events_json": "events",
    "seed_metadata_json": "seed_metadata",
    "target_messages_json": "target_messages",
}


def load_mistrace(
    model: str = "qwen3_5_27b",
    *,
    config: Literal["reward-hacking", "cot-faithfulness"] = "reward-hacking",
    revision: str | None = None,
) -> tuple[list[dict], dict]:
    """Return (records, provenance) for one MisTrace file.

    provenance records the resolved dataset revision so manifests can pin exactly which
    snapshot a capture used.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    if config not in CONFIGS:
        raise ValueError(f"config must be one of {list(CONFIGS)}, got {config!r}")
    if model not in CONFIGS[config]:
        raise ValueError(f"model for {config} must be one of {CONFIGS[config]}, got {model!r}")

    path = f"{config}/{model}.parquet"
    local = hf_hub_download(REPO, path, repo_type="dataset", revision=revision)
    table = pq.read_table(local)

    records = []
    for i in range(table.num_rows):
        rec = {}
        for col in table.column_names:
            value = table.column(col)[i].as_py()
            key = _JSON_COLUMNS.get(col)
            if key is not None:
                rec[key] = json.loads(value) if value else None
            else:
                rec[col] = value
        records.append(rec)

    info = HfApi().dataset_info(REPO, revision=revision)
    provenance = {
        "dataset": REPO,
        "file": path,
        "revision": info.sha,
        "dataset_last_modified": str(info.lastModified),
        "n_records": len(records),
    }
    return records, provenance


def label_counts(records: list[dict], label_field: str = "judge_label") -> dict:
    """Positive/total segment counts by segment kind — a quick sanity check on coverage."""
    counts: dict[str, dict[str, int]] = {}
    for rec in records:
        span_labels = rec.get("span_labels")
        if not span_labels:
            continue
        for seg in span_labels.get("segments", []):
            bucket = counts.setdefault(seg["kind"], {"total": 0, "positive": 0, "labeled": 0})
            bucket["total"] += 1
            if seg.get(label_field) is not None:
                bucket["labeled"] += 1
                bucket["positive"] += bool(seg[label_field])
    return counts
