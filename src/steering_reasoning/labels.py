"""Anti-hack label sets and segment classification.

The anti-hack labeling pass (experiments/anti-hack-labeling) produced two artifacts:
`labels_full.json` (DECLINE / EXCLUDED verdicts per segment key) and
`decline_unique.json` (a `kind` per candidate, used to drop junk sentences). This module
turns them into key-sets and classifies a rollout's segments into hack / decline / other.

No heavy deps — pure json/stdlib, importable from the eval env.
"""

import json
from pathlib import Path

#: decline candidates of these kinds are junk fragments, never real declines
JUNK_KINDS = {"no letters", "code / arithmetic / list item", "prompt quotation", "empty"}


def load_anti_hack_labels(labels_path: str | Path, kinds_path: str | Path):
    """Return (DECLINE, EXCLUDED) key-sets from the anti-hack labeling artifacts.

    labels_path: labels_full.json — [{task_id, message_index, segment_idx, label}, ...]
    kinds_path: decline_unique.json — same keys plus a `kind` field.

    Keys are (task_id, message_index, segment_idx). A segment is DECLINE only if its
    kind is not in JUNK_KINDS; EXCLUDED segments are dropped from every class.
    """
    labels = json.loads(Path(labels_path).read_text())
    kinds = {(o["task_id"], o["message_index"], o["segment_idx"]): o.get("kind")
             for o in json.loads(Path(kinds_path).read_text())}
    decline, excluded = set(), set()
    for o in labels:
        k = (o["task_id"], o["message_index"], o["segment_idx"])
        if o["label"] == "EXCLUDED":
            excluded.add(k)
        elif o["label"] == "DECLINE" and kinds.get(k) not in JUNK_KINDS:
            decline.add(k)
    return decline, excluded


def classify_segments(rollout, decline, excluded, dedup_text=True):
    """Split a rollout's segments into (hack_idx, decline_idx, other_idx).

    rollout: a load_dump row (uses task_id and segment_meta; judge_label must already be
    joined via activations.join_judge_labels). EXCLUDED segments are skipped entirely.
    Priority: judge hack label > DECLINE > other.

    dedup_text=True (phase2_sweep semantics): within each class, keep only the first
    segment per distinct stripped-lowercased text, so repeated sentences don't dominate a
    per-rollout mean. dedup_text=False (ramp_u_L35 semantics) keeps every segment.
    """
    seen = {"h": set(), "d": set(), "o": set()}
    h, dc, o = [], [], []
    for i, s in enumerate(rollout["segment_meta"]):
        key = (rollout["task_id"], s["message_index"], s["segment_idx"])
        if key in excluded:
            continue
        if s.get("judge_label"):
            cls, out = "h", h
        elif key in decline:
            cls, out = "d", dc
        else:
            cls, out = "o", o
        if dedup_text:
            txt = (s.get("text") or "").strip().lower()
            if txt in seen[cls]:
                continue
            seen[cls].add(txt)
        out.append(i)
    return h, dc, o
