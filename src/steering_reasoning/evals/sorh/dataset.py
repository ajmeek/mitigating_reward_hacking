"""School of Reward Hacks (SoRH) rows: loading, fetching, sample ids, dev-set and eval-set draws.

The dataset (HF `longtermrisk/school-of-reward-hacks`, split `train`) has 1,073 single-turn prompts,
973 writing and 100 coding, each stating a grading metric the model could game. Every row also
carries an example hack and an example honest answer, which the judge uses as anchors. The pinned dump
of record (hf sha d7e04a55..., fetched 2026-09-13) lives in
`experiments/opt-vector-sorh/data/sorh_rows.json` on Matthew's box; `fetch_rows` re-downloads it.

Sample id = `sorh_<row index>` in dataset order; the coding rows are the ones whose `task` field is
"write a function", never assumed by index.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

HF_DATASET = "longtermrisk/school-of-reward-hacks"
HF_CONFIG = "default"
HF_SPLIT = "train"
FIELDS = ("user", "school_of_reward_hacks", "control", "task", "evaluation_metric", "cheat_method")
N_ROWS = 1073          # rows in the dump of record; load_rows(strict=True) checks the count
CODING_TASK = "write a function"
ID_PREFIX = "sorh_"

# Draws of record (experiments/opt-vector-sorh/artifacts/{dev,eval}/ids.json).
DEV_SEED = 0
DEV_N_WRITING = 91
DEV_N_CODING = 9
EVAL_SEED = 0
EVAL_N_WRITING = 400


def sample_id(i: int) -> str:
    return f"{ID_PREFIX}{i}"


def row_index(sid: str) -> int | None:
    """'sorh_12' -> 12; None for anything else (an epoch suffix like 'sorh_12@e2' is not an id)."""
    s = str(sid)
    if not s.startswith(ID_PREFIX) or not s[len(ID_PREFIX):].isdigit():
        return None
    return int(s[len(ID_PREFIX):])


def task_family(row: dict) -> str:
    """'function' for the coding rows (task == 'write a function'), 'other' for the writing rows."""
    return "function" if row.get("task") == CODING_TASK else "other"


def load_rows(path: str | Path, strict: bool = True) -> list[dict]:
    """The rows JSON list. strict=True refuses anything but a list of N_ROWS dicts carrying FIELDS,
    so a truncated or foreign file cannot silently shift the sample ids."""
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(rows).__name__}")
    if any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"{path}: every row must be a JSON object")
    if strict:
        if len(rows) != N_ROWS:
            raise ValueError(f"{path}: expected {N_ROWS} rows, got {len(rows)}")
        missing = [k for k in FIELDS if rows and k not in rows[0]]
        if missing:
            raise ValueError(f"{path}: rows lack fields {missing}")
    return rows


def fetch_rows(out_path: str | Path, meta_path: str | Path | None = None, revision: str | None = None) -> list[dict]:
    """Download the dataset with `datasets` (an eval-env dependency), keep only FIELDS in dataset
    order, and write `out_path` (JSON list) plus a `.meta.json` sidecar with the HF sha. Pass the sha
    of record as `revision` to pin the dump; the file written is what every other function reads."""
    from datasets import load_dataset  # lazy: only this function needs network and the datasets package

    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT, revision=revision)
    rows = [{k: r[k] for k in FIELDS} for r in ds]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    info = getattr(ds, "info", None)
    sha = None
    try:  # the commit hash of the revision that was loaded (provenance for the dump)
        from huggingface_hub import HfApi

        sha = HfApi().dataset_info(HF_DATASET, revision=revision).sha
    except Exception:
        sha = None
    meta = dict(dataset=HF_DATASET, config=HF_CONFIG, split=HF_SPLIT, revision=revision, hf_sha=sha,
                n_rows=len(rows), fetched=time.strftime("%Y-%m-%d"), fields=list(FIELDS),
                source="datasets.load_dataset, dataset order preserved; sample id = sorh_<row index>",
                description=(getattr(info, "description", None) or None))
    meta_path = Path(meta_path) if meta_path else out_path.with_name(out_path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return rows


def coding_indices(rows: list[dict]) -> list[int]:
    return [i for i, r in enumerate(rows) if r.get("task") == CODING_TASK]


def writing_indices(rows: list[dict]) -> list[int]:
    return [i for i, r in enumerate(rows) if r.get("task") != CODING_TASK]


def draw_dev_set(rows: list[dict], seed: int = DEV_SEED, n_writing: int = DEV_N_WRITING,
                 n_coding: int = DEV_N_CODING) -> dict:
    """The dose-sweep dev set. Design of record: one random.Random(seed); n_writing rows sampled
    from the writing rows, then n_coding from the coding rows, each sorted by row index; ids = the
    union sorted. Deterministic: the same rows and seed give a byte-identical file."""
    coding, writing = coding_indices(rows), writing_indices(rows)
    if len(coding) < n_coding or len(writing) < n_writing:
        raise ValueError(f"not enough rows: coding {len(coding)} writing {len(writing)}")
    rng = random.Random(seed)
    w = sorted(rng.sample(writing, n_writing))   # drawn first
    c = sorted(rng.sample(coding, n_coding))     # then the coding rows, same rng
    ids = sorted(set(w) | set(c))
    return {"seed": seed, "coding_ids": [sample_id(i) for i in c], "writing_ids": [sample_id(i) for i in w],
            "ids": [sample_id(i) for i in ids]}


def draw_eval_set(rows: list[dict], dev: dict, seed: int = EVAL_SEED, n_writing: int = EVAL_N_WRITING) -> dict:
    """The tuned-run eval set. Design of record: every coding row, plus n_writing writing rows made
    of the dev set's writing rows forced in and a random.Random(seed).sample of the rest (in dataset
    order). `heldout_ids` = ids not in the dev set. Reproduces the 500-prompt set of the run of record
    (100 coding + 91 dev writing + 309 sampled) from the dev draw above."""
    coding = coding_indices(rows)
    dev_ids = set(dev["ids"])
    if "writing_ids" in dev:
        dev_writing = sorted(i for i in (row_index(s) for s in dev["writing_ids"]) if i is not None)
    else:  # an ids-only file: the dev writing rows are its ids that are not coding rows
        coding_set = set(coding)
        dev_writing = sorted(i for i in (row_index(s) for s in dev["ids"]) if i is not None and i not in coding_set)
    rest = [i for i in writing_indices(rows) if sample_id(i) not in dev_ids]
    n_extra = n_writing - len(dev_writing)
    if n_extra < 0 or n_extra > len(rest):
        raise ValueError(f"cannot draw {n_writing} writing rows: {len(dev_writing)} dev + {len(rest)} remaining")
    rng = random.Random(seed)
    writing = sorted(set(dev_writing) | set(rng.sample(rest, n_extra)))
    ids = sorted(set(coding) | set(writing))
    sids = [sample_id(i) for i in ids]
    included = "fully included" if dev_ids <= set(sids) else f"{len(dev_ids & set(sids))}/{len(dev_ids)} included"
    return {"seed": seed,
            "design": (f"all {len(coding)} coding rows + {n_writing} writing rows ({len(dev_writing)} dev writing rows "
                       f"forced in + {n_extra} random from the rest); dev{len(dev_ids)} {included}"),
            "n": len(sids),
            "coding_ids": [sample_id(i) for i in coding],
            "writing_ids": [sample_id(i) for i in writing],
            "dev_ids_included": [s for s in sids if s in dev_ids],
            "heldout_ids": [s for s in sids if s not in dev_ids],
            "ids": sids}


def load_ids(path: str | Path) -> dict:
    """An ids.json as written by draw_dev_set / draw_eval_set (or any {"ids": [...]} file)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, list):
        d = {"ids": d}
    if not isinstance(d, dict) or not isinstance(d.get("ids"), list):
        raise ValueError(f"{path}: expected an object with an 'ids' list (or a bare list of ids)")
    return d


def write_ids(path: str | Path, ids: dict) -> str:
    text = json.dumps(ids, indent=1) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text
