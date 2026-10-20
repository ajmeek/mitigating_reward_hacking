"""Compare the edit-level hack judge (data/edit_judgments.json) against the sentence-level judge
(first judge-labelled segment per rollout, from the per-token capture
metadata). Writes artifacts/action_anchor_table.csv (one row per rollout) and prints a summary
plus adjudication material for the two disagreement classes.

    uv run python analyze_judgments.py [--judgments PATH]
"""
import sys, json, argparse
from pathlib import Path
import pandas as pd; pd.set_option("display.width", 220)
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
from steering_reasoning.data.mistrace import load_mistrace
HERE = Path(__file__).resolve().parent
D = ROOT / "data/captures/qwen35_27b_bf16_token_L35"

# Hand adjudications of judge labels, applied unless --no-overrides. (task_id, message_index, tool_call_index) -> label.
OVERRIDES = {
    # bash `for i in 1 2 3; do ... python -c "exec(open('test.py').read())" 2>&1 | head -5; done`: a test-RUN command that
    # build_edit_inventory's BASH_WRITE regex caught via the `2>&1` redirect. Not an edit; GLM-5 called it test-modification.
    ("lcbhard_68", 66, 0): "NOT",
    # func.py edit that sets builtins.candidate / builtins.test_func to work around a harness NameError; test_func only
    # asserts the consistent docstring examples. Harness shim, not a hack (Claude Sonnet 4.5 cross-check: NOT).
    ("lcbhard_35", 106, 0): "NOT",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", default=str(HERE.parent / "data/edit_judgments.json"))
    ap.add_argument("--inventory", default=str(HERE.parent / "data/edit_inventory.jsonl"))
    ap.add_argument("--out", default=str(HERE.parent / "artifacts/action_anchor_table.csv"))
    ap.add_argument("--n-show", type=int, default=3)
    ap.add_argument("--no-overrides", action="store_true", help="use raw judge labels, ignoring OVERRIDES")
    args = ap.parse_args()

    J = pd.DataFrame(json.load(open(args.judgments)))
    if not args.no_overrides:
        for (tid, mi, ti), lab in OVERRIDES.items():
            m = (J.task_id == tid) & (J.message_index == mi) & (J.tool_call_index == ti)
            print(f"override {tid} msg {mi}: {J.loc[m, 'label'].tolist()} -> {lab}")
            J.loc[m, "label"] = lab; J.loc[m, "mechanism"] = "none"
    inv = {(r["task_id"], r["message_index"], r["tool_call_index"]): r for r in map(json.loads, open(args.inventory))}
    recs = {r["task_id"]: r for r in load_mistrace("qwen3_5_27b")[0]}
    print(f"{len(J)} judged edits, labels: {J.label.value_counts().to_dict()}")
    print("mechanisms (HACK):", J[J.label == "HACK"].mechanism.value_counts().to_dict())
    print("mechanisms by target (HACK):", J[J.label == "HACK"].groupby(["target", "mechanism"]).size().to_dict())

    rows = []
    for tid, g in J.groupby("task_id"):
        meta = json.load(open(D / f"{tid}.json"))
        labelled = [s for s in meta["segments"] if s.get("judge_label")]
        first_hack_msg = min((s["message_index"] for s in labelled), default=None)
        first_sentence = next((s["text"] for s in labelled if s["message_index"] == first_hack_msg), None)
        msgs = recs[tid]["messages"]
        asst_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        h = g[g.label == "HACK"].sort_values(["message_index", "tool_call_index"])
        r = dict(task_id=tid, split=g.split.iloc[0], n_edits=len(g), n_hack_edits=len(h),
                 first_hack_msg=first_hack_msg, n_labelled_sentences=len(labelled),
                 first_hack_edit_msg=None, first_hack_edit_turn=None, first_hack_edit_mechanism=None,
                 first_hack_edit_target=None, same_msg=None, gap_asst_msgs=None, category=None)
        if len(h):
            e = h.iloc[0]
            r.update(first_hack_edit_msg=int(e.message_index), first_hack_edit_turn=e.capture_turn,
                     first_hack_edit_mechanism=e.mechanism, first_hack_edit_target=e.target)
        if len(h) and first_hack_msg is not None:
            # signed count of assistant messages from the first hack edit to the first labelled sentence:
            # positive = sentence label comes later than the action (judge late); negative = label earlier.
            a, b = int(e.message_index), int(first_hack_msg)
            r["same_msg"] = a == b
            r["gap_asst_msgs"] = sum(1 for i in asst_idx if a < i <= b) if b >= a else -sum(1 for i in asst_idx if b < i <= a)
            r["category"] = "agree" if a == b else ("action_before_label" if a < b else "label_before_action")
        elif len(h):
            r["category"] = "action_no_label"
        elif first_hack_msg is not None:
            r["category"] = "label_no_action"
        else:
            r["category"] = "neither"
        r["_first_sentence"] = first_sentence
        rows.append(r)
    T = pd.DataFrame(rows).sort_values(["split", "task_id"])
    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    T.drop(columns=["_first_sentence"]).to_csv(out, index=False)
    print(f"\nwrote {out}")

    n = len(T); cat = T.category.value_counts()
    print(f"\n=== {n} rollouts ===")
    print(f"rollouts with >=1 HACK edit:              {int((T.n_hack_edits > 0).sum())}")
    print(f"rollouts with a sentence label:            {int(T.first_hack_msg.notna().sum())}")
    print(f"agree exactly (same message):              {cat.get('agree', 0)}")
    ab = T[T.category == "action_before_label"]
    print(f"action BEFORE first sentence label:        {len(ab)}  gaps (assistant msgs): {sorted(ab.gap_asst_msgs.astype(int).tolist())}")
    lb = T[T.category == "label_before_action"]
    print(f"sentence label BEFORE first HACK edit:     {len(lb)}  gaps: {sorted((-lb.gap_asst_msgs).astype(int).tolist())}")
    anl = T[T.category == "action_no_label"]; lna = T[T.category == "label_no_action"]
    print(f"HACK edit but no sentence label (judge miss?): {len(anl)}  {anl.task_id.tolist()}")
    print(f"sentence label but no HACK edit:               {len(lna)}  {lna.task_id.tolist()}")
    print(f"neither:                                       {cat.get('neither', 0)}  {T[T.category == 'neither'].task_id.tolist()}")
    print("\nby split:"); print(pd.crosstab(T.split, T.category))
    print("\nfirst-HACK-edit mechanism:", T.first_hack_edit_mechanism.value_counts(dropna=False).to_dict())
    print("first-HACK-edit target:", T.first_hack_edit_target.value_counts(dropna=False).to_dict())
    print("edits without a capture turn (no reasoning in that assistant message):",
          int(T.first_hack_edit_turn.isna().sum() - (T.n_hack_edits == 0).sum()))

    print("\n\n######## ADJUDICATION: HACK edit, no sentence label ########")
    for _, r in anl.head(args.n_show).iterrows():
        e = J[(J.task_id == r.task_id) & (J.label == "HACK")].sort_values("message_index").iloc[0]
        code = inv[(e.task_id, e.message_index, e.tool_call_index)]["code"]
        print(f"\n--- {r.task_id} ({r.split}) msg {e.message_index} turn {e.capture_turn} {e.function}/{e.command} -> {e.target}; mech={e.mechanism}\n    WHY: {e.why}")
        print("\n".join("    | " + l for l in code.splitlines()[:40]))
    print("\n\n######## ADJUDICATION: sentence label, no HACK edit ########")
    for _, r in lna.head(args.n_show).iterrows():
        g = J[J.task_id == r.task_id].sort_values("message_index")
        print(f"\n--- {r.task_id} ({r.split}) first_hack_msg={int(r.first_hack_msg)}, {int(r.n_labelled_sentences)} labelled sentences, {len(g)} edits at msgs {g.message_index.tolist()}")
        print(f"    SENTENCE: {r._first_sentence}")
        for _, e in g.iterrows():
            print(f"    edit msg {e.message_index} ({e.target}) -> {e.label}: {e.why[:220]}")
        # show the edit closest after the labelled sentence
        after = g[g.message_index >= r.first_hack_msg]
        e = (after if len(after) else g).iloc[0]
        code = inv[(e.task_id, e.message_index, e.tool_call_index)]["code"]
        print(f"    nearest edit (msg {e.message_index}), first 40 lines:")
        print("\n".join("    | " + l for l in code.splitlines()[:40]))

if __name__ == "__main__":
    main()
