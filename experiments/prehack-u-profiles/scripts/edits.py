"""Edit-judge labels with the hand adjudications applied (hack-action-anchor/scripts/analyze_judgments.py
OVERRIDES: two raw GLM-5 HACK labels that are not hacks). Every analysis here anchors on these.

    from edits import load_judgments, hack_edit_turns, first_hack_turn
"""
import json, os, sys
HA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hack-action-anchor")  # experiments/hack-action-anchor
sys.path.insert(0, f"{HA}/scripts")
from analyze_judgments import OVERRIDES  # noqa: E402


def load_judgments():
    J = json.load(open(f"{HA}/data/edit_judgments.json"))
    for j in J:
        lab = OVERRIDES.get((j["task_id"], j["message_index"], j["tool_call_index"]))
        if lab: j["label"] = lab
    return J


def hack_edit_turns(J=None):
    """{(task_id, capture_turn)} of HACK-judged edits."""
    return {(j["task_id"], j["capture_turn"]) for j in (J or load_judgments()) if j["label"] == "HACK"}


def first_hack_turn(J=None):
    """task_id -> capture turn of its first HACK-judged edit (rollouts without one are absent)."""
    out = {}
    for t, k in hack_edit_turns(J): out[t] = min(out.get(t, 10 ** 9), k)
    return out
