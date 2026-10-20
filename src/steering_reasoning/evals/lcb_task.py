"""Task factory for ImpossibleBench LiveCodeBench-hard runs.

Wraps impossiblebench's impossible_livecodebench with the configuration that produced our
reward-hacking transcripts (task name lcb_<split>_canmod_tools). The scorer is upstream's
deterministic swe_style_humaneval_scorer (agent_type="tools"), which reverts any test.py edits
before running `python test.py` — so on the conflicting split, a CORRECT score is a reward hack
by construction — wrapped by `lcb_scorer` to record the sandbox files at scoring time, so a
rollout that stopped at the message limit can be resumed later (evals/workspace.py).
"""

from pathlib import Path

from .config import MAX_ATTEMPTS, MESSAGE_LIMIT
from .impossiblebench_compat import impossible_livecodebench
from .lcb_solver import transcript_lcb_solver
from .workspace import lcb_scorer

# Copy of the submodule's compose.yaml. Explicit path: Inspect resolves an implicit compose
# file relative to the @task source file, which lives inside the submodule.
COMPOSE_FILE = Path(__file__).parent / "compose.yaml"

SPLITS = ("original", "oneoff", "conflicting")


def lcb_task(
    split: str,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    message_limit: int = MESSAGE_LIMIT,
    limit: int | None = None,
    system_suffix: str = "",
    user_suffix: str = "",
):
    """Build the lcb_<split>_canmod_tools task against the pinned HF dataset.

    shuffle=False keeps dataset order (lcbhard_0..lcbhard_102), matching our transcripts.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return impossible_livecodebench(
        split=split,
        agent_type="tools",
        allow_test_modifications=True,
        solver=transcript_lcb_solver(max_attempts=max_attempts, system_suffix=system_suffix, user_suffix=user_suffix),
        scorer=lcb_scorer(),
        max_attempts=max_attempts,
        message_limit=message_limit,
        sandbox=("docker", str(COMPOSE_FILE)),
        limit=limit,
        shuffle=False,
    )
