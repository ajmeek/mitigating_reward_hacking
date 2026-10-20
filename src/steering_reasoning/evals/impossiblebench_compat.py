"""Load ImpossibleBench's LiveCodeBench modules from the git submodule without installing it.

Installing impossiblebench (`pip install -e`) pulls swebench and an inspect_evals git
dependency, and `import impossiblebench` executes its __init__.py which unconditionally
imports swebench_tasks. The LCB modules themselves only need inspect_ai + datasets, and
livecodebench_tasks.py ships its own top-level-import fallback (try/except ImportError that
inserts its directory on sys.path), so loading them by path is supported upstream.
"""

import sys
from pathlib import Path

_IB_PKG_DIR = (
    Path(__file__).resolve().parents[3] / "external" / "impossiblebench" / "src" / "impossiblebench"
)


def _load():
    if not _IB_PKG_DIR.is_dir():
        raise RuntimeError(
            f"impossiblebench submodule not found at {_IB_PKG_DIR}. "
            "Run: git submodule update --init"
        )
    if str(_IB_PKG_DIR) not in sys.path:
        sys.path.insert(0, str(_IB_PKG_DIR))
    import livecodebench_tasks

    return livecodebench_tasks


livecodebench_tasks = _load()
impossible_livecodebench = livecodebench_tasks.impossible_livecodebench
