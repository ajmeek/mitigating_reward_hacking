"""School of Reward Hacks (SoRH) steering eval: dataset, arms, Inspect task and runner, pre-run checks,
blind judge, and analysis. `python -m steering_reasoning.evals.sorh --help` for the command line;
`scripts/run_sorh_arms.sh` launches one proxy and one runner per arm of an arms.tsv; the README's
School of Reward Hacks step is the recipe.

Only `task.run_eval`, `logs.finished_ids`/`remaining_ids`, `logs.collect_arm` and `grade.grade` need
inspect_ai (and `grade` the openai SDK); everything else runs on plain python 3.
"""

from .arms import ArmSpec, arm_labels, parse_arm_spec, parse_cell_name, read_arms_tsv
from .dataset import (CODING_TASK, N_ROWS, draw_dev_set, draw_eval_set, load_ids, load_rows, row_index,
                      sample_id, task_family)
from .logs import finished_dict, finished_sample, remaining_ids, scan_dir

__all__ = ["ArmSpec", "arm_labels", "parse_arm_spec", "parse_cell_name", "read_arms_tsv",
           "CODING_TASK", "N_ROWS", "draw_dev_set", "draw_eval_set", "load_ids", "load_rows", "row_index",
           "sample_id", "task_family", "finished_dict", "finished_sample", "remaining_ids", "scan_dir"]
