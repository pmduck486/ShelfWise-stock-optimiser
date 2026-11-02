"""Experiment log and the both-folds rule (Task 4; re-check strategy 4, approved 2026-10-30).

  - log_experiment: one row per try in outputs/forecast/scores/experiments.csv (what changed,
    its scores on each fold, kept or not, why). The first Task 4 run is the base row.
  - keep_change: a change stays only if it lowers the score on EVERY fold. A tie or a loss
    on any fold keeps the simpler version. A fold's score may be a list of seeds; their
    mean is compared (fix 2: one seed can move the score by up to 0.7%).
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def fold_score(value) -> float:
    """A fold's score: the number itself, or the mean over seeds."""
    return float(np.mean(value))


def keep_change(before: dict, after: dict) -> bool:
    """True only if `after` scores lower than `before` on every fold (lower is better)."""
    if set(before) != set(after):
        raise ValueError(f"before and after must score the same folds: {sorted(before)} vs "
                         f"{sorted(after)}")
    return all(fold_score(after[f]) < fold_score(before[f]) for f in before)


def read_log(path) -> pd.DataFrame:
    """The log as written: floats read back exactly (so a rerun ties instead of "winning" by
    one rounding step), seeds as text, and a missing kept value read as not kept."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    log = pd.read_csv(path, float_precision="round_trip", dtype={"seeds": str})
    if "kept" in log:
        log["kept"] = log["kept"].map({True: True, False: False, "True": True, "False": False})
        log["kept"] = log["kept"].fillna(False).astype(bool)
    return log


def log_experiment(path, row: dict) -> None:
    """Append one try to the log (created with a header on first use)."""
    path = Path(path)
    new = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), **row}])
    if path.exists():
        new = pd.concat([read_log(path), new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
