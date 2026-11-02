"""Post-processing of the raw model lines (Task 5).

postprocess sorts each row's four lines so they never cross (q50 <= q80 <= q90 <= q95),
clips them at 0, and sets all four to 0 on forecast.closed_dates (25 Dec: stores shut).
It runs after calibration too (Task 7), so a stretched line can never cross another.
A missing value is an error, never filled in.

dead_item_rule (decision 26, 2026-10-31): q50 = 0 for a store-item that sold nothing in the 28
days ending min_lag_days back (forecast.dead_item_q50). Only q50: the upper lines keep the model's
value, so an item that starts selling again still gets some stock.
"""

import numpy as np
import pandas as pd

from forecast.metrics import QUANTILES, closed_mask

LINES = list(QUANTILES)                     # q50, q80, q90, q95


def postprocess(lines: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    missing = [c for c in LINES if c not in lines]
    if missing:
        raise ValueError(f"postprocess needs all four lines; missing columns {missing}")
    a = lines[LINES].to_numpy(dtype=float)
    if np.isnan(a).any():
        raise ValueError(f"{int(np.isnan(a).any(axis=1).sum())} rows have a missing line")
    a = np.clip(np.sort(a, axis=1), 0.0, None)
    a[closed_mask(lines["date"], cfg).to_numpy()] = 0.0
    out = lines.copy()
    out[LINES] = a
    return out


def dead_item_rule(lines: pd.DataFrame, recent_mean, cfg: dict) -> pd.DataFrame:
    """q50 = 0 where recent_mean (the 28-day average ending min_lag_days back, roll_mean_28) is 0.
    A NaN average (too little history) is left alone. Returns a copy."""
    out = lines.copy()
    if cfg["forecast"]["dead_item_q50"]:
        out.loc[np.asarray(recent_mean) == 0, "q50"] = 0.0
    return out
