"""Two simple baselines: the bar the ML forecaster must beat (decision 20).

  naive: q50 = units on the same day last week (target - forecast.baseline_lag_days, 7).
         Also the fallback for store-items with no usable history.
  stub:  q50 = mean of units 7, 14, 21 and 28 days before the target. This is the
         teammate's forecasts_stub.parquet, i.e. what the order model uses today.

Both: q80/q90/q95 = q50 + z * spread, spread = std of daily units over the 28 days
ending 7 days before the target (forecast.baseline_lag_days / stubs.spread_end_lag), clipped at 0 and never crossing. Days with no row (not
on sale yet) are skipped, not counted as zero; no history at all gives 0.

Both are computed by the teammate's own function stubs.make_stubs.forecast_targets
(read only), so the stub bar is exactly her file. The naive baseline passes it a copy of
the settings with one lag and the forecast: section's z values; config.yaml is not changed.
Every input is at least 7 days old, so the baselines did not change when R2 went to 2 days.
"""

import copy

import pandas as pd

from stubs.make_stubs import forecast_targets

KEY = ["date", "store_id", "item_id"]
LINES = ["q50", "q80", "q90", "q95"]
METHODS = ("naive", "stub")


def _settings(cfg: dict, method: str) -> dict:
    """The config forecast_targets reads for this method (a copy; cfg is untouched)."""
    if method == "stub":
        return cfg
    if method != "naive":
        raise ValueError(f"unknown baseline method {method!r}; use one of {METHODS}")
    f = cfg["forecast"]
    c = copy.deepcopy(cfg)
    c["stubs"].update({
        "lags": [int(f["baseline_lag_days"])],            # same day last week
        "spread_end_lag": int(f["baseline_lag_days"]),   # unchanged baseline (7 days back)
        "spread_window_days": int(max(f["rolling_windows"])),
        "z": dict(f["baseline_z"]),
    })
    return c


def baseline_lines(sales: pd.DataFrame, targets: pd.DataFrame, cfg: dict,
                   method: str = "naive") -> pd.DataFrame:
    """q50..q95 for each target row (date, store_id, item_id), in the targets' order.
    A store-item missing from `sales` gets 0 on every line."""
    settings = _settings(cfg, method)
    by_target = forecast_targets(sales, targets["date"].unique(), settings)
    out = targets[KEY].merge(by_target, on=KEY, how="left")
    out[LINES] = out[LINES].fillna(0.0).astype("float64")
    return out[KEY + LINES].reset_index(drop=True)


def targets_for(sales: pd.DataFrame, start, end) -> pd.DataFrame:
    """Every store-item row of `sales` dated start..end (both included): the days to score."""
    d = sales["date"]
    rows = sales.loc[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)), KEY]
    return rows.sort_values(KEY).reset_index(drop=True)
