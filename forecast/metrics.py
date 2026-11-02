"""Scoring: error and coverage of forecast lines against what actually sold.

  pinball:  quantile loss. For the 95% line, a line too low costs 0.95 per unit missed and
            a line too high costs 0.05 per unit over (forecast-training-plan.md section 1).
  coverage: share of days whose sales were at or below the line, in %. A good 90% line
            covers about 90% of days.
  mae:      average units wrong per day, for the 50% line.

Scores are given for all store-items, perishables and slow sellers, plus coverage of the
line each product is actually stocked to (cover_used_line). Days in forecast.closed_dates
(25 Dec: stores shut) are left out of every score (decision 19).
"""

import numpy as np
import pandas as pd

KEY = ["date", "store_id", "item_id"]
QUANTILES = {"q50": 0.5, "q80": 0.8, "q90": 0.9, "q95": 0.95}


def pinball(y, q_pred, q: float) -> float:
    e = np.asarray(y, dtype=float) - np.asarray(q_pred, dtype=float)
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def coverage(y, q_pred) -> float:
    return float(np.mean(np.asarray(y) <= np.asarray(q_pred)) * 100)


def mae(y, q50) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(q50, dtype=float))))


# ---------------------------------------------------------------------------
# Product groups
# ---------------------------------------------------------------------------
def service_line(items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The line each item is stocked to, mirroring the order model's proposed service levels.
    First match wins: perishable, then top-n staple by rank, then slow mover, then default."""
    rules = cfg["forecast"]["service_lines"]
    staples, slow = rules["top_n_staples"], rules["slow_movers"]
    line = np.select(
        [items["is_perishable"].astype(bool),
         items["rank"] <= staples["n"],
         items["avg_daily_units"] < slow["below_avg_daily_units"]],
        [rules["perishable"], staples["line"], slow["line"]],
        default=rules["default"],
    )
    return pd.DataFrame({"item_id": items["item_id"].to_numpy(), "line": line})


def segments(sales: pd.DataFrame, items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per store-item in `sales`: perishable flag, slow-seller flag (items.csv average
    below forecast.slow_seller_max_avg_units; reporting only) and the line it is stocked to."""
    pairs = sales[["store_id", "item_id"]].drop_duplicates()
    info = items[["item_id", "is_perishable", "avg_daily_units"]].merge(
        service_line(items, cfg).rename(columns={"line": "used_line"}), on="item_id")
    seg = pairs.merge(info, on="item_id", how="left", validate="many_to_one")
    if seg["used_line"].isna().any():
        missing = seg.loc[seg["used_line"].isna(), "item_id"].unique()[:5]
        raise ValueError(f"items missing from items.csv: {list(missing)}")
    seg["is_perishable"] = seg["is_perishable"].astype(bool)
    seg["is_slow"] = seg["avg_daily_units"] < cfg["forecast"]["slow_seller_max_avg_units"]
    cols = ["store_id", "item_id", "is_perishable", "is_slow", "used_line"]
    return seg[cols].sort_values(["store_id", "item_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def closed_mask(dates: pd.Series, cfg: dict) -> pd.Series:
    """True on the month-days listed in forecast.closed_dates."""
    md = pd.to_datetime(dates).dt.strftime("%m-%d")
    return md.isin(cfg["forecast"]["closed_dates"])


def _group_scores(d: pd.DataFrame) -> dict:
    y = d["units"].to_numpy()
    row = {"n": int(len(d)), "mae": mae(y, d["q50"])}
    for name, q in QUANTILES.items():
        row[f"pinball_{name}"] = pinball(y, d[name], q)
    for name in QUANTILES:
        row[f"cover_{name}"] = coverage(y, d[name])
    row["cover_used_line"] = coverage(y, d["used_value"])
    row["used_line_target"] = float(d["used_target"].mean() * 100)
    return row


def score(lines: pd.DataFrame, actual: pd.DataFrame, segments: pd.DataFrame,
          cfg: dict) -> pd.DataFrame:
    """Scores per group (all / perishable / slow), one row each.

    lines: date, store_id, item_id, q50..q95. actual: date, store_id, item_id, units.
    Only rows present in both are scored; closed days are dropped first.
    """
    d = lines.merge(actual[KEY + ["units"]], on=KEY, how="inner", validate="one_to_one")
    d = d[~closed_mask(d["date"], cfg)]
    d = d.merge(segments, on=["store_id", "item_id"], how="left", validate="many_to_one")
    if d["used_line"].isna().any():
        raise ValueError("some scored store-items have no segment row")
    lines_arr = d[list(QUANTILES)].to_numpy()
    col = d["used_line"].map({name: i for i, name in enumerate(QUANTILES)}).to_numpy()
    d["used_value"] = lines_arr[np.arange(len(d)), col]
    d["used_target"] = d["used_line"].map(QUANTILES)

    groups = {"all": d, "perishable": d[d["is_perishable"]], "slow": d[d["is_slow"]]}
    rows = [{"segment": name, **_group_scores(g)} for name, g in groups.items() if len(g)]
    return pd.DataFrame(rows)
