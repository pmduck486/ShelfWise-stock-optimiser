"""Checks on the trained model (Task 5; re-check strategies 3 and 5, approved 2026-10-30).
Everything here reports; nothing changes the model.

  - compare_lines: q80/q90/q95 pinball of the model against the naive and stub baselines on
    each fold. No product is stocked to q50, so the lines that drive orders must beat the
    simple rules too. A losing line is flagged to the user.
  - item_traits / worst_items: the store-items with the largest q50 error relative to their
    own average sales, next to the stub's error on the same rows and their traits (perishable,
    late launch, long zero runs, odd prices). Closed days are left out.
  - shuffle_importance: scramble one input at a time on validation rows and measure how much
    the pinball rises. Shows what the model relies on; an input that matters far more than it
    should hints at a leak.
  - leak_check: a q50 MAE below forecast.review.leak_mae_floor on the leak fold means the
    model almost surely sees sales newer than min_lag_days; work stops for diagnosis.
"""

import numpy as np
import pandas as pd

from forecast.features import KEY, PAIR, zero_run_stats
from forecast.metrics import QUANTILES, closed_mask, pinball

UPPER = ["q80", "q90", "q95"]
MIN_MEAN_UNITS = 0.5        # relative error divides by at least this (a store-item with no sales)


def _all_row(records: list) -> dict:
    return next(r for r in records if r["segment"] == "all")


def compare_lines(model_scores: dict, naive_scores: dict, stub_scores: dict) -> pd.DataFrame:
    """One row per fold x upper line: pinball of model, naive and stub, and who wins.
    Each argument maps fold name -> that fold's score records (as in <method>_val.json)."""
    rows = []
    for fold, records in model_scores.items():
        m, n, s = _all_row(records), _all_row(naive_scores[fold]), _all_row(stub_scores[fold])
        for line in UPPER:
            key = f"pinball_{line}"
            rows.append({"fold": fold, "line": line, "model": m[key], "naive": n[key],
                         "stub": s[key], "beats_naive": m[key] < n[key], "beats_stub": m[key] < s[key]})
    return pd.DataFrame(rows)


def item_traits(sales: pd.DataFrame, items: pd.DataFrame, cfg: dict, *, cutoff) -> pd.DataFrame:
    """One row per store-item, from rows before `cutoff` only: perishable flag and average daily
    units (items.csv), first sales day, late launch (first day after the data's first day),
    number of long zero-sales runs, and rows priced below half the store-item's median."""
    past = sales[sales["date"] < pd.Timestamp(cutoff)]
    g = past.groupby(PAIR)
    t = g["date"].min().rename("first_day").reset_index()
    t["late_launch"] = t["first_day"] > past["date"].min()
    median = g["price"].transform("median")
    odd = (past["price"] < 0.5 * median).groupby([past["store_id"], past["item_id"]]).sum()
    t = t.merge(odd.rename("odd_price_rows").reset_index(), on=PAIR, how="left")
    runs = zero_run_stats(sales, cfg, cutoff=cutoff)["runs"]
    counts = runs.groupby(PAIR).size().rename("long_zero_runs").reset_index()
    t = t.merge(counts, on=PAIR, how="left")
    t["long_zero_runs"] = t["long_zero_runs"].fillna(0).astype(int)
    info = items[["item_id", "is_perishable", "avg_daily_units"]]
    return t.merge(info, on="item_id", how="left")


def worst_items(lines: pd.DataFrame, actual: pd.DataFrame, stub_lines: pd.DataFrame,
                traits: pd.DataFrame, cfg: dict, n: int = 20) -> pd.DataFrame:
    """The n store-items with the largest q50 MAE / mean units (closed days left out)."""
    d = lines[KEY + ["q50"]].merge(actual[KEY + ["units"]], on=KEY, how="inner")
    d = d.merge(stub_lines[KEY + ["q50"]].rename(columns={"q50": "stub_q50"}), on=KEY, how="left")
    d = d[~closed_mask(d["date"], cfg).to_numpy()]
    d["err"] = (d["units"] - d["q50"]).abs()
    d["stub_err"] = (d["units"] - d["stub_q50"]).abs()
    g = d.groupby(PAIR).agg(mae=("err", "mean"), mean_units=("units", "mean"),
                            stub_mae=("stub_err", "mean"), days=("err", "size")).reset_index()
    g["rel_error"] = g["mae"] / g["mean_units"].clip(lower=MIN_MEAN_UNITS)
    g = g.sort_values(["rel_error", "mae"], ascending=False).head(n)
    return g.merge(traits, on=PAIR, how="left").reset_index(drop=True)


def shuffle_importance(model, X: pd.DataFrame, y, q: float, n_repeats: int = 3,
                       seed: int = 0) -> pd.DataFrame:
    """Rise in pinball when each input is scrambled (mean and spread over n_repeats), largest
    first. `base` is the unscrambled score."""
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    base = pinball(y, model.predict(X), q)
    rows = []
    for col in X.columns:
        rises = []
        for _ in range(n_repeats):
            Xs = X.copy()
            Xs[col] = Xs[col].to_numpy()[rng.permutation(len(Xs))]
            rises.append(pinball(y, model.predict(Xs), q) - base)
        rows.append({"input": col, "rise": float(np.mean(rises)), "rise_spread": float(np.ptp(rises)),
                     "base": base})
    return pd.DataFrame(rows).sort_values("rise", ascending=False).reset_index(drop=True)


def leak_check(fold_mae: dict, cfg: dict):
    """A message if the leak fold's q50 MAE is suspiciously low, else None."""
    r = cfg["forecast"]["review"]
    value = fold_mae[r["leak_fold"]]
    if value < r["leak_mae_floor"]:
        return (f"suspected leak: {r['leak_fold']} q50 MAE {value:.3f} is below "
                f"{r['leak_mae_floor']} (a model allowed yesterday's sales scored about 3.62)")
    return None


def group_table(d: pd.DataFrame, groups: list) -> pd.DataFrame:
    """Pinball of the model and the stub per fold x group x line (re-check strategy 8).
    d: one row per scored store-item day with fold, units, q50..q95, stub_q50..stub_q95 and
    the group columns."""
    rows = []
    for by in groups:
        for (fold, value), g in d.groupby(["fold", by], observed=True):
            for line, q in QUANTILES.items():
                rows.append({"fold": fold, "by": by, "group": str(value), "line": line, "n": len(g),
                             "model": pinball(g["units"], g[line], q),
                             "stub": pinball(g["units"], g[f"stub_{line}"], q)})
    return pd.DataFrame(rows)


def group_check(table: pd.DataFrame) -> pd.DataFrame:
    """Groups x lines where the model's pinball is not below the stub's on every fold. Only
    these would get the baseline fallback; an empty result means nothing to build."""
    t = table.assign(loses=table["model"] >= table["stub"])
    both = t.groupby(["by", "group", "line"], sort=False)["loses"].all()
    return both[both].reset_index()[["by", "group", "line"]]
