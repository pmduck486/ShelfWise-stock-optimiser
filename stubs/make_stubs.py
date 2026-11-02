"""Stub ("dummy") inputs for the order model, so Lump 2 can start before the real forecaster.

THIS IS A PLACEHOLDER, NOT THE REAL FORECASTER. The real one (Lump 1, quantile gradient
boosting, plan/forecast-training-plan.md) will replace forecasts_stub.parquet with
forecasts.parquet in exactly the same columns. The stub only has to be the right shape
and obey the 7-day rule (R2), so the order model can be built and tested against it.

What this file writes into data/processed/:
  forecasts_stub.parquet          one origin per test day (56), each covering origin .. origin+6,
                                  all stores x items in sales.parquet
  forecasts_stub_small.parquet    the Gate 1 case: 2 stores x 10 items, the first 7 test days
  state_start_store_stock.csv     store stock on the morning of test_start
  state_start_dc_stock.csv        warehouse (DC) stock on the morning of test_start
  state_start_pipeline.csv        open orders on the way (empty at the start)
  ... and the same three state files with a _small suffix.

The stub forecast for target day D (every input is at least 7 days older than D):
  q50 = mean of units on D-7, D-14, D-21 and D-28
  sd  = standard deviation of daily units over D-34 .. D-7 (28 days)
  q80, q90, q95 = q50 + z * sd   (z = 0.84, 1.28, 1.645), clipped at 0, never crossing.
Days with no row (the item was not on sale yet) are skipped, not counted as zero.
All numbers come from the `stubs:` and `network:` sections of config.yaml.

Run it from the retailops folder:
    python -m stubs.make_stubs --config config.yaml
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_prep.prepare import load_config  # noqa: E402

FORECAST_COLS = ["origin", "date", "store_id", "item_id", "q50", "q80", "q90", "q95"]
STORE_STOCK_COLS = ["date", "store_id", "item_id", "units"]
DC_STOCK_COLS = ["date", "dc_id", "item_id", "units"]
PIPELINE_COLS = ["order_date", "arrival_date", "location_type", "location_id", "item_id", "units"]
QUANTILES = ["q80", "q90", "q95"]   # the upper lines; q50 is the middle line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_window(cfg: dict) -> tuple:
    w = cfg["data"]["windows"]
    return pd.Timestamp(w["test_start"]), pd.Timestamp(w["test_end"])


def check_network(cfg: dict, stores) -> None:
    """Every store must have a home DC, and that DC must be listed in network.dcs."""
    net = cfg["network"]
    missing = [s for s in stores if s not in net["home_dc"]]
    if missing:
        raise ValueError(f"stores with no home_dc in config.yaml network: {missing}")
    unknown = sorted({net["home_dc"][s] for s in stores} - set(net["dcs"]))
    if unknown:
        raise ValueError(f"home_dc names not listed in network.dcs: {unknown}")


def units_grid(sales: pd.DataFrame, first_day, last_day) -> pd.DataFrame:
    """Wide table: one row per (store_id, item_id), one column per day first_day..last_day.
    A missing row (item not on sale yet, or a day outside the data) is NaN."""
    days = pd.date_range(first_day, last_day, freq="D")
    wide = sales.pivot_table(index=["store_id", "item_id"], columns="date",
                             values="units", aggfunc="sum")
    wide.columns = pd.DatetimeIndex(wide.columns)
    return wide.reindex(columns=days).astype(float)


# ---------------------------------------------------------------------------
# Forecast stub
# ---------------------------------------------------------------------------
def forecast_targets(sales: pd.DataFrame, targets, cfg: dict) -> pd.DataFrame:
    """Stub quantiles for each target date (they depend only on the target date, since every
    input is >= 7 days old). Returns date, store_id, item_id, q50, q80, q90, q95."""
    st = cfg["stubs"]
    lags = [int(x) for x in st["lags"]]
    end_lag, width = int(st["spread_end_lag"]), int(st["spread_window_days"])
    if min(lags) < 7 or end_lag < 7:
        raise ValueError("rule R2: every lag must be at least 7 days")
    targets = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(targets))))
    reach = max(max(lags), end_lag + width - 1)
    first = targets.min() - pd.Timedelta(days=reach)
    last = targets.max() - pd.Timedelta(days=min(min(lags), end_lag))   # newest day we ever read
    grid = units_grid(sales, first, last)
    u = grid.to_numpy()                                   # [series, day]
    pos = ((targets - first).days).to_numpy()             # column of each target day (may be past `last`)

    lag_vals = np.stack([u[:, pos - L] for L in lags], axis=2)            # [series, target, lag]
    # spread window: target-(end_lag+width-1) .. target-end_lag
    offs = np.arange(end_lag, end_lag + width)
    win = np.stack([u[:, pos - o] for o in offs], axis=2)                 # [series, target, day]
    with warnings.catch_warnings():               # all-NaN rows (not on sale yet) are expected
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q50 = np.nanmean(lag_vals, axis=2)
        n = np.sum(~np.isnan(win), axis=2)
        sd = np.nanstd(win, axis=2, ddof=1)
    q50 = np.nan_to_num(q50, nan=0.0)
    sd = np.where(n >= 2, np.nan_to_num(sd, nan=0.0), 0.0)

    out = {"q50": np.clip(q50, 0, None)}
    prev = out["q50"]
    for q in QUANTILES:
        val = np.clip(q50 + float(st["z"][q]) * sd, 0, None)
        prev = np.maximum(prev, val)                      # lines never cross
        out[q] = prev

    n_series, n_t = q50.shape
    idx = grid.index
    df = pd.DataFrame({
        "date": np.tile(targets.to_numpy(), n_series),
        "store_id": np.repeat(idx.get_level_values("store_id").to_numpy(), n_t),
        "item_id": np.repeat(idx.get_level_values("item_id").to_numpy(), n_t),
    })
    for q in ["q50"] + QUANTILES:
        df[q] = out[q].reshape(-1).astype("float64")
    return df


def make_forecasts(sales: pd.DataFrame, origins, cfg: dict) -> pd.DataFrame:
    """Contract table forecasts.parquet: each origin covers origin .. origin+horizon-1."""
    h = int(cfg["stubs"]["horizon_days"])
    origins = pd.DatetimeIndex(origins)
    pairs = pd.DataFrame({
        "origin": np.repeat(origins.to_numpy(), h),
        "date": (np.repeat(origins, h) + pd.to_timedelta(np.tile(np.arange(h), len(origins)), unit="D")),
    })
    by_target = forecast_targets(sales, pairs["date"].unique(), cfg)
    fc = pairs.merge(by_target, on="date", how="left")
    fc = fc.sort_values(["origin", "date", "store_id", "item_id"]).reset_index(drop=True)
    fc["origin"] = pd.to_datetime(fc["origin"]).astype("datetime64[ns]")
    fc["date"] = pd.to_datetime(fc["date"]).astype("datetime64[ns]")
    return fc[FORECAST_COLS]


# ---------------------------------------------------------------------------
# Starting state
# ---------------------------------------------------------------------------
def start_state(sales: pd.DataFrame, cfg: dict) -> tuple:
    """Stock on the morning of test_start: (store_stock, dc_stock, pipeline)."""
    ss = cfg["stubs"]["start_state"]
    start, _ = test_window(cfg)
    width = int(ss["avg_window_days"])
    stores = sorted(sales["store_id"].unique())
    check_network(cfg, stores)

    grid = units_grid(sales, start - pd.Timedelta(days=width), start - pd.Timedelta(days=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nan_to_num(np.nanmean(grid.to_numpy(), axis=1), nan=0.0)
    means = pd.DataFrame({
        "store_id": grid.index.get_level_values("store_id"),
        "item_id": grid.index.get_level_values("item_id"),
        "mean": mean,
    })

    store_stock = means.assign(
        date=start, units=np.round(float(ss["store_stock_days"]) * means["mean"]).astype("int64"))
    store_stock = store_stock[STORE_STOCK_COLS].sort_values(["store_id", "item_id"]).reset_index(drop=True)

    means["dc_id"] = means["store_id"].map(cfg["network"]["home_dc"])
    dc = means.groupby(["dc_id", "item_id"], as_index=False)["mean"].sum()
    dc_stock = dc.assign(date=start,
                         units=np.round(float(ss["dc_stock_days"]) * dc["mean"]).astype("int64"))
    dc_stock = dc_stock[DC_STOCK_COLS].sort_values(["dc_id", "item_id"]).reset_index(drop=True)

    pipeline = pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        PIPELINE_COLS, ["datetime64[ns]", "datetime64[ns]", "object", "object", "object", "int64"])})
    return store_stock, dc_stock, pipeline


# ---------------------------------------------------------------------------
# Small (Gate 1) case
# ---------------------------------------------------------------------------
def small_case(sales: pd.DataFrame, items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sm = cfg["stubs"]["small"]
    top = items.sort_values("rank")["item_id"].head(int(sm["n_items"])).tolist()
    return sales[sales["store_id"].isin(sm["stores"]) & sales["item_id"].isin(top)]


def write_state(out_dir: Path, state: tuple, suffix: str) -> list:
    paths = []
    for name, df in zip(["store_stock", "dc_stock", "pipeline"], state):
        p = out_dir / f"state_start_{name}{suffix}.csv"
        out = df.copy()
        for c in ("date", "order_date", "arrival_date"):
            if c in out.columns:
                out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
        out.to_csv(p, index=False)
        paths.append((p, len(out)))
    return paths


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Write stub forecasts and starting state (PLACEHOLDERS).")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config)
    out_dir = Path(cfg["data"]["processed_dir"])
    sales = pd.read_parquet(out_dir / "sales.parquet", columns=["date", "store_id", "item_id", "units"])
    sales["date"] = pd.to_datetime(sales["date"])
    items = pd.read_csv(out_dir / "items.csv")
    check_network(cfg, sorted(sales["store_id"].unique()))

    start, end = test_window(cfg)
    origins = pd.date_range(start, end, freq="D")
    small_origins = origins[: int(cfg["stubs"]["small"]["days"])]
    small = small_case(sales, items, cfg)

    print("STUB forecasts (placeholder, not the real forecaster)")
    written = []
    for name, data, orig in [("forecasts_stub.parquet", sales, origins),
                             ("forecasts_stub_small.parquet", small, small_origins)]:
        fc = make_forecasts(data, orig, cfg)
        p = out_dir / name
        fc.to_parquet(p, index=False)
        written.append((p, len(fc)))
        print(f"  {name}: {len(fc):,} rows, {fc['origin'].nunique()} origins, "
              f"{fc['store_id'].nunique()} stores x {fc['item_id'].nunique()} items, "
              f"target dates {fc['date'].min().date()} to {fc['date'].max().date()}")

    print(f"Starting state for {start.date()}")
    written += write_state(out_dir, start_state(sales, cfg), "")
    written += write_state(out_dir, start_state(small, cfg), "_small")
    for p, n in written:
        print(f"  {p.name:36s} {n:>9,} rows  {p.stat().st_size / 1024:8.1f} KB")
    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
