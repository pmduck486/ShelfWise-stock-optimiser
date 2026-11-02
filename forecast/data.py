"""Safe data access for the ML forecaster.

What this file does:
  - load_config: config.yaml with the forecast: paths made absolute (relative to the file);
    refuses a closed day that is not also dropped from training (decision 21).
  - load_items: items.csv (perishable flag, rank, average daily units), read only.
  - load_sales: sales.parquet for development. Refuses a file that reaches into the
    sealed holdout (rule R1) instead of trimming it quietly.
  - check_cutoff: raises if a frame holds any row on or after a model's cutoff (R1).
  - windows / val_folds: the time windows and the two validation folds, parsed and checked.
  - load_calendar: the M5 calendar for every target day, including days after sales end.
    Calendars are published in advance, so these are allowed same-day inputs (R3).
"""

from pathlib import Path

import pandas as pd

from data_prep.prepare import WINDOW_ORDER, check_windows
from data_prep.prepare import load_config as _prepare_load_config

# forecast: keys holding a path relative to config.yaml.
PATH_KEYS = ("calendar_file", "model_dir", "scores_dir")

ITEM_COLS = ["item_id", "dept_id", "rank", "avg_daily_units", "is_perishable", "shelf_life_days"]

CALENDAR_COLS = ["date", "wday", "month", "event_name", "is_event", "snap_CA", "snap_TX", "snap_WI"]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def load_config(path) -> dict:
    """Read config.yaml (via data_prep) and make the forecast: paths absolute.

    The data:, network: and stubs: sections are exactly what data_prep returns.
    Raises ValueError if a closed day is not also dropped from training (decision 21).
    """
    path = Path(path).resolve()
    cfg = _prepare_load_config(path)
    f = cfg["forecast"]
    for key in PATH_KEYS:
        p = Path(f[key])
        if not p.is_absolute():
            f[key] = str(path.parent / p)
    if int(f["horizon_days"]) > int(f["min_lag_days"]):
        raise ValueError(f"forecast.horizon_days ({f['horizon_days']}) must not exceed "
                         f"forecast.min_lag_days ({f['min_lag_days']}): rule R2")
    not_dropped = set(f["closed_dates"]) - set(f["drop_dates"])
    if not_dropped:
        raise ValueError(f"forecast.closed_dates {sorted(not_dropped)} must also be in "
                         f"forecast.drop_dates, or training and the 0 forecasts disagree")
    return cfg


def windows(cfg: dict) -> dict:
    """All window dates in data.windows, as Timestamps (checked for order first)."""
    check_windows(cfg)
    return {k: pd.Timestamp(cfg["data"]["windows"][k]) for k in WINDOW_ORDER}


def val_folds(cfg: dict) -> list:
    """The validation folds in time order, dates parsed. Each fold's model is trained on
    train_start .. train_end and scored on val_start .. val_end.

    Raises ValueError unless: train_end is the day before val_start, each fold ends before
    the next starts, every fold ends before test_start, and training starts inside the
    train window.
    """
    w = windows(cfg)
    folds = []
    for raw in cfg["forecast"]["val_folds"]:
        f = {"name": raw["name"]}
        for key in ("train_end", "val_start", "val_end"):
            f[key] = pd.Timestamp(raw[key])
        f["peak_days"] = [pd.Timestamp(d) for d in raw.get("peak_days", [])]
        folds.append(f)

    for f in folds:
        if f["train_end"] + pd.Timedelta(days=1) != f["val_start"]:
            raise ValueError(f"fold {f['name']}: train_end must be the day before val_start")
        if f["val_start"] > f["val_end"]:
            raise ValueError(f"fold {f['name']}: val_start is after val_end")
        if f["train_end"] < w["train_start"]:
            raise ValueError(f"fold {f['name']}: train_end is before train_start")
        if f["val_end"] >= w["test_start"]:
            raise ValueError(f"fold {f['name']}: val_end must be before test_start "
                             f"({w['test_start'].date()})")
        if any(not f["val_start"] <= d <= f["val_end"] for d in f["peak_days"]):
            raise ValueError(f"fold {f['name']}: a peak day lies outside the fold")
    for before, after in zip(folds, folds[1:]):
        if after["val_start"] <= before["val_end"]:
            raise ValueError(f"fold {after['name']} starts before fold {before['name']} ends")
    return folds


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def check_cutoff(sales: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    """Raise if any row is dated on or after `cutoff` (a model may only see earlier days)."""
    late = sales["date"] >= pd.Timestamp(cutoff)
    if late.any():
        raise ValueError(f"{int(late.sum())} rows on or after cutoff {pd.Timestamp(cutoff).date()} "
                         f"(latest {sales.loc[late, 'date'].max().date()})")


def load_sales(cfg: dict) -> pd.DataFrame:
    """sales.parquet for development. Raises if any row is on or after holdout_start."""
    sales = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "sales.parquet")
    holdout_start = pd.Timestamp(cfg["data"]["windows"]["holdout_start"])
    late = sales["date"] >= holdout_start
    if late.any():
        raise ValueError(f"sales.parquet has {int(late.sum())} rows in the sealed holdout "
                         f"(on or after {holdout_start.date()}); rebuild it without --include-holdout")
    return sales


def load_items(cfg: dict) -> pd.DataFrame:
    """items.csv (written by data_prep): one row per item with dept, rank, average daily
    units, perishable flag and shelf life. Read only."""
    items = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "items.csv",
                        dtype={"item_id": str, "dept_id": str})
    items["is_perishable"] = items["is_perishable"].astype(bool)
    items["shelf_life_days"] = items["shelf_life_days"].astype("Int64")
    return items[ITEM_COLS]


def load_calendar(cfg: dict) -> pd.DataFrame:
    """The M5 calendar, coded like sales.parquet (wday 1 = Saturday, event_name = event_name_1,
    "" when none). One row per day, through the last M5 day. Reads no sales (R3)."""
    cal = pd.read_csv(cfg["forecast"]["calendar_file"], parse_dates=["date"])
    out = pd.DataFrame({
        "date": cal["date"],
        "wday": cal["wday"].astype("int64"),
        "month": cal["month"].astype("int64"),
        "event_name": cal["event_name_1"].fillna("").astype(str),
    })
    out["is_event"] = out["event_name"] != ""
    for st in ("CA", "TX", "WI"):
        out[f"snap_{st}"] = cal[f"snap_{st}"] == 1
    return out[CALENDAR_COLS]
