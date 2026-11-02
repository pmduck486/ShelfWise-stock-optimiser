"""Prepare the M5 sales data for the RetailOps stock-ordering project.

What this file does, step by step:
  0. Check the time windows in config.yaml (in order, no overlap, inside M5).
  1. Read the M5 calendar and the wide sales file (one column per day), keeping
     only the days from `history_start` to the day before `holdout_start`.
     The holdout is sealed: it is read only with --include-holdout (the one
     final run), which writes sales_full.parquet instead of sales.parquet.
  2. Keep only FOODS items in the chosen stores, then pick the product set
     (top sellers for `main`, a seeded random sample for `random`).
     Rule R5: the choice uses ONLY sales before `test_start`, so it can't favour
     items that happened to sell well in the test or holdout weeks.
  3. Flag some items as perishable and give them a shelf life (an assumption).
  4. "Melt" the wide table into a long one: one row per store, item and day.
  5. Add the weekly price, the calendar events and the SNAP (food benefit) flag.
  6. Drop days before a product first had a price (not on sale yet) and fill
     any later price gaps with the last known price.

Run it from the retailops folder:
    python -m data_prep.prepare --config config.yaml
    python -m data_prep.prepare --config config.yaml --product-set random --seed 3
    python -m data_prep.prepare --config config.yaml --include-holdout   # final run only
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Columns at the start of the wide M5 sales file (before d_1, d_2, ...).
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Final column order of the sales table (the data contract other lumps rely on).
SALES_COLS = [
    "date", "store_id", "item_id", "dept_id", "units", "price",
    "is_perishable", "shelf_life_days", "wday", "month",
    "event_name", "is_event", "snap",
]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def load_config(path) -> dict:
    """Read config.yaml. Folder paths in it are relative to the config file's folder."""
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("m5_dir", "processed_dir"):
        folder = Path(cfg["data"][key])
        if not folder.is_absolute():
            cfg["data"][key] = str(path.parent / folder)
    return cfg


# Window dates in time order: each start <= its end, and each window starts after the last one ends.
WINDOW_ORDER = ["warmup_start", "train_start", "train_end", "val_start", "val_end",
                "test_start", "test_end", "holdout_start", "holdout_end"]


def check_windows(cfg: dict) -> None:
    """Raise a ValueError if the time windows are out of order, overlap or leave the M5 range."""
    d = cfg["data"]
    w = {k: pd.Timestamp(d["windows"][k]) for k in WINDOW_ORDER}
    first, last = pd.Timestamp(d["m5_first_day"]), pd.Timestamp(d["m5_last_day"])
    history_start = pd.Timestamp(d["history_start"])

    if history_start < first:
        raise ValueError(f"history_start {history_start.date()} is before the first M5 day {first.date()}")
    if history_start > w["warmup_start"]:
        raise ValueError(f"history_start {history_start.date()} is after warmup_start "
                         f"{w['warmup_start'].date()}: the warm-up days would not be read")
    if w["warmup_start"] >= w["train_start"]:
        raise ValueError("warmup_start must come before train_start")
    for name in ("train", "val", "test", "holdout"):            # each window: start <= end
        if w[f"{name}_start"] > w[f"{name}_end"]:
            raise ValueError(f"{name}_start is after {name}_end")
    for before, after in (("train", "val"), ("val", "test"), ("test", "holdout")):   # no overlap
        if w[f"{after}_start"] <= w[f"{before}_end"]:
            raise ValueError(f"{after}_start ({w[f'{after}_start'].date()}) must be after "
                             f"{before}_end ({w[f'{before}_end'].date()}): the windows overlap")
    if w["holdout_end"] > last:
        raise ValueError(f"holdout_end {w['holdout_end'].date()} is after the last M5 day {last.date()}")


def selection_end(cfg: dict) -> pd.Timestamp:
    """Last day whose sales may be used to choose products: the day before test_start (rule R5)."""
    return pd.Timestamp(cfg["data"]["windows"]["test_start"]) - pd.Timedelta(days=1)


def read_end(cfg: dict, include_holdout: bool) -> pd.Timestamp:
    """Last day read from the sales file: the day before the sealed holdout, unless the
    final holdout run opts in, in which case the whole holdout is read too."""
    w = cfg["data"]["windows"]
    if include_holdout:
        return pd.Timestamp(w["holdout_end"])
    return pd.Timestamp(w["holdout_start"]) - pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Reading the raw files
# ---------------------------------------------------------------------------
def read_calendar(m5_dir) -> pd.DataFrame:
    """The M5 calendar: one row per day (d_1, d_2, ...) with date, week, events, SNAP."""
    cal = pd.read_csv(Path(m5_dir) / "calendar.csv", parse_dates=["date"])
    cal["event_name_1"] = cal["event_name_1"].fillna("")
    return cal


def window_day_columns(cal: pd.DataFrame, available_cols, first_day, last_day) -> list:
    """Day columns (d_...) dated first_day..last_day (both included) that exist in available_cols."""
    available = set(available_cols)
    in_window = cal[(cal["date"] >= pd.Timestamp(first_day)) & (cal["date"] <= pd.Timestamp(last_day))]
    return [d for d in in_window["d"] if d in available]


def read_sales_wide(cfg: dict, cal: pd.DataFrame, include_holdout: bool = False) -> pd.DataFrame:
    """Read the wide sales file once, keeping only the history window (cut before the
    sealed holdout unless include_holdout), the chosen category and the configured stores."""
    d = cfg["data"]
    path = Path(d["m5_dir"]) / "sales_train_evaluation.csv"
    header = pd.read_csv(path, nrows=0).columns          # just the column names
    day_cols = window_day_columns(cal, header, d["history_start"], read_end(cfg, include_holdout))
    dtypes = {c: "int32" for c in day_cols}
    wide = pd.read_csv(path, usecols=ID_COLS + day_cols, dtype=dtypes)
    keep = (wide["cat_id"] == d["category"]) & (wide["store_id"].isin(d["stores"]))
    return wide[keep].reset_index(drop=True)


def read_prices(m5_dir, stores, items) -> pd.DataFrame:
    """Weekly prices (store, item, week) for just the chosen stores and items."""
    prices = pd.read_csv(
        Path(m5_dir) / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category",
               "wm_yr_wk": "int32", "sell_price": "float64"},
    )
    keep = prices["store_id"].isin(stores) & prices["item_id"].isin(items)
    prices = prices[keep].rename(columns={"sell_price": "price"})
    prices["store_id"] = prices["store_id"].astype(str)
    prices["item_id"] = prices["item_id"].astype(str)
    return prices.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Choosing products
# ---------------------------------------------------------------------------
def day_columns(sales_wide: pd.DataFrame) -> list:
    """The d_... columns of a wide sales table."""
    return [c for c in sales_wide.columns if c.startswith("d_")]


def selection_columns(sales_wide: pd.DataFrame, cal: pd.DataFrame, cfg: dict) -> list:
    """The day columns product choice may look at: history_start .. the day before test_start."""
    return window_day_columns(cal, day_columns(sales_wide), cfg["data"]["history_start"], selection_end(cfg))


def item_stats(sales_wide: pd.DataFrame, stores, days) -> pd.DataFrame:
    """Total units and average daily units per item over the chosen stores and the given days.

    avg_daily_units = total units / (number of stores x number of days), i.e. the
    average sold per store per day over those days.
    Sorted biggest seller first (ties broken by item_id so the order never changes).
    """
    rows = sales_wide[sales_wide["store_id"].isin(stores)]
    totals = rows[days].sum(axis=1)
    stats = (
        pd.DataFrame({"item_id": rows["item_id"], "dept_id": rows["dept_id"], "total_units": totals})
        .groupby(["item_id", "dept_id"], as_index=False)["total_units"].sum()
    )
    stats["avg_daily_units"] = stats["total_units"] / (len(stores) * len(days))
    stats = stats.sort_values(["total_units", "item_id"], ascending=[False, True])
    return stats.reset_index(drop=True)


def select_products(sales_wide: pd.DataFrame, cal: pd.DataFrame, cfg: dict) -> tuple:
    """Pick the stores and items to study. Returns (stores, items).

    Only sales from history_start to the day before test_start are looked at (rule R5),
    whatever later days `sales_wide` also holds.
      - main:   all configured stores; the top `n_items` items by total units.
      - random: the seed picks `n_stores` stores, then `n_items` items among those
                selling at least `min_avg_daily_units` per store per day.
    Items come back biggest seller first. Same config and seed => same answer.
    """
    d = cfg["data"]
    foods = sales_wide[sales_wide["cat_id"] == d["category"]]
    days = selection_columns(foods, cal, cfg)

    if d["product_set"] == "main":
        stores = list(d["stores"])
        stats = item_stats(foods, stores, days)
        items = stats["item_id"].head(d["main"]["n_items"]).tolist()

    elif d["product_set"] == "random":
        r = d["random"]
        rng = np.random.default_rng(d["seed"])
        # Sort before sampling so the result depends only on the seed, not on file order.
        stores = sorted(rng.choice(sorted(d["stores"]), size=r["n_stores"], replace=False).tolist())
        stats = item_stats(foods, stores, days)
        eligible = sorted(stats.loc[stats["avg_daily_units"] >= r["min_avg_daily_units"], "item_id"])
        if len(eligible) < r["n_items"]:
            raise ValueError(f"Only {len(eligible)} items meet min_avg_daily_units; need {r['n_items']}.")
        picked = set(rng.choice(eligible, size=r["n_items"], replace=False).tolist())
        items = [i for i in stats["item_id"] if i in picked]   # keep biggest-seller-first order

    else:
        raise ValueError(f"product_set must be 'main' or 'random', not {d['product_set']!r}")

    return stores, items


def flag_perishables(items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Mark the fastest-selling items of the perishable department as perishable.

    ASSUMPTION (see config.yaml): M5 items are anonymous, so 'top FOODS_3 sellers'
    stands in for fresh goods. Shelf lives cycle through shelf_life_cycle_days in rank order.
    """
    d = cfg["data"]
    n = d[d["product_set"]]["n_perishable"]            # main or random block
    cycle = d["shelf_life_cycle_days"]

    candidates = items[items["dept_id"] == d["perishable_dept"]]
    candidates = candidates.sort_values(["avg_daily_units", "item_id"], ascending=[False, True])
    chosen = candidates["item_id"].head(n).tolist()

    items = items.copy()
    items["is_perishable"] = items["item_id"].isin(chosen)
    shelf = {item: cycle[i % len(cycle)] for i, item in enumerate(chosen)}
    items["shelf_life_days"] = items["item_id"].map(shelf).astype("Int64")   # <NA> if not perishable
    return items


# ---------------------------------------------------------------------------
# Building the long daily table
# ---------------------------------------------------------------------------
def build_sales_table(cfg: dict, verbose: bool = False, include_holdout: bool = False) -> tuple:
    """Build (sales, items) as described at the top of this file.

    include_holdout=False (the default): sales stop the day before holdout_start (sealed).
    include_holdout=True: sales run to holdout_end, for the one final holdout run only.
    The items table is the same either way, because it only uses pre-test sales.
    """
    check_windows(cfg)
    d = cfg["data"]
    cal = read_calendar(d["m5_dir"])
    wide = read_sales_wide(cfg, cal, include_holdout)

    # Choose the products, then shrink the wide table BEFORE melting it.
    stores, item_ids = select_products(wide, cal, cfg)
    wide = wide[wide["store_id"].isin(stores) & wide["item_id"].isin(item_ids)]

    # Item table: rank by volume, average daily units, perishable flag + shelf life.
    # These stats are the SAME pre-test stats used to choose the items (rule R5), so
    # rank and avg_daily_units cover history_start .. the day before test_start only.
    items = item_stats(wide, stores, selection_columns(wide, cal, cfg))
    items.insert(2, "rank", range(1, len(items) + 1))
    items = flag_perishables(items, cfg)
    items = items[["item_id", "dept_id", "rank", "avg_daily_units", "is_perishable", "shelf_life_days"]]

    # Wide -> long: one row per store, item and day.
    days = day_columns(wide)
    sales = wide.melt(id_vars=["store_id", "item_id", "dept_id", "state_id"],
                      value_vars=days, var_name="d", value_name="units")

    # Calendar: date, week number (to join prices), weekday, month, event, SNAP flags.
    cal_cols = ["d", "date", "wm_yr_wk", "wday", "month", "event_name_1",
                "snap_CA", "snap_TX", "snap_WI"]
    sales = sales.merge(cal[cal_cols], on="d", how="left")
    sales["event_name"] = sales["event_name_1"]
    sales["is_event"] = sales["event_name"] != ""
    # SNAP days differ by state: take the flag of the store's own state.
    sales["snap"] = False
    for state in ("CA", "TX", "WI"):
        in_state = sales["state_id"] == state
        sales.loc[in_state, "snap"] = sales.loc[in_state, f"snap_{state}"] == 1

    # Weekly price -> every day of that week.
    prices = read_prices(d["m5_dir"], stores, item_ids)
    sales = sales.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    sales = sales.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # Not on sale yet: drop days before an item-store's first price.
    group = [sales["store_id"], sales["item_id"]]
    on_sale = sales["price"].notna().groupby(group).cummax()
    n_rows_before = len(sales)
    units_dropped = int(sales.loc[~on_sale, "units"].sum())
    sales = sales[on_sale].reset_index(drop=True)

    # Later gaps: carry the last known price forward (within the same item-store).
    n_gaps = int(sales["price"].isna().sum())
    sales["price"] = sales.groupby(["store_id", "item_id"])["price"].ffill()

    # Add the perishable flag and shelf life from the item table.
    sales = sales.merge(items[["item_id", "is_perishable", "shelf_life_days"]], on="item_id", how="left")

    # Final types and column order.
    sales["units"] = sales["units"].astype("int32")
    sales["snap"] = sales["snap"].astype(bool)
    sales = sales[SALES_COLS]

    if verbose:
        print(f"  dropped {n_rows_before - len(sales):,} pre-launch rows "
              f"(units sold on them: {units_dropped:,})")
        print(f"  filled {n_gaps:,} daily price gaps with the last known price")
    return sales, items


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def output_names(cfg: dict, include_holdout: bool = False) -> tuple:
    """File names for the chosen product set. The full series (with the holdout) goes to
    a separate sales_full... file, so the everyday sales file never holds holdout rows."""
    d = cfg["data"]
    sales = "sales_full" if include_holdout else "sales"
    if d["product_set"] == "main":
        return f"{sales}.parquet", "items.csv"
    return f"{sales}_random_seed{d['seed']}.parquet", f"items_random_seed{d['seed']}.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare the M5 sales table for RetailOps.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--product-set", choices=["main", "random"], help="override data.product_set")
    parser.add_argument("--seed", type=int, help="override data.seed (used when product set is random)")
    parser.add_argument("--include-holdout", action="store_true",
                        help="FINAL HOLDOUT RUN ONLY: also read the sealed holdout and write sales_full.parquet")
    args = parser.parse_args(argv)

    start = time.time()
    cfg = load_config(args.config)
    if args.product_set:
        cfg["data"]["product_set"] = args.product_set
    if args.seed is not None:
        cfg["data"]["seed"] = args.seed
    check_windows(cfg)

    d = cfg["data"]
    label = d["product_set"] + (f" (seed {d['seed']})" if d["product_set"] == "random" else "")
    print(f"Preparing product set: {label}")
    print(f"  products chosen on sales {d['history_start']} to {selection_end(cfg).date()} (before test_start)")
    if args.include_holdout:
        print("  WARNING: --include-holdout reads the sealed holdout. Use it only for the one final run.")
    else:
        print(f"  holdout sealed: no rows on or after {d['windows']['holdout_start']}")
    sales, items = build_sales_table(cfg, verbose=True, include_holdout=args.include_holdout)

    out_dir = Path(d["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sales_name, items_name = output_names(cfg, args.include_holdout)
    sales_path, items_path = out_dir / sales_name, out_dir / items_name
    sales.to_parquet(sales_path, index=False)
    items.to_csv(items_path, index=False)

    zero_share = (sales["units"] == 0).mean()
    print("Summary")
    print(f"  stores      : {sorted(sales['store_id'].unique())}")
    print(f"  items       : {len(items)} ({int(items['is_perishable'].sum())} perishable)")
    print(f"  date range  : {sales['date'].min().date()} to {sales['date'].max().date()}")
    print(f"  rows        : {len(sales):,}")
    print(f"  total units : {int(sales['units'].sum()):,}  (zero-sales days: {zero_share:.1%})")
    print(f"  wrote       : {sales_path.name} ({sales_path.stat().st_size / 1e6:.1f} MB), "
          f"{items_path.name} ({items_path.stat().st_size / 1e3:.1f} KB)")
    print(f"  run time    : {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
