"""Tests for data_prep.prepare.

Most tests use a TINY fake M5 dataset written to a temporary folder, so they run in
about a second. One integration test uses the real M5 files and is skipped if they
are missing.

Fake data: 2 stores (CA_1, TX_1), 42 days of sales starting Sat 2013-12-28
(day index t = 0..41). The time windows are scaled down to fit:
  history/warm-up t 7..13, train t 14..20, validation t 21..28,
  test t 29..34, holdout t 35..41 (sealed).
So products are chosen on t 7..28 (22 days), and the default sales table holds
t 7..34 (28 days); the opt-in full table holds t 7..41.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from data_prep.prepare import (  # noqa: E402
    build_sales_table, check_windows, day_columns, item_stats, load_config, main,
    read_calendar, read_sales_wide, select_products, selection_end,
)

N_DAYS = 42                       # days in the fake sales file
FIRST_DAY = "2013-12-28"          # a Saturday, like the real M5 week start
HISTORY_START = "2014-01-04"      # day index 7
STORES = ["CA_1", "TX_1"]


def day(t):
    """Date of day index t in the fake data."""
    return pd.Timestamp(FIRST_DAY) + pd.Timedelta(days=t)


def iso(t):
    return day(t).strftime("%Y-%m-%d")


WINDOWS = {
    "warmup_start": iso(7),
    "train_start": iso(14), "train_end": iso(20),
    "val_start": iso(21), "val_end": iso(28),
    "test_start": iso(29), "test_end": iso(34),
    "holdout_start": iso(35), "holdout_end": iso(41),
}
TEST_T, HOLDOUT_T = 29, 35        # first test day and first holdout day (day index)

# item_id -> units per day (a function of the day index)
ITEMS = {
    "FOODS_3_001": lambda t: 10,                        # top food seller
    "FOODS_3_002": lambda t: 6 if t >= 21 else 0,       # launches on day 21 (week 3)
    "FOODS_3_003": lambda t: 4,
    "FOODS_1_001": lambda t: 8,                         # has a price gap at CA_1 (week 2)
    "FOODS_1_002": lambda t: 2,
    "FOODS_2_001": lambda t: 1 if t % 5 == 0 else 0,    # almost dead: 0.2 units/day
    "HOBBIES_1_001": lambda t: 50,                      # biggest seller but NOT food
}
LAUNCH_ITEM, LAUNCH_WEEK = "FOODS_3_002", 3
GAP_STORE, GAP_ITEM, GAP_WEEK = "CA_1", "FOODS_1_001", 2


def week_price(week):
    """Fake price that changes every week, so a forward-fill is easy to spot."""
    return round(1.0 + 0.1 * week, 2)


@pytest.fixture
def m5_dir(tmp_path):
    """Write calendar.csv, sales_train_evaluation.csv and sell_prices.csv in M5 shape."""
    folder = tmp_path / "m5"
    folder.mkdir()

    # Calendar has 2 more days than the sales file (like the real M5 calendar).
    dates = pd.date_range(FIRST_DAY, periods=N_DAYS + 2)
    cal = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "wm_yr_wk": [11400 + i // 7 for i in range(len(dates))],
        "weekday": dates.day_name(),
        "wday": [i % 7 + 1 for i in range(len(dates))],
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{i + 1}" for i in range(len(dates))],
        "event_name_1": "", "event_type_1": "", "event_name_2": "", "event_type_2": "",
        "snap_CA": [1 if i % 3 == 0 else 0 for i in range(len(dates))],
        "snap_TX": [1 if i % 4 == 0 else 0 for i in range(len(dates))],
        "snap_WI": 0,
    })
    cal.loc[10, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]
    cal.to_csv(folder / "calendar.csv", index=False)

    rows, prices = [], []
    for store in STORES:
        for item, units_fn in ITEMS.items():
            dept = item.rsplit("_", 1)[0]
            row = {"id": f"{item}_{store}_evaluation", "item_id": item, "dept_id": dept,
                   "cat_id": dept.split("_")[0], "store_id": store, "state_id": store[:2]}
            row.update({f"d_{t + 1}": units_fn(t) for t in range(N_DAYS)})
            rows.append(row)
            for week in range(N_DAYS // 7):
                if item == LAUNCH_ITEM and week < LAUNCH_WEEK:
                    continue                                    # not on sale yet
                if (store, item, week) == (GAP_STORE, GAP_ITEM, GAP_WEEK):
                    continue                                    # missing price week
                prices.append({"store_id": store, "item_id": item,
                               "wm_yr_wk": 11400 + week, "sell_price": week_price(week)})
    pd.DataFrame(rows).to_csv(folder / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(prices).to_csv(folder / "sell_prices.csv", index=False)
    return folder


def make_cfg(m5_dir, product_set="main", seed=1):
    """A config like config.yaml, scaled down to the fake data."""
    return {"data": {
        "m5_dir": str(m5_dir), "processed_dir": str(Path(m5_dir).parent / "processed"),
        "category": "FOODS", "stores": list(STORES), "history_start": HISTORY_START,
        "m5_first_day": "2011-01-29", "m5_last_day": "2016-05-22",
        "windows": dict(WINDOWS),
        "product_set": product_set, "seed": seed,
        "perishable_dept": "FOODS_3", "shelf_life_cycle_days": [2, 3, 4, 5, 7],
        "main": {"n_items": 5, "n_perishable": 2},
        "random": {"n_stores": 1, "n_items": 3, "n_perishable": 1,
                   "seeds": [1, 2, 3, 4, 5], "min_avg_daily_units": 0.5},
    }}


# ---------------------------------------------------------------------------
# Main product set on the fake data
# ---------------------------------------------------------------------------
def test_one_row_per_store_item_date(m5_dir):
    sales, items = build_sales_table(make_cfg(m5_dir))
    assert not sales.duplicated(["store_id", "item_id", "date"]).any()
    counts = sales.groupby(["store_id", "item_id"]).size()
    for (store, item), n in counts.items():
        expected = 14 if item == LAUNCH_ITEM else 28      # launch item: days 21..34 only
        assert n == expected, (store, item, n)
    assert len(counts) == len(STORES) * 5                 # every chosen item in every store


def test_only_top_food_items_kept(m5_dir):
    sales, items = build_sales_table(make_cfg(m5_dir))
    assert set(sales["item_id"]) == {"FOODS_3_001", "FOODS_1_001", "FOODS_3_003",
                                     "FOODS_3_002", "FOODS_1_002"}
    assert sales["item_id"].str.startswith("FOODS").all()  # HOBBIES item dropped
    assert list(items["rank"]) == [1, 2, 3, 4, 5]
    assert items.iloc[0]["item_id"] == "FOODS_3_001"


def test_dates_and_units(m5_dir):
    sales, _ = build_sales_table(make_cfg(m5_dir))
    assert sales["date"].min() == pd.Timestamp(HISTORY_START)
    assert sales["date"].max() == day(HOLDOUT_T - 1)          # stops before the sealed holdout
    assert pd.api.types.is_integer_dtype(sales["units"])
    assert (sales["units"] >= 0).all()


def test_no_rows_before_first_price(m5_dir):
    sales, _ = build_sales_table(make_cfg(m5_dir))
    launch = sales[sales["item_id"] == LAUNCH_ITEM]
    first_sale_day = pd.Timestamp(FIRST_DAY) + pd.Timedelta(days=7 * LAUNCH_WEEK)
    assert launch["date"].min() == first_sale_day
    assert sales["price"].notna().all()


def test_price_gap_is_forward_filled(m5_dir):
    sales, _ = build_sales_table(make_cfg(m5_dir))
    gap = sales[(sales["store_id"] == GAP_STORE) & (sales["item_id"] == GAP_ITEM)]
    gap_start = pd.Timestamp(FIRST_DAY) + pd.Timedelta(days=7 * GAP_WEEK)
    in_gap = gap[(gap["date"] >= gap_start) & (gap["date"] < gap_start + pd.Timedelta(days=7))]
    assert len(in_gap) == 7
    assert (in_gap["price"] == week_price(GAP_WEEK - 1)).all()   # last known price carried on
    other_store = sales[(sales["store_id"] == "TX_1") & (sales["item_id"] == GAP_ITEM)]
    assert (other_store.loc[other_store["date"] == gap_start, "price"] == week_price(GAP_WEEK)).all()


def test_perishables(m5_dir):
    cfg = make_cfg(m5_dir)
    sales, items = build_sales_table(cfg)
    per = items[items["is_perishable"]]
    assert len(per) == cfg["data"]["main"]["n_perishable"]
    assert (per["dept_id"] == "FOODS_3").all()
    assert list(per["item_id"]) == ["FOODS_3_001", "FOODS_3_003"]   # fastest FOODS_3 sellers
    assert list(per["shelf_life_days"]) == [2, 3]                   # cycle order
    assert set(per["shelf_life_days"]) <= set(cfg["data"]["shelf_life_cycle_days"])
    assert items.loc[~items["is_perishable"], "shelf_life_days"].isna().all()
    # The sales table carries the same flag.
    assert set(sales.loc[sales["is_perishable"], "item_id"]) == set(per["item_id"])


def test_calendar_columns(m5_dir):
    sales, _ = build_sales_table(make_cfg(m5_dir))
    superbowl = pd.Timestamp(FIRST_DAY) + pd.Timedelta(days=10)
    on_day = sales[sales["date"] == superbowl]
    assert (on_day["event_name"] == "SuperBowl").all() and on_day["is_event"].all()
    assert not sales.loc[sales["date"] != superbowl, "is_event"].any()
    # SNAP uses the store's own state: CA every 3rd day, TX every 4th day.
    day_index = (sales["date"] - pd.Timestamp(FIRST_DAY)).dt.days
    expected = np.where(sales["store_id"] == "CA_1", day_index % 3 == 0, day_index % 4 == 0)
    assert (sales["snap"] == expected).all()


# ---------------------------------------------------------------------------
# Random product sets
# ---------------------------------------------------------------------------
def wide_table(m5_dir, cfg):
    return read_sales_wide(cfg, read_calendar(m5_dir))


def test_same_seed_same_products(m5_dir):
    cfg = make_cfg(m5_dir, product_set="random", seed=3)
    wide = wide_table(m5_dir, cfg)
    cal = read_calendar(m5_dir)
    assert select_products(wide, cal, cfg) == select_products(wide, cal, copy.deepcopy(cfg))


def test_different_seeds_usually_differ(m5_dir):
    picks = set()
    for seed in [1, 2, 3, 4, 5]:
        cfg = make_cfg(m5_dir, product_set="random", seed=seed)
        stores, items = select_products(wide_table(m5_dir, cfg), read_calendar(m5_dir), cfg)
        picks.add((tuple(stores), tuple(items)))
    assert len(picks) >= 2


def test_random_set_respects_sizes_and_minimum(m5_dir):
    for seed in range(1, 11):
        cfg = make_cfg(m5_dir, product_set="random", seed=seed)
        r = cfg["data"]["random"]
        sales, items = build_sales_table(cfg)
        assert sales["store_id"].nunique() == r["n_stores"]
        assert len(items) == r["n_items"]
        assert "FOODS_2_001" not in set(items["item_id"])     # below min_avg_daily_units
        assert (items["avg_daily_units"] >= r["min_avg_daily_units"]).all()
        assert sales["item_id"].str.startswith("FOODS").all()
        n_foods3 = (items["dept_id"] == "FOODS_3").sum()
        assert items["is_perishable"].sum() == min(r["n_perishable"], n_foods3)


# ---------------------------------------------------------------------------
# Rule R5: product choice uses only sales before test_start
# ---------------------------------------------------------------------------
def scale_sales(m5_dir, t_from, t_to, factor, items=None):
    """Multiply the fake sales on day indexes t_from..t_to (for `items`, or all) by `factor`."""
    path = Path(m5_dir) / "sales_train_evaluation.csv"
    wide = pd.read_csv(path)
    rows = wide["item_id"].isin(items) if items else slice(None)
    cols = [f"d_{t + 1}" for t in range(t_from, t_to + 1)]
    wide.loc[rows, cols] = wide.loc[rows, cols] * factor
    wide.to_csv(path, index=False)


def chosen(m5_dir, cfg):
    """The selection result to compare: the item table (ids, rank, stats, perishable flags)."""
    _, items = build_sales_table(cfg)
    return items.reset_index(drop=True)


def test_selection_ignores_test_and_holdout_sales(m5_dir):
    configs = [make_cfg(m5_dir)] + [make_cfg(m5_dir, "random", seed) for seed in [1, 2, 3, 4, 5]]
    before = [chosen(m5_dir, cfg) for cfg in configs]

    scale_sales(m5_dir, TEST_T, N_DAYS - 1, 1000)               # poison every day from test_start on
    after = [chosen(m5_dir, cfg) for cfg in configs]
    for b, a in zip(before, after):
        pd.testing.assert_frame_equal(b, a)                     # same items, rank, stats, flags, shelf lives

    # The poison is strong enough to matter: a whole-history ranking WOULD have changed.
    full = read_sales_wide(configs[0], read_calendar(m5_dir), include_holdout=True)
    foods = full[full["cat_id"] == "FOODS"]
    leaky = item_stats(foods, STORES, day_columns(foods))["item_id"].head(5).tolist()
    assert leaky != before[0]["item_id"].tolist()


def test_selection_does_react_to_pre_test_sales(m5_dir):
    # Sanity check of the test above: changing sales BEFORE test_start can change the choice.
    configs = [make_cfg(m5_dir)] + [make_cfg(m5_dir, "random", seed) for seed in [1, 2, 3, 4, 5]]
    before = [chosen(m5_dir, cfg) for cfg in configs]

    scale_sales(m5_dir, 7, TEST_T - 1, 1000, items=["FOODS_2_001", "FOODS_3_002"])
    after = [chosen(m5_dir, cfg) for cfg in configs]
    main_after = after[0]
    assert "FOODS_2_001" in set(main_after["item_id"])          # the almost-dead item is now a top seller
    assert set(main_after.loc[main_after["is_perishable"], "item_id"]) != \
        set(before[0].loc[before[0]["is_perishable"], "item_id"])
    assert any(not b["item_id"].equals(a["item_id"]) for b, a in zip(before[1:], after[1:]))


# ---------------------------------------------------------------------------
# Holdout seal
# ---------------------------------------------------------------------------
def test_holdout_sealed_by_default_and_opt_in_reads_it(m5_dir):
    cfg = make_cfg(m5_dir)
    sales, items = build_sales_table(cfg)
    assert sales["date"].max() < pd.Timestamp(WINDOWS["holdout_start"])

    full, full_items = build_sales_table(cfg, include_holdout=True)
    assert full["date"].max() == day(N_DAYS - 1)                # the last day in the sales file
    pd.testing.assert_frame_equal(items, full_items)             # same products, ranks, perishables
    assert set(full["item_id"]) == set(sales["item_id"])
    # Before the holdout the two tables agree row for row.
    pd.testing.assert_frame_equal(full[full["date"] < pd.Timestamp(WINDOWS["holdout_start"])]
                                  .reset_index(drop=True), sales)


def test_command_line_writes_separate_full_file(m5_dir):
    cfg = make_cfg(m5_dir)
    cfg_path = Path(m5_dir).parent / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = Path(cfg["data"]["processed_dir"])

    main(["--config", str(cfg_path)])
    assert (out / "sales.parquet").exists() and not (out / "sales_full.parquet").exists()
    sealed = pd.read_parquet(out / "sales.parquet")
    assert sealed["date"].max() < pd.Timestamp(WINDOWS["holdout_start"])

    main(["--config", str(cfg_path), "--include-holdout"])
    full = pd.read_parquet(out / "sales_full.parquet")
    assert full["date"].max() == day(N_DAYS - 1)
    assert pd.read_parquet(out / "sales.parquet").equals(sealed)   # the everyday file is untouched


# ---------------------------------------------------------------------------
# Window check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key, value, message", [
    ("val_start", iso(20), "overlap"),                  # validation starts on the last training day
    ("holdout_start", iso(33), "overlap"),              # holdout starts inside the test window
    ("test_start", iso(40), "after test_end"),          # out of order
    ("train_start", iso(7), "warmup_start must come before train_start"),
    ("holdout_end", "2016-06-01", "last M5 day"),       # past the end of M5
])
def test_bad_windows_raise(m5_dir, key, value, message):
    cfg = make_cfg(m5_dir)
    cfg["data"]["windows"][key] = value
    with pytest.raises(ValueError, match=message):
        check_windows(cfg)
    with pytest.raises(ValueError, match=message):
        build_sales_table(cfg)


@pytest.mark.parametrize("history_start, message", [
    (iso(8), "after warmup_start"),
    ("2010-01-01", "before the first M5 day"),
])
def test_bad_history_start_raises(m5_dir, history_start, message):
    cfg = make_cfg(m5_dir)
    cfg["data"]["history_start"] = history_start
    with pytest.raises(ValueError, match=message):
        check_windows(cfg)


def test_real_config_windows():
    cfg = load_config(ROOT / "config.yaml")
    check_windows(cfg)                                          # the agreed windows are valid
    assert selection_end(cfg) == pd.Timestamp("2015-10-31")    # day before test_start
    assert cfg["data"]["history_start"] == cfg["data"]["windows"]["warmup_start"]


# ---------------------------------------------------------------------------
# Integration test on the real M5 files (skipped if they are not downloaded)
# ---------------------------------------------------------------------------
REAL_FILES = ["calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"]


@pytest.mark.skipif(not all((ROOT / "data" / "m5" / f).exists() for f in REAL_FILES),
                    reason="real M5 files not in data/m5")
def test_real_main_set():
    cfg = load_config(ROOT / "config.yaml")
    cfg["data"]["product_set"] = "main"
    sales, items = build_sales_table(cfg)
    assert sales["store_id"].nunique() == 10
    assert len(items) == 120 and sales["item_id"].nunique() == 120
    assert items["is_perishable"].sum() == 20
    assert (items.loc[items["is_perishable"], "dept_id"] == "FOODS_3").all()
    assert sales["date"].min() >= pd.Timestamp(cfg["data"]["history_start"])
    assert sales["date"].max() == pd.Timestamp(cfg["data"]["windows"]["holdout_start"]) - pd.Timedelta(days=1)
    assert not sales.duplicated(["store_id", "item_id", "date"]).any()
    assert sales["price"].notna().all() and (sales["units"] >= 0).all()
