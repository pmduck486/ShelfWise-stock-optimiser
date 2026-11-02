"""Fairness tests for the ML forecaster (plan/forecast-training-plan.md, section 4).

These guard the rules that keep the forecaster honest: no row on or after a model's
cutoff is used to fit it (R1), the holdout stays sealed (R1), the validation folds
are in order and end before the test window (R6), and every store-data input is at
least forecast.min_lag_days older than the day it forecasts (R2). Calendar inputs may be
same-day (R3).

R2 changed on 2026-11-02 from 7 to 2 days (store-only model, 2-day forecast). The builder's
arithmetic tests below keep their worked numbers at the old 7-day settings (fixture `cfg`);
the leak tests at the end check the live config (fixture `live_cfg`, 2 days).

Some tests read the real files (sales.parquet, calendar.csv) and are skipped if missing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast.data import (  # noqa: E402
    check_cutoff, load_calendar, load_config, load_sales, val_folds, windows,
)
from forecast.features import (  # noqa: E402
    CALENDAR_FEATURES, ID_FEATURES, build_features, categorical_mask,
    empty_shelf_filter, feature_names, feature_sources, fit_encoders, training_rows, zero_run_stats,
)

CONFIG = ROOT / "config.yaml"
SALES_FILE = ROOT / "data" / "processed" / "sales.parquet"
needs_sales = pytest.mark.skipif(not SALES_FILE.exists(), reason="sales.parquet not built")


def seven_day(c):
    """The config with the pre-2026-11-02 rule (7-day gap, 7-day horizon)."""
    c["forecast"].update(min_lag_days=7, horizon_days=7, lags=[7, 14, 21, 28])
    return c


@pytest.fixture
def cfg():
    return seven_day(load_config(CONFIG))


@pytest.fixture
def live_cfg():
    return load_config(CONFIG)


def tiny_sales(last_day):
    """Three days of sales for one store-item, the last one on `last_day`."""
    days = pd.date_range(end=last_day, periods=3, freq="D")
    return pd.DataFrame({"date": days, "store_id": "CA_1", "item_id": "FOODS_3_001", "units": [1, 2, 3]})


# ---------------------------------------------------------------------------
# Cutoff guard (R1)
# ---------------------------------------------------------------------------
def test_cutoff_guard_raises():
    sales = tiny_sales("2015-07-01")
    with pytest.raises(ValueError, match="on or after cutoff"):
        check_cutoff(sales, pd.Timestamp("2015-07-01"))


def test_cutoff_guard_allows_day_before():
    check_cutoff(tiny_sales("2015-06-30"), pd.Timestamp("2015-07-01"))


# ---------------------------------------------------------------------------
# Holdout seal (R1)
# ---------------------------------------------------------------------------
@needs_sales
def test_dev_loader_never_returns_holdout_rows(cfg):
    sales = load_sales(cfg)
    assert sales["date"].max() < windows(cfg)["holdout_start"]


def test_loader_refuses_holdout_rows(cfg, tmp_path):
    """A sales file that reaches into the holdout is refused, not trimmed quietly."""
    sales = tiny_sales(cfg["data"]["windows"]["holdout_start"])
    sales.to_parquet(tmp_path / "sales.parquet")
    cfg["data"]["processed_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match="holdout"):
        load_sales(cfg)


# ---------------------------------------------------------------------------
# Fold check (R1, R6)
# ---------------------------------------------------------------------------
def test_val_folds_in_order(cfg):
    folds = val_folds(cfg)
    w = windows(cfg)
    assert [f["name"] for f in folds] == ["fold1_festive", "fold2_main"]
    for f in folds:
        assert f["train_end"] + pd.Timedelta(days=1) == f["val_start"]
        assert f["val_start"] <= f["val_end"] < w["test_start"]
        assert f["train_end"] >= w["train_start"]
    assert folds[0]["val_end"] < folds[1]["val_start"]
    # fold 2 is the main validation window of data.windows
    assert folds[1]["train_end"] == w["train_end"]
    assert folds[1]["val_start"] == w["val_start"]
    assert folds[1]["val_end"] == w["val_end"]


def test_val_folds_reject_overlap(cfg):
    cfg["forecast"]["val_folds"][0]["val_end"] = "2015-07-15"     # runs into fold 2
    with pytest.raises(ValueError, match="fold"):
        val_folds(cfg)


def test_val_folds_reject_fold_reaching_test(cfg):
    cfg["forecast"]["val_folds"][1]["val_end"] = cfg["data"]["windows"]["test_start"]
    with pytest.raises(ValueError, match="test_start"):
        val_folds(cfg)


# ---------------------------------------------------------------------------
# Calendar (R3): published in advance, read from calendar.csv, never from sales
# ---------------------------------------------------------------------------
@needs_sales
def test_calendar_matches_sales(cfg):
    """For every day in sales.parquet, calendar.csv gives the same wday, month, event and SNAP."""
    sales = load_sales(cfg)
    cal = load_calendar(cfg)
    m = sales.merge(cal, on="date", how="left", suffixes=("", "_cal"), validate="many_to_one")
    assert m["wday_cal"].notna().all()
    assert (m["wday"] == m["wday_cal"]).all()
    assert (m["month"] == m["month_cal"]).all()
    assert (m["event_name"] == m["event_name_cal"]).all()
    assert (m["is_event"] == m["is_event_cal"]).all()
    state = m["store_id"].str[:2]
    snap_cal = pd.Series(False, index=m.index)
    for st in ("CA", "TX", "WI"):
        snap_cal[state == st] = m.loc[state == st, f"snap_{st}"]
    assert (m["snap"] == snap_cal).all()


def test_calendar_covers_forecast_days(cfg):
    """The last test origin (2015-12-26) forecasts up to 2016-01-01, after sales end."""
    cal = load_calendar(cfg)
    w = windows(cfg)
    last_target = w["test_end"] + pd.Timedelta(days=cfg["forecast"]["horizon_days"] - 1)
    days = pd.date_range(w["warmup_start"], last_target, freq="D")
    assert set(days) <= set(cal["date"])
    assert cal["date"].is_unique


def test_calendar_columns_and_types(cfg):
    cal = load_calendar(cfg)
    assert list(cal.columns) == ["date", "wday", "month", "event_name", "is_event",
                                 "snap_CA", "snap_TX", "snap_WI"]
    assert str(cal["date"].dtype).startswith("datetime64")
    for c in ("is_event", "snap_CA", "snap_TX", "snap_WI"):
        assert cal[c].dtype == bool
    assert (cal["event_name"] != "").eq(cal["is_event"]).all()


# ---------------------------------------------------------------------------
# Feature builder (Task 3): store data >= 7 days old (R2), calendar same-day (R3),
# statistics from training rows only (R4)
# ---------------------------------------------------------------------------
KEY = ["date", "store_id", "item_id"]
SALES_COLS = ["date", "store_id", "item_id", "dept_id", "units", "price", "is_perishable",
              "event_name"]
DAY = pd.Timedelta(days=1)


def synth_sales(cfg, start, days, stores=("CA_1", "TX_1"), n_items=4, seed=0):
    """Made-up daily sales for a few store-items, with the real calendar's event names."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    frames = []
    for st in stores:
        for i in range(n_items):
            frames.append(pd.DataFrame({
                "date": dates, "store_id": st, "item_id": f"FOODS_3_{i:03d}", "dept_id": "FOODS_3",
                "units": rng.poisson(5 + 3 * i, days).astype("int32"),
                "price": np.round(2.0 + 0.5 * i + rng.choice([0.0, 0.0, 0.0, -0.3], days), 2),
                "is_perishable": i == 0,
            }))
    sales = pd.concat(frames, ignore_index=True)
    cal = load_calendar(cfg)[["date", "event_name"]]
    return sales.merge(cal, on="date", how="left")[SALES_COLS]


def one(day, store="CA_1", item="FOODS_3_001"):
    return pd.DataFrame({"date": [pd.Timestamp(day)], "store_id": [store], "item_id": [item]})


def with_later(day, **kw):
    """The target `day` plus the same store-item 7 days later, so day - 6 lies inside the day
    grid; tests compare row 0 only."""
    return pd.concat([one(day, **kw), one(pd.Timestamp(day) + 7 * DAY, **kw)], ignore_index=True)


def row0(frame):
    return frame.iloc[[0]]


def all_targets(sales, start, end):
    """Every store-item in `sales` on every day start..end."""
    pairs = sales[["store_id", "item_id"]].drop_duplicates()
    days = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    return pairs.merge(days, how="cross")[KEY].sort_values(KEY).reset_index(drop=True)


def bump(sales, day, col, by, store=None, item=None, other_items=False):
    """A copy of `sales` with `col` changed by `by` on `day` (for one store-item, or for every
    store-item except it when other_items=True)."""
    out = sales.copy()
    on_day = out["date"] == pd.Timestamp(day)
    this = (out["store_id"] == store) & (out["item_id"] == item)
    m = on_day & (~this if other_items else this)
    out.loc[m, col] = out.loc[m, col] + by
    return out


_REAL = {}


def real_two_stores(cfg):
    """The real sales of stores CA_1 and TX_1 (240 store-items), read once."""
    if "sales" not in _REAL:
        s = load_sales(cfg)
        _REAL["sales"] = s[s["store_id"].isin(["CA_1", "TX_1"])].reset_index(drop=True)
    return _REAL["sales"]


@needs_sales
@pytest.mark.parametrize("origin", ["2014-11-01", "2015-07-01", "2015-11-26"])
def test_future_poison_features(cfg, origin):
    """Units x1000 and price NaN on every row >= origin: inputs for origin..origin+6 don't move.
    Two weeks are built together, so the poisoned days sit inside the day grid; only the
    first week is compared."""
    sales = real_two_stores(cfg)
    origin = pd.Timestamp(origin)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2014-11-01"))
    targets = all_targets(sales, origin, origin + 13 * DAY)
    week1 = (targets["date"] <= origin + 6 * DAY).to_numpy()
    poisoned = sales.copy()
    late = poisoned["date"] >= origin
    poisoned.loc[late, "units"] = poisoned.loc[late, "units"] * 1000
    poisoned.loc[late, "price"] = np.nan
    a = build_features(sales, targets, cfg, enc)[week1]
    b = build_features(poisoned, targets, cfg, enc)[week1]
    assert a["lag_7"].notna().mean() > 0.9
    pd.testing.assert_frame_equal(a, b)


def test_feature_age_registry(cfg):
    names = feature_names(cfg)
    assert len(names) == len(set(names))
    store_cols = [n for n in names if n not in CALENDAR_FEATURES and n not in ID_FEATURES]
    assert store_cols
    src = feature_sources(cfg)
    assert all(c in src for c in store_cols), set(store_cols) - set(src)
    assert all(age >= cfg["forecast"]["min_lag_days"] for age in src.values())
    assert len(categorical_mask(cfg)) == len(names)


def test_feature_age_by_perturbation(cfg):
    """Units on target - 6 change nothing; units on target - 7 change lag_7. (The target is not
    in the week after 25 Dec: a lag on a closed day stays NaN by design.)"""
    sales = synth_sales(cfg, "2015-02-01", 120)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-05-01"))
    t = pd.Timestamp("2015-05-20")
    target = with_later(t)
    base = row0(build_features(sales, target, cfg, enc))
    six = bump(sales, t - 6 * DAY, "units", 50, "CA_1", "FOODS_3_001")
    pd.testing.assert_frame_equal(base, row0(build_features(six, target, cfg, enc)))
    seven = bump(sales, t - 7 * DAY, "units", 50, "CA_1", "FOODS_3_001")
    changed = row0(build_features(seven, target, cfg, enc))
    assert changed["lag_7"].iloc[0] == base["lag_7"].iloc[0] + 50


def test_encoders_ignore_post_train_rows(cfg):
    """An item first sold after the cutoff gets no code; prices after it don't move the
    normal price (R4)."""
    sales = synth_sales(cfg, "2015-02-01", 120)
    cutoff = pd.Timestamp("2015-05-01")
    new = sales[(sales["item_id"] == "FOODS_3_000") & (sales["date"] >= cutoff)].assign(
        item_id="FOODS_3_999")
    both = pd.concat([sales, new], ignore_index=True)
    enc = fit_encoders(both, cutoff=cutoff)
    f = build_features(both, pd.concat([one("2015-05-20", item="FOODS_3_999"),
                                        one("2015-05-20", item="FOODS_3_000")]), cfg, enc)
    assert np.isnan(f["item_id"].iloc[0]) and not np.isnan(f["item_id"].iloc[1])
    repriced = both.copy()
    repriced.loc[repriced["date"] >= cutoff, "price"] *= 3
    enc2 = fit_encoders(repriced, cutoff=cutoff)
    pd.testing.assert_series_equal(enc["normal_price"], enc2["normal_price"])


def test_long_zero_rule_uses_train_only(cfg):
    sales = synth_sales(cfg, "2014-02-01", 400)
    m = ((sales["store_id"] == "CA_1") & (sales["item_id"] == "FOODS_3_002")
         & sales["date"].between("2014-03-01", "2014-06-08"))           # 100 zero days
    sales.loc[m, "units"] = 0
    cutoff = pd.Timestamp("2014-09-01")
    a = zero_run_stats(sales, cfg, cutoff=cutoff)
    altered = sales.copy()
    altered.loc[altered["date"] >= cutoff, "units"] = 0                 # a ~185-day run after cutoff
    b = zero_run_stats(altered, cfg, cutoff=cutoff)
    pd.testing.assert_frame_equal(a["runs"], b["runs"])
    assert a["runs"][["store_id", "item_id"]].values.tolist() == [["CA_1", "FOODS_3_002"]]


def test_sale_features_age(cfg):
    sales = synth_sales(cfg, "2015-02-01", 120)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-05-01"))
    for name in ("price_rel_item", "momentum", "store_rank_28"):
        assert feature_sources(cfg)[name] >= 7
    t = pd.Timestamp("2015-05-20")
    target = with_later(t)
    cols = ["price_rel_item", "momentum", "store_rank_28"]
    base = row0(build_features(sales, target, cfg, enc))[cols]
    for col, by in (("units", 40), ("price", -1.0)):
        moved = bump(sales, t - 6 * DAY, col, by, "CA_1", "FOODS_3_001")
        pd.testing.assert_frame_equal(base, row0(build_features(moved, target, cfg, enc))[cols])


def test_sale_features_no_inf(cfg):
    """A 28-day mean of 0 gives momentum 1; a zero or missing price gives a finite
    price_rel_item."""
    sales = synth_sales(cfg, "2015-02-01", 120)
    t = pd.Timestamp("2015-05-20")
    item = lambda i: (sales["store_id"] == "CA_1") & (sales["item_id"] == f"FOODS_3_{i:03d}")
    sales.loc[item(1) & (sales["date"] >= t - 40 * DAY), "units"] = 0
    sales.loc[item(2) & (sales["date"] == t - 7 * DAY), "price"] = 0.0
    sales.loc[item(3) & (sales["date"] == t - 7 * DAY), "price"] = np.nan
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-05-01"))
    targets = pd.concat([one(t, item=f"FOODS_3_{i:03d}") for i in (1, 2, 3)], ignore_index=True)
    f = build_features(sales, targets, cfg, enc)
    assert f["momentum"].iloc[0] == 1.0
    assert np.isfinite(f["price_rel_item"]).all()
    assert np.isfinite(f["price_chg_4w"]).all()


def test_p7_hook_off_is_identity(cfg):
    assert cfg["forecast"]["layers"]["p7_empty_shelf"] is False
    sales = synth_sales(cfg, "2015-02-01", 60)
    rows = sales[KEY + ["units"]]
    pd.testing.assert_frame_equal(empty_shelf_filter(rows, sales, cfg), rows)


def test_calendar_features_ignore_sales(cfg):
    """Calendar inputs read calendar.csv only: poisoning every sale and price changes none of
    them. The day before Thanksgiving (2015-11-25) is 1 day from it, and its next event is
    Thanksgiving."""
    sales = synth_sales(cfg, "2014-11-01", 395, stores=("CA_1",), n_items=2)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-11-20"))
    targets = pd.concat([one(d) for d in pd.date_range("2015-11-20", "2015-11-28")],
                        ignore_index=True)
    a = build_features(sales, targets, cfg, enc)[CALENDAR_FEATURES]
    poisoned = sales.assign(units=sales["units"] * 1000, price=np.nan)
    b = build_features(poisoned, targets, cfg, enc)[CALENDAR_FEATURES]
    pd.testing.assert_frame_equal(a, b)
    eve = a[targets["date"] == pd.Timestamp("2015-11-25")].iloc[0]
    assert eve["days_to_event"] == 1
    assert eve["next_event_name"] == enc["event_name"]["Thanksgiving"]


def test_cross_item_perturbation(cfg):
    """Changing every OTHER store-item's units and price on target - 6 leaves every input of
    the target row unchanged, store_rank_28 included. On target - 7 the rank does move."""
    sales = synth_sales(cfg, "2015-02-01", 120)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-05-01"))
    t = pd.Timestamp("2015-05-20")
    target = with_later(t)
    base = row0(build_features(sales, target, cfg, enc))
    for col, by in (("units", 500), ("price", 1.0)):
        moved = bump(sales, t - 6 * DAY, col, by, "CA_1", "FOODS_3_001", other_items=True)
        pd.testing.assert_frame_equal(base, row0(build_features(moved, target, cfg, enc)))
    # the test can see a change: other items selling far more 7 days back lowers the rank
    moved = bump(sales, t - 7 * DAY, "units", 5000, "CA_1", "FOODS_3_001", other_items=True)
    assert build_features(moved, target, cfg, enc)["store_rank_28"].iloc[0] < \
        base["store_rank_28"].iloc[0]


@needs_sales
def test_targets_after_sales_end(cfg):
    """2015-12-27 -> 2016-01-01 (after the last sales day): inputs build, the calendar comes
    from calendar.csv, and store data is read >= 7 days back."""
    sales = real_two_stores(cfg)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-11-01"))
    targets = all_targets(sales, "2015-12-27", "2016-01-01")
    f = build_features(sales, targets, cfg, enc)
    assert len(f) == len(targets) == 6 * 240
    new_year = (targets["date"] == pd.Timestamp("2016-01-01")).to_numpy()
    assert (f.loc[new_year, "event_name"] == enc["event_name"]["NewYear"]).all()
    assert (f.loc[new_year, "wday"] == 7).all()
    assert f.loc[new_year, "lag_7"].isna().all()                        # 25 Dec is closed
    first = (targets["date"] == pd.Timestamp("2015-12-27")).to_numpy()
    lag = targets[first].merge(
        sales.assign(date=sales["date"] + 7 * DAY)[KEY + ["units"]], on=KEY, how="left")
    np.testing.assert_array_equal(f.loc[first, "lag_7"].to_numpy(), lag["units"].to_numpy())


@needs_sales
def test_categories_fit_model(cfg):
    """Every category column has at most 255 codes (the model's bin limit)."""
    sales = real_two_stores(cfg)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-07-01"))
    f = build_features(sales, all_targets(sales, "2015-06-01", "2015-06-07"), cfg, enc)
    cats = [n for n, c in zip(feature_names(cfg), categorical_mask(cfg)) if c]
    assert set(cats) == {"store_id", "item_id", "dept_id", "event_name", "next_event_name"}
    for c in cats:
        assert f[c].nunique() <= 255 and (f[c].dropna() < 255).all(), c
    for name in ("store_id", "item_id", "dept_id", "event_name"):
        assert len(enc[name]) <= 255, name


def test_closed_day_is_missing_in_features(cfg):
    """25 Dec's near-zero sales never feed a later day's inputs: 1 Jan's lag_7 is NaN, and its
    7-day average is the mean of 19-24 Dec (window rule: the days present)."""
    sales = synth_sales(cfg, "2014-10-01", 120)
    key = (sales["store_id"] == "CA_1") & (sales["item_id"] == "FOODS_3_001")
    sales.loc[sales["date"] == pd.Timestamp("2014-12-25"), "units"] = 0
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2014-12-31"))
    f = build_features(sales, one("2015-01-01"), cfg, enc).iloc[0]
    assert np.isnan(f["lag_7"])
    u = sales[key].set_index("date")["units"].astype(float)
    open_7 = u["2014-12-19":"2014-12-24"].mean()
    open_28 = u["2014-11-28":"2014-12-24"].mean()
    assert f["roll_mean_7"] == pytest.approx(open_7)
    assert f["roll_mean_28"] == pytest.approx(open_28)
    assert f["momentum"] == pytest.approx(open_7 / open_28)


def test_zero_run_skips_closed_day(cfg):
    """A closed day neither breaks a zero run nor adds to it: 47 + 25 Dec + 47 zero days is one
    94-day run (> 90); 45 + 25 Dec + 45 is 90 days (not > 90)."""
    def stats(before, after):
        sales = synth_sales(cfg, "2014-08-01", 240, stores=("CA_1",), n_items=1)
        xmas = pd.Timestamp("2014-12-25")
        zero = sales["date"].between(xmas - before * DAY, xmas + after * DAY)
        sales.loc[zero, "units"] = 0
        sales.loc[sales["date"] == xmas, "units"] = 2                   # a few stray sales
        return zero_run_stats(sales, cfg, cutoff=pd.Timestamp("2015-03-01"))["runs"]
    long_run = stats(47, 47)
    assert len(long_run) == 1
    assert long_run["open_days"].iloc[0] == 94
    assert len(stats(45, 45)) == 0


@needs_sales
def test_same_target_alone_or_in_batch(cfg):
    """A day's inputs don't depend on which other days are built with it: the same target
    alone and inside a two-year training batch give exactly the same row (review finding:
    float rounding in the price window made price_chg_4w depend on the batch)."""
    sales = real_two_stores(cfg)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-07-01"))
    day = pd.Timestamp("2015-06-20")
    alone = build_features(sales, all_targets(sales, day, day), cfg, enc)
    batch_t = all_targets(sales, "2013-07-01", "2015-06-30")
    batch = build_features(sales, batch_t, cfg, enc)
    inside = batch[(batch_t["date"] == day).to_numpy()].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone, inside, check_exact=True)


def test_statistics_need_a_cutoff(cfg):
    """Codes, normal prices and zero runs must say which rows they came from (R4): no
    cutoff is refused, and training rows must end before the zero-run cutoff."""
    sales = synth_sales(cfg, "2015-02-01", 120)
    with pytest.raises(TypeError):
        fit_encoders(sales)
    with pytest.raises(TypeError):
        zero_run_stats(sales, cfg)
    no_cutoff = {"cutoff": None, "min_days": 90, "runs": zero_run_stats(
        sales, cfg, cutoff=pd.Timestamp("2015-05-01"))["runs"]}
    with pytest.raises(ValueError, match="cutoff"):
        training_rows(sales, "2015-02-01", "2015-04-30", cfg, no_cutoff)
    stats = zero_run_stats(sales, cfg, cutoff=pd.Timestamp("2015-05-01"))
    with pytest.raises(ValueError, match="cutoff|zero-run"):
        training_rows(sales, "2015-02-01", "2015-05-01", cfg, stats)    # end not before cutoff
    training_rows(sales, "2015-02-01", "2015-04-30", cfg, stats)


def test_training_rows_drop_long_zero_runs(cfg):
    """With drop_long_zero_runs on, exactly the rows inside a long zero run are left out."""
    cfg["forecast"]["drop_long_zero_runs"] = True        # config.yaml has it off (decision 26)
    sales = synth_sales(cfg, "2014-02-01", 300)
    run = ((sales["store_id"] == "CA_1") & (sales["item_id"] == "FOODS_3_002")
           & sales["date"].between("2014-03-01", "2014-06-08"))            # 100 zero days
    sales.loc[run, "units"] = 0
    cutoff = pd.Timestamp("2014-09-01")
    stats = zero_run_stats(sales, cfg, cutoff=cutoff)
    rows = training_rows(sales, "2014-02-01", cutoff - DAY, cfg, stats)
    kept = sales[sales["date"] < cutoff]
    assert len(rows) == len(kept) - int(run.sum())
    dropped = kept[run[kept.index]][KEY].merge(rows, on=KEY)
    assert dropped.empty
    cfg["forecast"]["drop_long_zero_runs"] = False
    assert len(training_rows(sales, "2014-02-01", cutoff - DAY, cfg, stats)) == len(kept)


# ---------------------------------------------------------------------------
# Exact values (review findings: logic the leak tests above don't pin down)
# ---------------------------------------------------------------------------
def series(sales, store="CA_1", item="FOODS_3_001"):
    return (sales["store_id"] == store) & (sales["item_id"] == item)


def test_price_inputs_exact(cfg):
    """price_7, price_chg_4w (vs the mean of t-35..t-8) and price_rel_item (vs the mean price
    before the cutoff) on a price step from 2.00 to 1.50."""
    sales = synth_sales(cfg, "2015-01-01", 150)
    me = series(sales)
    step = pd.Timestamp("2015-04-01")
    sales.loc[me, "price"] = np.where(sales.loc[me, "date"] < step, 2.00, 1.50)
    cutoff = pd.Timestamp("2015-04-10")
    enc = fit_encoders(sales, cutoff=cutoff)
    t = step + 10 * DAY                                     # t - 7 = 4 Apr, 3 days into the step
    f = build_features(sales, one(t), cfg, enc).iloc[0]
    p = sales[me].set_index("date")["price"]
    prior = p[t - 35 * DAY:t - 8 * DAY]
    assert len(prior) == 28
    assert f["price_7"] == pytest.approx(1.50)
    assert f["price_chg_4w"] == pytest.approx(1.50 / prior.mean() - 1, rel=1e-6)
    assert f["price_rel_item"] == pytest.approx(1.50 / p[p.index < cutoff].mean(), rel=1e-6)


def test_window_half_rule_std_and_momentum(cfg):
    """Averages need half the window present (4 of 7, 14 of 28); the spread is the sample std
    (ddof=1); momentum is NaN when an average is missing."""
    sales = synth_sales(cfg, "2015-01-01", 150)
    t = pd.Timestamp("2015-04-20")
    end = t - 7 * DAY
    me = series(sales)
    u = sales[me].set_index("date")["units"].astype(float)
    win28 = u[end - 27 * DAY:end]
    enc = fit_encoders(sales, cutoff=t)
    f = build_features(sales, one(t), cfg, enc).iloc[0]
    assert f["roll_mean_28"] == pytest.approx(win28.mean())
    assert f["roll_std_28"] == pytest.approx(np.std(win28.to_numpy(), ddof=1))
    assert f["momentum"] == pytest.approx(u[end - 6 * DAY:end].mean() / win28.mean())

    def gappy(keep_in_week):                                # rows missing in the 7-day window
        week = pd.date_range(end - 6 * DAY, end)
        gone = me & sales["date"].isin(week[keep_in_week:])
        return sales[~gone]
    three = build_features(gappy(3), one(t), cfg, enc).iloc[0]
    four = build_features(gappy(4), one(t), cfg, enc).iloc[0]
    assert np.isnan(three["roll_mean_7"]) and np.isnan(three["momentum"])
    assert four["roll_mean_7"] == pytest.approx(u[end - 6 * DAY:end - 3 * DAY].mean())


def test_snap_by_state(cfg):
    """Each store gets its own state's SNAP day from calendar.csv."""
    cal = load_calendar(cfg)
    day = cal.loc[(cal["snap_CA"] != cal["snap_TX"]) & (cal["date"] > "2015-04-01"), "date"].iloc[0]
    sales = synth_sales(cfg, "2015-01-01", 150)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-04-01"))
    f = build_features(sales, pd.concat([one(day), one(day, store="TX_1")], ignore_index=True),
                       cfg, enc)
    row = cal[cal["date"] == day].iloc[0]
    assert f["snap"].tolist() == [float(row["snap_CA"]), float(row["snap_TX"])]


def test_event_distances_match_calendar(cfg):
    """days_to_event / days_since_event (0 on the event, capped at 30) and next_event_name (NaN
    when the next event is over 30 days away), against a plain loop over calendar.csv."""
    cap = cfg["forecast"]["event_distance_cap_days"]
    cal = load_calendar(cfg)
    events = cal[cal["is_event"]]
    sales = synth_sales(cfg, "2014-01-01", 760, stores=("CA_1",), n_items=1)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2016-01-01"))
    days = pd.date_range("2014-03-01", "2015-12-31")
    targets = pd.concat([one(d, item="FOODS_3_000") for d in days], ignore_index=True)
    f = build_features(sales, targets, cfg, enc)
    want_to, want_since, want_next = [], [], []
    for d in days:
        nxt = events[events["date"] >= d].iloc[0]
        prv = events[events["date"] <= d].iloc[-1]
        to_next = (nxt["date"] - d).days
        want_to.append(min(to_next, cap))
        want_since.append(min((d - prv["date"]).days, cap))
        want_next.append(enc["event_name"][nxt["event_name"]] if to_next <= cap else np.nan)
    np.testing.assert_array_equal(f["days_to_event"], want_to)
    np.testing.assert_array_equal(f["days_since_event"], want_since)
    np.testing.assert_array_equal(f["next_event_name"], np.array(want_next, dtype=np.float32))
    assert (f["days_to_event"] == cap).any() and f["next_event_name"].isna().any()
    assert (f["days_since_event"] == 0).any() and (f["days_since_event"] == 1).any()


def test_store_rank_is_per_store(cfg):
    """An item's rank compares it with its own store's items only."""
    sales = synth_sales(cfg, "2015-02-01", 120)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-05-01"))
    t = pd.Timestamp("2015-05-20")
    base = build_features(sales, one(t), cfg, enc)["store_rank_28"].iloc[0]
    tx = sales.copy()
    window = tx["date"].between(t - 34 * DAY, t - 7 * DAY) & (tx["store_id"] == "TX_1")
    tx.loc[window, "units"] = tx.loc[window, "units"] * 100        # TX_1 sells far more
    assert build_features(tx, one(t), cfg, enc)["store_rank_28"].iloc[0] == base


# ---------------------------------------------------------------------------
# The live rule (R2 = 2 days since 2026-11-02): every input known on the origin's morning
# ---------------------------------------------------------------------------
def test_live_rule_two_days(live_cfg):
    f = live_cfg["forecast"]
    assert f["min_lag_days"] == 2 and f["horizon_days"] <= f["min_lag_days"]
    src = feature_sources(live_cfg)
    names = feature_names(live_cfg)
    store_cols = [n for n in names if n not in CALENDAR_FEATURES and n not in ID_FEATURES]
    assert set(store_cols) == set(src)
    assert min(src.values()) == 2


def test_live_rule_refuses_a_lag_newer_than_the_gap(live_cfg):
    live_cfg["forecast"]["lags"] = [1, 7]
    with pytest.raises(ValueError, match="newer than forecast.min_lag_days"):
        feature_names(live_cfg)


def test_live_rule_by_perturbation(live_cfg):
    """Units on target - 1 change nothing; units on target - 2 change lag_2."""
    cfg = live_cfg
    sales = synth_sales(cfg, "2015-02-01", 120)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-05-01"))
    t = pd.Timestamp("2015-05-20")
    target = with_later(t)
    base = row0(build_features(sales, target, cfg, enc))
    for col, by in (("units", 50), ("price", -1.0)):
        moved = sales.copy()
        moved.loc[moved["date"] == t - 1 * DAY, col] += by          # every store-item, the day before
        assert not moved.equals(sales)
        pd.testing.assert_frame_equal(base, row0(build_features(moved, target, cfg, enc)))
    two = bump(sales, t - 2 * DAY, "units", 50, "CA_1", "FOODS_3_001")
    assert row0(build_features(two, target, cfg, enc))["lag_2"].iloc[0] == base["lag_2"].iloc[0] + 50


@needs_sales
@pytest.mark.parametrize("origin", ["2014-11-01", "2015-07-01", "2015-11-26"])
def test_live_future_poison(live_cfg, origin):
    """Every row >= origin poisoned: the inputs for the forecast days (origin, origin + 1)
    don't move."""
    cfg = live_cfg
    sales = real_two_stores(cfg)
    origin = pd.Timestamp(origin)
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2014-11-01"))
    targets = all_targets(sales, origin, origin + 13 * DAY)
    days = (targets["date"] <= origin + (cfg["forecast"]["horizon_days"] - 1) * DAY).to_numpy()
    poisoned = sales.copy()
    late = poisoned["date"] >= origin
    poisoned.loc[late, "units"] = poisoned.loc[late, "units"] * 1000
    poisoned.loc[late, "price"] = np.nan
    a = build_features(sales, targets, cfg, enc)[days]
    b = build_features(poisoned, targets, cfg, enc)[days]
    assert a["lag_2"].notna().mean() > 0.9
    pd.testing.assert_frame_equal(a, b)
