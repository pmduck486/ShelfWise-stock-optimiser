"""Tests for stubs.make_stubs (the placeholder forecasts and starting state).

All tests use a TINY fake sales table built in memory, so they run in about a second.
The `stubs:` and `network:` settings are read from the real config.yaml; only the time
windows are scaled down to fit the fake data.

Fake data: 2 stores (CA_1, TX_1), 3 items, 70 days starting 2015-01-01 (day index t = 0..69).
Test window t 50..55 (6 origins). One item launches late (no rows before t 20), so the
"not on sale yet" case is covered.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_prep.prepare import load_config  # noqa: E402
from stubs.make_stubs import (  # noqa: E402
    DC_STOCK_COLS, FORECAST_COLS, PIPELINE_COLS, STORE_STOCK_COLS,
    check_network, make_forecasts, start_state,
)

FIRST_DAY = "2015-01-01"
N_DAYS = 70
TEST_T, TEST_END_T = 50, 55
STORES = ["CA_1", "TX_1"]
ITEMS = ["FOODS_3_001", "FOODS_3_002", "FOODS_1_001"]
LATE_ITEM, LAUNCH_T = "FOODS_1_001", 20


def day(t):
    return pd.Timestamp(FIRST_DAY) + pd.Timedelta(days=t)


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config(ROOT / "config.yaml"))
    c["data"]["windows"]["test_start"] = day(TEST_T).strftime("%Y-%m-%d")
    c["data"]["windows"]["test_end"] = day(TEST_END_T).strftime("%Y-%m-%d")
    return c


@pytest.fixture
def sales():
    rng = np.random.default_rng(0)
    rows = []
    for s in STORES:
        for i in ITEMS:
            for t in range(N_DAYS):
                if i == LATE_ITEM and t < LAUNCH_T:
                    continue
                rows.append((day(t), s, i, int(rng.poisson(5 + 3 * (t % 7 == 0)))))
    return pd.DataFrame(rows, columns=["date", "store_id", "item_id", "units"])


@pytest.fixture
def origins():
    return pd.date_range(day(TEST_T), day(TEST_END_T), freq="D")


def test_forecast_contract_columns(sales, origins, cfg):
    fc = make_forecasts(sales, origins, cfg)
    assert list(fc.columns) == FORECAST_COLS
    assert pd.api.types.is_datetime64_any_dtype(fc["origin"])
    assert pd.api.types.is_datetime64_any_dtype(fc["date"])
    for q in ["q50", "q80", "q90", "q95"]:
        assert pd.api.types.is_float_dtype(fc[q])
    assert len(fc) == len(origins) * 7 * len(STORES) * len(ITEMS)


def test_each_origin_has_seven_dates(sales, origins, cfg):
    fc = make_forecasts(sales, origins, cfg)
    for o, grp in fc.groupby("origin"):
        expected = pd.date_range(o, periods=7, freq="D")
        assert list(pd.DatetimeIndex(sorted(grp["date"].unique()))) == list(expected)
        assert grp.groupby("date").size().eq(len(STORES) * len(ITEMS)).all()


def test_quantiles_ordered_nonnegative_no_nan(sales, origins, cfg):
    fc = make_forecasts(sales, origins, cfg)
    q = fc[["q50", "q80", "q90", "q95"]]
    assert not q.isna().any().any()
    assert (q >= 0).all().all()
    assert (fc["q50"] <= fc["q80"]).all()
    assert (fc["q80"] <= fc["q90"]).all()
    assert (fc["q90"] <= fc["q95"]).all()


def test_stub_rule_matches_hand_calculation(sales, origins, cfg):
    fc = make_forecasts(sales, origins, cfg)
    D, s, i = day(TEST_T + 3), "CA_1", "FOODS_3_001"
    u = sales.set_index(["store_id", "item_id", "date"])["units"]
    q50 = np.mean([u[(s, i, D - pd.Timedelta(days=L))] for L in (7, 14, 21, 28)])
    sd = np.std([u[(s, i, D - pd.Timedelta(days=k))] for k in range(7, 35)], ddof=1)
    row = fc[(fc.origin == day(TEST_T)) & (fc.date == D) & (fc.store_id == s) & (fc.item_id == i)].iloc[0]
    assert row["q50"] == pytest.approx(q50)
    assert row["q95"] == pytest.approx(max(q50 + cfg["stubs"]["z"]["q95"] * sd, 0))


@pytest.mark.parametrize("k", range(TEST_END_T - TEST_T + 1))
def test_future_poison(sales, origins, cfg, k):
    """Rule R2: sales on or after an origin must not change that origin's forecasts."""
    origin = origins[k]
    base = make_forecasts(sales, [origin], cfg)
    poisoned = sales.copy()
    poisoned.loc[poisoned["date"] >= origin, "units"] *= 1000
    after = make_forecasts(poisoned, [origin], cfg)
    pd.testing.assert_frame_equal(base, after)


def test_late_launch_item_has_no_nan(cfg):
    """An item with no rows at all before the target still gets 0, not NaN."""
    df = pd.DataFrame({"date": [day(60)], "store_id": ["CA_1"], "item_id": ["X"], "units": [9]})
    fc = make_forecasts(df, [day(10)], cfg)
    assert (fc[["q50", "q80", "q90", "q95"]] == 0).all().all()


def test_state_no_negatives_and_columns(sales, cfg):
    store_stock, dc_stock, pipeline = start_state(sales, cfg)
    assert list(store_stock.columns) == STORE_STOCK_COLS
    assert list(dc_stock.columns) == DC_STOCK_COLS
    assert list(pipeline.columns) == PIPELINE_COLS
    assert len(pipeline) == 0
    assert (store_stock["units"] >= 0).all() and (dc_stock["units"] >= 0).all()
    assert (store_stock["date"] == day(TEST_T)).all()
    assert len(store_stock) == len(STORES) * len(ITEMS)


def test_state_matches_rule(sales, cfg):
    """Store stock = round(3 x 28-day mean); DC stock = round(7 x sum of its home stores' means)."""
    store_stock, dc_stock, _ = start_state(sales, cfg)
    win = sales[(sales["date"] >= day(TEST_T - 28)) & (sales["date"] < day(TEST_T))]
    means = win.groupby(["store_id", "item_id"])["units"].mean()
    ss = cfg["stubs"]["start_state"]
    got = store_stock.set_index(["store_id", "item_id"])["units"]
    assert (got == np.round(ss["store_stock_days"] * means).astype(int)).all()
    home = cfg["network"]["home_dc"]
    for i in ITEMS:
        west = means[("CA_1", i)]
        row = dc_stock[(dc_stock.dc_id == home["CA_1"]) & (dc_stock.item_id == i)].iloc[0]
        assert row["units"] == round(ss["dc_stock_days"] * west)


def test_state_ignores_future_sales(sales, cfg):
    before = start_state(sales, cfg)
    poisoned = sales.copy()
    poisoned.loc[poisoned["date"] >= day(TEST_T), "units"] *= 1000
    after = start_state(poisoned, cfg)
    for a, b in zip(before[:2], after[:2]):
        pd.testing.assert_frame_equal(a, b)


def test_every_config_store_has_a_dc(cfg):
    stores = cfg["data"]["stores"]
    check_network(cfg, stores)                       # does not raise
    assert set(cfg["network"]["home_dc"].values()) <= set(cfg["network"]["dcs"])


def test_unknown_store_raises(cfg):
    with pytest.raises(ValueError, match="home_dc"):
        check_network(cfg, ["CA_1", "ZZ_9"])
