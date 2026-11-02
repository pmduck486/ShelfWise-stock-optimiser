"""Tests for the ML forecaster package (forecast/): config, baselines, scorer.

Fairness tests (cutoff guard, holdout seal, fold check, calendar) live in
test_forecast_fairness.py. These use tiny in-memory frames so they run in about a second.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast.baseline import baseline_lines, targets_for  # noqa: E402
from forecast.data import load_config  # noqa: E402
from forecast.metrics import (  # noqa: E402
    coverage, mae, pinball, score, segments, service_line,
)

CONFIG = ROOT / "config.yaml"

FORECAST_KEYS = [
    "quantiles", "horizon_days", "min_lag_days", "lags", "baseline_lag_days", "rolling_windows",
    "price_change_weeks",
    "baseline_z", "drop_dates", "closed_dates", "event_distance_cap_days", "long_zero_run_days",
    "drop_long_zero_runs", "dead_item_q50", "slow_seller_max_avg_units", "val_folds", "calibration_tolerance_pts",
    "seed", "keep_seeds", "params", "line_params", "tuning_grid", "review", "freeze_platform", "model_dir", "scores_dir", "calendar_file", "service_lines",
    "layers",
]


@pytest.fixture
def cfg():
    return load_config(CONFIG)


def test_forecast_config_has_all_keys(cfg):
    f = cfg["forecast"]
    missing = [k for k in FORECAST_KEYS if k not in f]
    assert not missing, f"forecast: section is missing {missing}"
    assert f["quantiles"] == [0.5, 0.8, 0.9, 0.95]
    assert f["min_lag_days"] == 2                          # rule R2 (2 days since 2026-11-02, was 7)
    assert f["horizon_days"] <= f["min_lag_days"]           # every input known on the origin's morning
    assert min(f["lags"]) >= f["min_lag_days"]
    assert f["baseline_lag_days"] == 7                     # the baselines did not change
    assert set(f["baseline_z"]) == {"q80", "q90", "q95"}


def test_layers_default_off(cfg):
    """Every forecaster layer is a switch, and the core runs with all of them off."""
    layers = cfg["forecast"]["layers"]
    assert layers, "no layer switches"
    assert all(v is False for v in layers.values()), layers


def test_tuning_grid_size_limit(cfg):
    grid = cfg["forecast"]["tuning_grid"]
    n = 1
    for values in grid.values():
        n *= len(values)
    assert n <= 6


def test_forecast_paths_resolved(cfg):
    """Paths in the forecast: section are relative to config.yaml, like the data: ones."""
    f = cfg["forecast"]
    for key in ("calendar_file", "model_dir", "scores_dir"):
        assert Path(f[key]).is_absolute(), key
    assert Path(f["calendar_file"]).parent == Path(cfg["data"]["m5_dir"])


def test_load_items_columns(cfg):
    from forecast.data import ITEM_COLS, load_items
    if not (Path(cfg["data"]["processed_dir"]) / "items.csv").exists():
        pytest.skip("items.csv not built")
    items = load_items(cfg)
    assert list(items.columns) == ITEM_COLS
    assert len(items) == cfg["data"]["main"]["n_items"]
    assert int(items["is_perishable"].sum()) == cfg["data"]["main"]["n_perishable"]
    assert items["item_id"].is_unique


def test_closed_dates_are_dropped(tmp_path):
    """A closed day must also be dropped from training, or training and the 0 forecasts
    disagree (25 Dec report, decision 21). load_config refuses such a config."""
    text = CONFIG.read_text(encoding="utf-8")
    line = 'drop_dates: ["12-25"]'
    assert text.count(line) == 1
    bad = tmp_path / "config.yaml"
    bad.write_text(text.replace(line, "drop_dates: []"), encoding="utf-8")
    with pytest.raises(ValueError, match="closed_dates"):
        load_config(bad)
    load_config(CONFIG)                                    # the real config passes


def test_other_sections_untouched(cfg):
    """Adding forecast: must not change what the teammate's code reads."""
    from data_prep.prepare import load_config as prepare_load_config
    base = prepare_load_config(CONFIG)
    for key in ("data", "network", "stubs"):
        assert cfg[key] == base[key], key


# ---------------------------------------------------------------------------
# Task 2: baselines + scorer
# ---------------------------------------------------------------------------
FIRST = pd.Timestamp("2015-01-01")
SALES_FILE = ROOT / "data" / "processed" / "sales.parquet"
STUB_FILE = ROOT / "data" / "processed" / "forecasts_stub.parquet"


def tiny(units, store="CA_1", item="FOODS_3_001", start=FIRST):
    """One store-item with the given daily units from `start`."""
    days = pd.date_range(start, periods=len(units), freq="D")
    return pd.DataFrame({"date": days, "store_id": store, "item_id": item,
                         "units": np.asarray(units, dtype="int32")})


def one_target(day, store="CA_1", item="FOODS_3_001"):
    return pd.DataFrame({"date": [pd.Timestamp(day)], "store_id": [store], "item_id": [item]})


def test_baseline_hand_calc(cfg):
    units = np.arange(60) % 9                     # varied, known values
    sales = tiny(units)
    t = 50
    out = baseline_lines(sales, one_target(FIRST + pd.Timedelta(days=t)), cfg, method="naive")
    q50 = units[t - 7]
    sd = np.std(units[t - 34:t - 6], ddof=1)      # days t-34 .. t-7
    assert out["q50"].iloc[0] == pytest.approx(q50)
    assert out["q95"].iloc[0] == pytest.approx(q50 + 1.645 * sd)
    assert out["q80"].iloc[0] == pytest.approx(q50 + 0.84 * sd)


def test_stub_baseline_hand_calc(cfg):
    units = np.arange(60) % 9
    sales = tiny(units)
    t = 50
    out = baseline_lines(sales, one_target(FIRST + pd.Timedelta(days=t)), cfg, method="stub")
    q50 = np.mean([units[t - L] for L in (7, 14, 21, 28)])
    assert out["q50"].iloc[0] == pytest.approx(q50)


def test_baseline_ignores_last_six_days(cfg):
    """Rule R2: the baselines read nothing newer than 7 days before the target."""
    units = np.arange(60) % 9
    t = 50
    target = one_target(FIRST + pd.Timedelta(days=t))
    poisoned = units.copy()
    poisoned[t - 6:] = 1000
    for method in ("naive", "stub"):
        a = baseline_lines(tiny(units), target, cfg, method=method)
        b = baseline_lines(tiny(poisoned), target, cfg, method=method)
        pd.testing.assert_frame_equal(a, b)


def test_baseline_late_launch_no_nan(cfg):
    """An item with no rows before the target gets finite, zero lines (not NaN)."""
    sales = pd.concat([tiny(np.full(60, 5)),
                       tiny(np.full(5, 3), item="FOODS_1_001", start=FIRST + pd.Timedelta(days=55))])
    targets = pd.concat([one_target(FIRST + pd.Timedelta(days=56)),
                         one_target(FIRST + pd.Timedelta(days=56), item="FOODS_1_001")])
    for method in ("naive", "stub"):
        out = baseline_lines(sales, targets, cfg, method=method)
        lines = out[["q50", "q80", "q90", "q95"]]
        assert len(out) == 2 and np.isfinite(lines.to_numpy()).all()
        late = out[out["item_id"] == "FOODS_1_001"]
        assert (late[["q50", "q80", "q90", "q95"]] == 0).all(axis=None)


def test_baseline_lines_ordered(cfg):
    rng = np.random.default_rng(0)
    sales = tiny(rng.poisson(6, 90))
    targets = pd.concat([one_target(FIRST + pd.Timedelta(days=d)) for d in range(40, 90)])
    for method in ("naive", "stub"):
        out = baseline_lines(sales, targets, cfg, method=method)
        q = out[["q50", "q80", "q90", "q95"]].to_numpy()
        assert (np.diff(q, axis=1) >= 0).all() and (q >= 0).all()


@pytest.mark.skipif(not (SALES_FILE.exists() and STUB_FILE.exists()), reason="data not built")
def test_stub_baseline_matches_stub_file(cfg):
    """method="stub" reproduces the teammate's forecasts_stub.parquet exactly."""
    from forecast.data import load_sales
    sales = load_sales(cfg)
    stub = pd.read_parquet(STUB_FILE)
    first = stub[stub["origin"] == stub["origin"].min()]
    targets = first[["date", "store_id", "item_id"]].reset_index(drop=True)
    out = baseline_lines(sales, targets, cfg, method="stub")
    for q in ("q50", "q80", "q90", "q95"):
        np.testing.assert_allclose(out[q].to_numpy(), first[q].to_numpy(), rtol=0, atol=1e-9)


def test_pinball_example():
    """The two examples from forecast-training-plan.md section 1."""
    assert pinball(np.array([25.0]), np.array([23.0]), 0.95) == pytest.approx(1.9)
    assert pinball(np.array([15.0]), np.array([23.0]), 0.95) == pytest.approx(0.4)


def test_coverage_basic():
    assert coverage(np.array([1, 2, 3, 4]), np.array([2, 2, 2, 2])) == pytest.approx(50.0)
    assert mae(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)


def items_frame():
    return pd.DataFrame({
        "item_id": ["P1", "BIG", "SLOW", "NORM"],
        "dept_id": ["FOODS_3", "FOODS_2", "FOODS_1", "FOODS_1"],
        "rank": [5, 2, 100, 31],
        "avg_daily_units": [30.0, 50.0, 2.0, 8.0],
        "is_perishable": [True, False, False, False],
        "shelf_life_days": pd.array([2, None, None, None], dtype="Int64"),
    })


def test_service_line_rules(cfg):
    lines = service_line(items_frame(), cfg).set_index("item_id")["line"]
    assert lines.to_dict() == {"P1": "q95", "BIG": "q95", "SLOW": "q80", "NORM": "q90"}


def test_segments_have_used_line(cfg):
    sales = pd.DataFrame({"store_id": ["CA_1"] * 4, "item_id": ["P1", "BIG", "SLOW", "NORM"]})
    seg = segments(sales, items_frame(), cfg).set_index("item_id")
    assert set(seg["used_line"]) <= {"q80", "q90", "q95"}
    assert seg.loc["SLOW", "is_slow"] and not seg.loc["NORM", "is_slow"]
    assert seg.loc["P1", "is_perishable"]


def score_frames(dates, units, q):
    """Lines (all four = q) and actuals for item NORM at CA_1 on the given dates."""
    dates = pd.to_datetime(dates)
    key = {"date": dates, "store_id": "CA_1", "item_id": "NORM"}
    lines = pd.DataFrame({**key, "q50": q, "q80": q, "q90": q, "q95": q})
    actual = pd.DataFrame({**key, "units": units})
    return lines, actual


def test_cover_used_line(cfg):
    """NORM is stocked to q90: coverage of that line is reported against a 90% target."""
    seg = segments(pd.DataFrame({"store_id": ["CA_1"], "item_id": ["NORM"]}), items_frame(), cfg)
    q90 = np.array([5.0, 5.0, 5.0, 5.0])
    lines, actual = score_frames(["2015-07-01", "2015-07-02", "2015-07-03", "2015-07-04"],
                                 [4, 5, 6, 7], 5.0)
    lines["q90"] = q90
    s = score(lines, actual, seg, cfg).set_index("segment")
    assert s.loc["all", "cover_used_line"] == pytest.approx(50.0)
    assert s.loc["all", "used_line_target"] == pytest.approx(90.0)


def test_score_skips_closed_days(cfg):
    """Rows on closed_dates (25 Dec) change no score."""
    seg = segments(pd.DataFrame({"store_id": ["CA_1"], "item_id": ["NORM"]}), items_frame(), cfg)
    lines, actual = score_frames(["2014-12-24", "2014-12-26"], [3, 9], 5.0)
    with_xmas_l, with_xmas_a = score_frames(["2014-12-24", "2014-12-25", "2014-12-26"], [3, 0, 9], 5.0)
    a = score(lines, actual, seg, cfg)
    b = score(with_xmas_l, with_xmas_a, seg, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert int(b.set_index("segment").loc["all", "n"]) == 2


@pytest.mark.skipif(not SALES_FILE.exists(), reason="sales.parquet not built")
def test_targets_for_real_window(cfg):
    from forecast.data import load_sales
    sales = load_sales(cfg)
    t = targets_for(sales, pd.Timestamp("2015-07-01"), pd.Timestamp("2015-07-07"))
    assert list(t.columns) == ["date", "store_id", "item_id"]
    assert len(t) == 7 * 1200


# ---------------------------------------------------------------------------
# Task 3: review focus (late launches, zero prices, Christmas)
# ---------------------------------------------------------------------------
def feature_sales(n_days=90, start="2015-02-01"):
    """Two items at CA_1 with steady sales and prices, plus event names from calendar.csv."""
    days = pd.date_range(start, periods=n_days, freq="D")
    frames = [pd.DataFrame({"date": days, "store_id": "CA_1", "item_id": item, "dept_id": "FOODS_3",
                            "units": np.full(n_days, u, dtype="int32"), "price": p,
                            "is_perishable": False, "event_name": ""})
              for item, u, p in (("FOODS_3_001", 6, 2.5), ("FOODS_3_002", 3, 1.0))]
    return pd.concat(frames, ignore_index=True)


def test_late_launch_features_nan_not_error(cfg):
    """An item on sale for 3 days, and one never sold, still get a row of inputs: store data
    NaN, calendar and store filled in."""
    from forecast.features import build_features, fit_encoders
    sales = feature_sales()
    t = pd.Timestamp("2015-04-15")
    late = pd.DataFrame({"date": pd.date_range(t - pd.Timedelta(days=3), periods=3),
                         "store_id": "CA_1", "item_id": "FOODS_3_003", "dept_id": "FOODS_3",
                         "units": np.int32(4), "price": 3.0, "is_perishable": False,
                         "event_name": ""})
    sales = pd.concat([sales, late], ignore_index=True)
    enc = fit_encoders(sales, cutoff=t)
    targets = pd.concat([one_target(t, item=i) for i in ("FOODS_3_003", "FOODS_3_404")],
                        ignore_index=True)
    f = build_features(sales, targets, cfg, enc)
    assert len(f) == 2
    assert f[["lag_7", "lag_28", "roll_mean_28"]].isna().all(axis=None)
    assert f[["wday", "month", "store_id"]].notna().all(axis=None)


def test_zero_price_change_no_inf(cfg):
    """Weeks at price 0 give a finite price change (0), never inf or a division error."""
    from forecast.features import build_features, fit_encoders
    sales = feature_sales()
    sales.loc[sales["item_id"] == "FOODS_3_002", "price"] = 0.0
    enc = fit_encoders(sales, cutoff=pd.Timestamp("2015-04-01"))
    f = build_features(sales, one_target("2015-04-15", item="FOODS_3_002"), cfg, enc)
    assert f["price_chg_4w"].iloc[0] == 0.0
    assert np.isfinite(f["price_rel_item"].iloc[0])


def test_christmas_dropped_from_training_rows(cfg):
    from forecast.features import training_rows, zero_run_stats
    sales = feature_sales(n_days=60, start="2014-12-01")
    stats = zero_run_stats(sales, cfg, cutoff=pd.Timestamp("2015-01-30"))
    rows = training_rows(sales, pd.Timestamp("2014-12-01"), pd.Timestamp("2015-01-29"), cfg, stats)
    assert list(rows.columns) == ["date", "store_id", "item_id", "units"]
    assert not (rows["date"] == pd.Timestamp("2014-12-25")).any()
    assert len(rows) == 2 * (60 - 1)
