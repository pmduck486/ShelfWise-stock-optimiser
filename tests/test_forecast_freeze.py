"""Tests for Phase E: the frozen model (Task 8), forecast.predict and the forecasts.parquet
writer (Task 9), and the slide scores (Task 10).

Tiny made-up data as in test_forecast_model.py, so the file runs in seconds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast.data import load_config  # noqa: E402
from test_forecast_model import TINY_PARAMS, tiny_items, tiny_sales  # noqa: E402

CONFIG = ROOT / "config.yaml"
KEY = ["date", "store_id", "item_id"]
LINES = ["q50", "q80", "q90", "q95"]
CONTRACT = ["origin", "date", "store_id", "item_id"] + LINES
START = pd.Timestamp("2014-08-01")
CUT = pd.Timestamp("2014-12-15")


@pytest.fixture
def cfg(tmp_path):
    c = load_config(CONFIG)
    c["forecast"]["params"] = TINY_PARAMS
    c["forecast"]["line_params"] = {}
    c["forecast"]["model_dir"] = str(tmp_path / "model")
    c["forecast"]["scores_dir"] = str(tmp_path / "scores")
    c["forecast"]["freeze_platform"] = "any"
    c["data"]["windows"]["train_start"] = str(START.date())
    return c


@pytest.fixture
def sales(cfg):
    return tiny_sales(cfg, start=START, days=160)            # to 2015-01-07


def frozen(sales, cfg, cutoff=CUT, shifts=None):
    from forecast.freeze import freeze, load_frozen
    out = freeze(sales[sales["date"] < cutoff], cfg, cutoff, Path(cfg["forecast"]["model_dir"]),
                 shifts=shifts or {"q50": 0.0, "q80": 1.0, "q90": 1.0, "q95": 1.0})
    return out, load_frozen(out, cutoff, cfg)


# ---------------------------------------------------------------------------
# Task 8: freeze + manifest
# ---------------------------------------------------------------------------
def test_manifest_has_all_fields(sales, cfg):
    out, _ = frozen(sales, cfg)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for k in ["cutoff", "windows", "features", "quantiles", "params", "shifts", "seed",
              "sklearn_version", "config_sha256", "created", "platform", "train_rows"]:
        assert k in m, k
    assert m["cutoff"] == "2014-12-15" and m["quantiles"] == [0.5, 0.8, 0.9, 0.95]
    assert set(m["params"]) == set(LINES)
    assert (out / "bundle.joblib").exists()


def test_frozen_guard_refuses_late_cutoff(sales, cfg):
    from forecast.freeze import load_frozen
    out, _ = frozen(sales, cfg)
    load_frozen(out, CUT, cfg)                                  # same day is fine
    with pytest.raises(ValueError, match="cutoff"):
        load_frozen(out, CUT - pd.Timedelta(days=1), cfg)       # replay starts before the cutoff


def test_frozen_guard_refuses_changed_config(sales, cfg):
    from forecast.freeze import load_frozen
    out, _ = frozen(sales, cfg)
    cfg["forecast"]["min_lag_days"] = 8
    with pytest.raises(ValueError, match="config"):
        load_frozen(out, CUT, cfg)


def test_config_hash_same_in_any_folder(tmp_path):
    """The same config.yaml gives the same fingerprint wherever the project folder sits."""
    import shutil

    from forecast.freeze import config_sha256
    cfgs = []
    for folder in [tmp_path / "laptop" / "RetailOps", tmp_path / "clone" / "deeper" / "retailops"]:
        folder.mkdir(parents=True)
        shutil.copy(CONFIG, folder / "config.yaml")
        cfgs.append(load_config(folder / "config.yaml"))
    a, b = cfgs
    assert a["forecast"]["calendar_file"] != b["forecast"]["calendar_file"]   # absolute, different
    assert config_sha256(a) == config_sha256(b) == config_sha256(load_config(CONFIG))


def test_frozen_guard_refuses_other_sklearn(sales, cfg):
    from forecast.freeze import load_frozen
    out, _ = frozen(sales, cfg)
    p = out / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["sklearn_version"] = "0.0.1"
    p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="scikit-learn"):
        load_frozen(out, CUT, cfg)


def test_freeze_refuses_other_platform(sales, cfg):
    """Loophole 6 (user's call 2026-10-31): only the laptop makes the frozen model."""
    from forecast.freeze import freeze
    cfg["forecast"]["freeze_platform"] = "NoSuchOS"
    with pytest.raises(RuntimeError, match="laptop|platform"):
        freeze(sales[sales["date"] < CUT], cfg, CUT, Path(cfg["forecast"]["model_dir"]),
               shifts={"q50": 0.0, "q80": 1.0, "q90": 1.0, "q95": 1.0})


def test_config_freeze_platform_is_windows():
    assert load_config(CONFIG)["forecast"]["freeze_platform"] == "Windows"


def test_reproducible_same_seed(sales, cfg):
    from forecast.predict import predict
    _, b1 = frozen(sales, cfg)
    _, b2 = frozen(sales, cfg)
    a = predict(sales[sales["date"] < CUT], CUT, bundle=b1, cfg=cfg)
    b = predict(sales[sales["date"] < CUT], CUT, bundle=b2, cfg=cfg)
    pd.testing.assert_frame_equal(a, b)


def test_frozen_bundle_carries_shifts(sales, cfg):
    _, b = frozen(sales, cfg, shifts={"q50": 0.0, "q80": 1.2, "q90": 1.0, "q95": 1.0})
    assert b.shifts["q80"] == 1.2


# ---------------------------------------------------------------------------
# Task 9: predict + writer
# ---------------------------------------------------------------------------
def test_predict_contract_columns(sales, cfg):
    from forecast.predict import predict
    _, b = frozen(sales, cfg)
    out = predict(sales[sales["date"] < CUT], CUT, bundle=b, cfg=cfg)
    assert list(out.columns) == CONTRACT
    assert str(out["origin"].dtype) == "datetime64[ns]" and str(out["date"].dtype) == "datetime64[ns]"
    pairs = sales[["store_id", "item_id"]].drop_duplicates()
    H = cfg["forecast"]["horizon_days"]                     # 2: today and tomorrow
    assert len(out) == H * len(pairs)
    assert set(out["date"]) == set(pd.date_range(CUT, periods=H))
    assert (out["origin"] == CUT).all()


def test_future_poison_end_to_end(sales, cfg):
    """Sales on or after the origin are poisoned: identical output (they are never read)."""
    from forecast.predict import predict
    _, b = frozen(sales, cfg)
    origin = CUT + pd.Timedelta(days=5)
    poisoned = sales.copy()
    poisoned.loc[poisoned["date"] >= origin, "units"] = 999
    pd.testing.assert_frame_equal(predict(sales, origin, bundle=b, cfg=cfg),
                                  predict(poisoned, origin, bundle=b, cfg=cfg))


def test_christmas_rows_present_and_zero(sales, cfg):
    from forecast.predict import predict
    _, b = frozen(sales, cfg)
    out = predict(sales, pd.Timestamp("2014-12-24"), bundle=b, cfg=cfg)
    xmas = out[out["date"] == pd.Timestamp("2014-12-25")]
    assert len(xmas) == 10 and (xmas[LINES] == 0).all().all()


def test_sanity_full_output(sales, cfg):
    from forecast.predict import predict
    _, b = frozen(sales, cfg)
    out = predict(sales, pd.Timestamp("2014-12-20"), bundle=b, cfg=cfg)
    a = out[LINES].to_numpy()
    assert np.isfinite(a).all() and (a >= 0).all() and (np.diff(a, axis=1) >= 0).all()


def test_days_after_sales_end_have_calendar(sales, cfg):
    """Targets after the last sales day still get calendar inputs (R3): 2015-01-01 is NewYear."""
    from forecast.features import build_features
    from forecast.predict import predict
    _, b = frozen(sales, cfg)
    origin = pd.Timestamp("2014-12-31")
    out = predict(sales[sales["date"] < pd.Timestamp("2014-12-29")], origin, bundle=b, cfg=cfg)
    assert out["date"].max() == pd.Timestamp("2015-01-01") and len(out) == 20
    t = out[out["date"] == pd.Timestamp("2015-01-01")][KEY]
    X = build_features(sales[sales["date"] < origin], t, cfg, b.encoders)
    assert (X["is_event"] == 1).all()


def test_no_history_falls_back_to_baseline(sales, cfg):
    """A store-item with no rows before the origin gets the baseline (0 lines), counted."""
    from forecast.predict import predict
    _, b = frozen(sales, cfg)
    new = sales[(sales["item_id"] == "FOODS_3_000") & (sales["store_id"] == "CA_1")].copy()
    new["item_id"] = "FOODS_3_999"
    new = new[new["date"] >= CUT + pd.Timedelta(days=10)]          # launches after the origin
    out = predict(pd.concat([sales, new]), CUT, bundle=b, cfg=cfg,
                  pairs=pd.concat([sales, new])[["store_id", "item_id"]].drop_duplicates())
    nh = out[out["item_id"] == "FOODS_3_999"]
    assert len(nh) == cfg["forecast"]["horizon_days"] and (nh[LINES] == 0).all().all()
    assert out.attrs["fallback_pairs"] == 1


def test_closed_day_end_to_end(sales, cfg):
    """25 Dec everywhere: dropped from training rows, missing in inputs, all lines 0 in the
    output, left out of the score."""
    from forecast.features import build_features, training_rows, zero_run_stats
    from forecast.metrics import score, segments
    from forecast.predict import predict
    xmas = pd.Timestamp("2014-12-25")
    stats = zero_run_stats(sales, cfg, cutoff=pd.Timestamp("2015-01-08"))
    rows = training_rows(sales, START, pd.Timestamp("2015-01-07"), cfg, stats)
    assert not (rows["date"] == xmas).any()
    _, b = frozen(sales, cfg)
    t = sales[sales["date"] == pd.Timestamp("2015-01-01")][KEY]
    X = build_features(sales, t, cfg, b.encoders)
    assert X["lag_7"].isna().all()                              # 1 Jan's last week = 25 Dec: missing
    out = pd.concat([predict(sales, pd.Timestamp(o), bundle=b, cfg=cfg)
                     for o in ("2014-12-20", "2014-12-22", "2014-12-24")], ignore_index=True)
    out = out.drop_duplicates(["date", "store_id", "item_id"])   # 20..25 Dec, one row each
    assert (out.loc[out["date"] == xmas, LINES] == 0).all().all()
    seg = segments(sales, tiny_items(sales), cfg)
    lines = out.drop(columns="origin")
    lines.loc[lines["date"] == xmas, LINES] = 1000.0             # would wreck the score if read
    s = score(lines, sales[KEY + ["units"]], seg, cfg)
    assert s.loc[s["segment"] == "all", "n"].iloc[0] == 5 * 10              # 20-24 Dec open, 25 Dec left out


def test_write_forecasts_and_fast_path(sales, cfg, tmp_path):
    """The writer covers every origin x horizon days x store-item; forecasts_for(o) equals predict."""
    from forecast.predict import predict
    from forecast.write_forecasts import forecasts_for, write_forecasts
    _, b = frozen(sales, cfg)
    origins = pd.date_range(CUT, periods=3)
    path = tmp_path / "fc.parquet"
    write_forecasts(sales, origins, cfg, path, bundle=b)
    fc = pd.read_parquet(path)
    assert list(fc.columns) == CONTRACT and len(fc) == 3 * cfg["forecast"]["horizon_days"] * 10
    for o in origins[[0, 2]]:
        pd.testing.assert_frame_equal(forecasts_for(o, cfg, path=path),
                                      predict(sales, o, bundle=b, cfg=cfg))


def test_writer_refuses_other_platform(sales, cfg, tmp_path):
    from forecast.write_forecasts import write_forecasts
    _, b = frozen(sales, cfg)
    cfg["forecast"]["freeze_platform"] = "NoSuchOS"
    with pytest.raises(RuntimeError, match="laptop|platform"):
        write_forecasts(sales, [CUT], cfg, tmp_path / "fc.parquet", bundle=b)


def test_contract_matches_stub():
    """On the written file (laptop only): same columns and dtypes as the stub; keys = the stub's
    first horizon_days (2) days of every origin (the stub covers 7)."""
    cfg = load_config(CONFIG)
    d = Path(cfg["data"]["processed_dir"])
    if not (d / "forecasts.parquet").exists():
        pytest.skip("forecasts.parquet is written on the laptop only (loophole 6)")
    fc, stub = pd.read_parquet(d / "forecasts.parquet"), pd.read_parquet(d / "forecasts_stub.parquet")
    assert list(fc.columns) == list(stub.columns)
    assert (fc.dtypes == stub.dtypes).all()
    k = ["origin", "date", "store_id", "item_id"]
    H = cfg["forecast"]["horizon_days"]
    stub = stub[stub["date"] - stub["origin"] < pd.Timedelta(days=H)]
    assert len(fc) == len(stub) == 470_400 // 7 * H
    pd.testing.assert_frame_equal(fc[k].reset_index(drop=True), stub[k].reset_index(drop=True))
    a = fc[LINES].to_numpy()
    assert np.isfinite(a).all() and (a >= 0).all() and (np.diff(a, axis=1) >= 0).all()


# ---------------------------------------------------------------------------
# Task 10: scores for the slides
# ---------------------------------------------------------------------------
def test_report_runs_on_saved_scores(tmp_path):
    """Reads the saved score files and writes summary.md and three charts."""
    from forecast.report import write_report
    cfg = load_config(CONFIG)
    scores = Path(cfg["forecast"]["scores_dir"])
    need = ["baseline_val.json", "stub_val.json", "model_val_untuned.json", "model_val_calibrated.json"]
    if not all((scores / n).exists() for n in need):
        pytest.skip("needs the saved validation scores")
    out = write_report(cfg, scores_dir=scores, out_dir=tmp_path, example=None)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "MAE" in text and "coverage" in text.lower() and "25 Dec" in text
    for name in ["mae_vs_baselines.png", "coverage.png"]:
        assert (tmp_path / name).stat().st_size > 1000
    assert out == tmp_path / "summary.md"
