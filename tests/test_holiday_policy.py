"""Unit tests for the Holiday Surge & Valley Policy (CS2105)."""

from datetime import date
from copy import deepcopy
import pandas as pd
import pytest
import yaml

from replay.store_model import StoreModel
from optimize.parameters import load_case
from replay.cli import ROOT, _path


@pytest.fixture(scope="module")
def case_data():
    case = load_case(
        _path(ROOT / "data" / "processed"),
        _path(ROOT / "configs" / "validation_fast.yaml"),
        profile="full",
        forecast_path=_path(ROOT / "data" / "processed" / "forecasts.parquet"),
    )
    main_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    case.parameters.config["analysis"] = main_cfg["analysis"]
    case.parameters.config["replay"] = main_cfg["replay"]
    fc = pd.read_parquet(ROOT / "data" / "processed" / "forecasts.parquet")
    fc["origin"] = pd.to_datetime(fc["origin"]).dt.date
    fc["date"] = pd.to_datetime(fc["date"]).dt.date
    return case, fc


def test_holiday_policy_inactive_by_default(case_data):
    case, fc_raw = case_data
    cfg = deepcopy(case.parameters.config)
    cfg["analysis"]["store_ilp"]["layers"]["holiday_policy"] = {"enabled": False}

    model = StoreModel()
    keys = [("CA_1", "FOODS_3_318"), ("CA_1", "FOODS_1_004")]
    today = date(2015, 11, 24)  # Surge day
    window = [today, date(2015, 11, 25)]
    fc_today = fc_raw[fc_raw["origin"] == today].set_index(["store_id", "item_id", "date"])
    ctx = {"cfg": cfg, "fc": fc_today, "today": today}

    line_base, caps_base = model.lines_and_caps(ctx, keys, window)
    assert line_base[0] > 0
    assert line_base[1] > 0


def test_holiday_policy_active_phases(case_data):
    case, fc_raw = case_data
    cfg = deepcopy(case.parameters.config)
    cfg["analysis"]["store_ilp"]["layers"]["holiday_policy"] = {
        "enabled": True,
        "pre_surge_days": 3,
        "eve_run_to_zero": True,
        "perishable_throttle_days": 2,
        "calibration_file": "outputs/holiday_policy.json",
    }

    model = StoreModel()
    sensitive_item = "FOODS_3_318"  # High surge perishable
    dry_item = "FOODS_1_004"        # Dry grocery
    keys = [("CA_1", sensitive_item), ("CA_1", dry_item)]

    # 1. Phase 1: Surge day (2015-11-24)
    today_surge = date(2015, 11, 24)
    fc_surge = fc_raw[fc_raw["origin"] == today_surge].set_index(["store_id", "item_id", "date"])
    ctx_surge = {"cfg": cfg, "fc": fc_surge, "today": today_surge}
    line_surge, _ = model.lines_and_caps(ctx_surge, keys, [today_surge, date(2015, 11, 25)])

    cfg_off = deepcopy(cfg)
    cfg_off["analysis"]["store_ilp"]["layers"]["holiday_policy"]["enabled"] = False
    ctx_off = {"cfg": cfg_off, "fc": fc_surge, "today": today_surge}
    line_off, _ = model.lines_and_caps(ctx_off, keys, [today_surge, date(2015, 11, 25)])

    # Sensitive item target should be elevated by M_surge
    assert line_surge[0] > line_off[0]

    # 2. Phase 2: Surge on Eve (2015-11-25)
    today_eve = date(2015, 11, 25)
    fc_eve = fc_raw[fc_raw["origin"] == today_eve].set_index(["store_id", "item_id", "date"])
    ctx_eve = {"cfg": cfg, "fc": fc_eve, "today": today_eve}
    line_eve, _ = model.lines_and_caps(ctx_eve, keys, [today_eve])
    ctx_eve_off = {"cfg": cfg_off, "fc": fc_eve, "today": today_eve}
    line_eve_off, _ = model.lines_and_caps(ctx_eve_off, keys, [today_eve])
    assert line_eve[0] > line_eve_off[0]

    # 3. Phase 3: Slump / Throttle day (2015-11-27, Friday order arriving Saturday post-holiday)
    today_slump = date(2015, 11, 27)
    fc_slump = fc_raw[fc_raw["origin"] == today_slump].set_index(["store_id", "item_id", "date"])
    ctx_slump = {"cfg": cfg, "fc": fc_slump, "today": today_slump}
    line_slump, caps_slump = model.lines_and_caps(ctx_slump, keys, [today_slump, date(2015, 11, 28)])
    ctx_slump_off = {"cfg": cfg_off, "fc": fc_slump, "today": today_slump}
    line_slump_off, caps_slump_off = model.lines_and_caps(ctx_slump_off, keys, [today_slump, date(2015, 11, 28)])
    
    # Perishable target and spoilage cap should be aggressively scaled down by D_post to flush pipeline
    assert line_slump[0] < line_slump_off[0]
    assert caps_slump[0] < caps_slump_off[0]

    # Non-perishable has D_post completely removed: replenishment matches baseline exactly
    assert line_slump[1] == pytest.approx(line_slump_off[1])
    assert caps_slump[1] == float("inf")
