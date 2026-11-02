"""Store-only mode and the simplified store order model (decided 2026-11-02).

Uses the two-by-two fixture of test_replay.py with replay.warehouse off and a 2-day forecast
(today and tomorrow), plus one perishable product for the spoilage cap.
"""

from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from optimize import OptimizerPolicy
from replay import run
from replay.baselines import make_policy
from replay.store_model import StoreModel, stock_level
from tests.test_replay import (D1, LINES, MAIN_CFG, make_cfg, make_forecasts, make_sales, make_state,
                               run_fixture)


def store_cfg(**kw):
    cfg = make_cfg(**kw)
    cfg['replay']['warehouse'] = 'off'
    cfg['replay']['forecast_days'] = 2
    return cfg


def two_day(fc):
    fc = fc.copy()
    keep = pd.to_datetime(fc.date) - pd.to_datetime(fc.origin) < pd.Timedelta(days=2)
    return fc[keep].reset_index(drop=True)


def run_store(method, days=3, start=D1, cfg=None, **kw):
    cfg = cfg or store_cfg()
    end = start + timedelta(days=days - 1)
    return run_fixture(method, days=days, start=start, cfg=cfg,
                       forecasts=two_day(make_forecasts(start, end)), keep_decisions=True, **kw)


# ---------------------------------------------------------------- config

def test_config_store_only_and_two_day_forecast():
    f, r = MAIN_CFG['forecast'], MAIN_CFG['replay']
    assert r['warehouse'] in (False, 'off')
    assert f['horizon_days'] == r['forecast_days'] == 2
    assert f['horizon_days'] <= f['min_lag_days']           # every input known on the origin's morning
    assert min(f['lags']) >= f['min_lag_days']
    assert 'store_ilp' in r['methods'] and 'store_model' in r['methods'] and 'optimizer' not in r['methods']


# ---------------------------------------------------------------- engine in store-only mode

@pytest.mark.parametrize('method', ['last_week', 'topup_avg', 'topup_forecast', 'store_model'])
def test_store_only_never_buys_from_suppliers_and_passes_rules_and_audit(method, tmp_path):
    daily, k = run_store(method, output_dir=tmp_path / method)
    assert k['rules']['passed']
    assert k["independent_audit"]
    assert k['supplier_units'] == 0 and k['supplier_order_lines'] == 0
    assert k['avg_warehouse_stock_value'] is None
    assert k['avg_stock_value'] == pytest.approx(k['avg_store_stock_value'])
    assert k['dc_short_lines'] == 0


def test_store_only_charges_goods_when_they_reach_a_store(tmp_path):
    daily, k = run_store('topup_forecast', output_dir=tmp_path / 'm')
    sent = sum(d.store_orders.units.sum() for d in k['_decisions'])     # buy cost is 1.0 in the fixture
    assert k['cost_breakdown']['purchase_cash'] == pytest.approx(sent)
    assert k['total_cost'] == pytest.approx(sum(k['cost_breakdown'].values()))
    # warehouse holding left out: stores' holding only (0.01 per unit-day)
    led = pd.read_csv(tmp_path / 'm' / 'inventory_ledger.csv')
    stores = led[led.location_type == 'store']
    assert k['cost_breakdown']['holding_cash'] == pytest.approx(stores.holding_cost.sum())


def test_store_only_order_is_never_cut_by_the_warehouse():
    cfg = store_cfg()
    state = make_state(wa=0, wb=0, pipeline=False)         # an empty warehouse is ignored in store-only mode
    _, k = run_store('topup_forecast', cfg=cfg, state=state)
    first = k['_decisions'][0].store_orders.set_index(['store_id', 'item_id']).cases
    assert first[('S2', 'A')] > 0


def test_optimizer_refused_in_store_only_mode():
    with pytest.raises(ValueError, match='warehouse on'):
        run(OptimizerPolicy(), make_sales(D1 - timedelta(days=60), D1), D1, None, store_cfg(),
            state=make_state(), forecasts=two_day(make_forecasts(D1, D1)), end=D1)


def test_store_only_refuses_a_forecast_missing_tomorrow():
    fc = make_forecasts(D1, D1 + timedelta(days=2))
    fc = fc[pd.to_datetime(fc.date) == pd.to_datetime(fc.origin)]
    with pytest.raises(ValueError, match='Incomplete forecast horizon'):
        run_fixture('store_model', cfg=store_cfg(), forecasts=fc)


# ---------------------------------------------------------------- the store model's targets

def _ctx(cfg, keys, lines, today=D1, days=2):
    rows = [dict(store_id=s, item_id=p, date=today + timedelta(days=t), **lines[(s, p)])
            for (s, p) in keys for t in range(days)]
    fc = pd.DataFrame(rows).set_index(['store_id', 'item_id', 'date'])
    return dict(today=today, cfg=cfg, fc=fc, store_keys=keys)


def test_long_life_target_is_q95_over_the_cover_days():
    cfg = store_cfg()
    keys = [('S1', 'A'), ('S2', 'A')]
    ctx = _ctx(cfg, keys, LINES)
    window = [D1, D1 + timedelta(days=1)]
    got = StoreModel().store_targets(ctx, keys, window)
    np.testing.assert_allclose(got, [2 * LINES[('S1', 'A')]['q95'], 2 * LINES[('S2', 'A')]['q95']])


def test_perishable_level_cap_and_nearest_carton():
    cfg = store_cfg()
    items = cfg['order_model']['parameters']['items']
    items['P'] = dict(units_per_case=4, cases_per_pallet=2, price=2.0, buy_cost=1.0,
                      holding_cost_per_unit_day=0.01, service_level=0.8, shelf_life_days=2)
    level = stock_level(cfg, 'P')
    om = cfg['order_model']
    cu, co = om['lost_sale_penalty_share'] * 2.0, 1.0 * (1 + om['waste_disposal_share'])
    assert level == pytest.approx(cu / (cu + co))
    window = [D1, D1 + timedelta(days=1)]
    # wide spread: the level line is above the cap, so the cap (q50 today..today+life) binds
    wide = {('S1', 'P'): dict(q50=10.0, q80=30.0, q90=40.0, q95=50.0)}
    t = StoreModel().store_targets(_ctx(cfg, [('S1', 'P')], wide), [('S1', 'P')], window)[0]
    line = 2 * (10 + (level - 0.5) / 0.3 * 20)
    cap = 10 + 10 + 10                       # days 0, 1 from the forecast, day 2 = their mean
    assert min(line, cap) == cap
    assert t == pytest.approx(cap - 0.5 * 4)
    # narrow spread: the level line binds
    narrow = {('S1', 'P'): dict(q50=10.0, q80=11.0, q90=12.0, q95=13.0)}
    t = StoreModel().store_targets(_ctx(cfg, [('S1', 'P')], narrow), [('S1', 'P')], window)[0]
    assert t == pytest.approx(2 * (10 + (level - 0.5) / 0.3 * 1) - 2)


def test_closed_day_counts_zero_and_needs_no_forecast():
    cfg = store_cfg()
    keys = [('S1', 'A')]
    today = date(2015, 12, 23)
    ctx = _ctx(cfg, keys, LINES, today=today)
    window = [today, today + timedelta(days=1), today + timedelta(days=2)]      # 25 Dec closed, not in fc
    got = StoreModel().store_targets(ctx, keys, window)
    assert got[0] == pytest.approx(2 * LINES[('S1', 'A')]['q95'])


def test_store_model_runs_through_christmas(tmp_path):
    start = date(2015, 12, 22)
    _, k = run_store('store_model', days=5, start=start, output_dir=tmp_path / 'xmas')
    assert k['rules']['passed']
