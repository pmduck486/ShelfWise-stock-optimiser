"""L4 replay tests (plan/l4-replay-plan.md). Small synthetic fixtures; the real
`replay:` settings come from config.yaml so the tests check what the runs use.

Fixture "two-by-two" (used by most tests):
    stores S1, S2 -> one warehouse W; products A, B; nothing perishable.
    A: 10 units/carton, 2 cartons/pallet (pallet = 20 units), service 0.90 (line q90, z 1.2816)
    B:  5 units/carton, 2 cartons/pallet (pallet = 10 units), service 0.80 (line q80, z 0.8416)
    Daily demand, both in the past and in the replay (recorded sales = demand):
        S1,A = 10   S2,A = 20   S2,B = 0
        S1,B = b(day) = 2 if day.toordinal() is even else 4   (alternates; 2015-11-10 is even)
    Forecast lines (every origin, every day): S1,A q90 12; S2,A q90 24; S1,B q80 4; S2,B 0.
    Start state on d1 = 2015-11-10: stores S1,A 15, S2,A 30, S1,B 1, S2,B 0;
        warehouse W,A 200, W,B 20; pipeline: 10 units of A to S1 arriving d1,
        40 units of A to W arriving d1+2. Budget 1e6, truck 400 cartons, minimum 1 carton / 1 pallet.
"""

from copy import deepcopy
from datetime import date, timedelta
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from optimize import Decision, OptimizerPolicy, State
from optimize.policy import PIPELINE_COLUMNS
from optimize.replay import execute_day
from replay import run
from replay.baselines import (POLICIES, make_policy, scale_to_budget, split_whole_cartons,
                              store_cover_window, warehouse_cover_window)
from replay.history import PastSales
from replay.kpis import ROADMAP_KEYS, compute_kpis, check_rules

ROOT = Path(__file__).resolve().parents[1]
MAIN_CFG = yaml.safe_load((ROOT / 'config.yaml').read_text())
D1 = date(2015, 11, 10)
STORES, ITEMS = ['S1', 'S2'], ['A', 'B']
BASELINES = ['last_week', 'topup_avg', 'topup_forecast']


def b(day):
    return 2 if day.toordinal() % 2 == 0 else 4


def demand(store, item, day):
    if day.strftime('%m-%d') == '12-25':
        return 0                        # stores closed; M5 records ~0 then
    fixed = {('S1', 'A'): 10, ('S2', 'A'): 20, ('S2', 'B'): 0}
    return fixed[(store, item)] if (store, item) in fixed else b(day)


LINES = {('S1', 'A'): dict(q50=10, q80=11, q90=12, q95=14),
         ('S2', 'A'): dict(q50=20, q80=22, q90=24, q95=27),
         ('S1', 'B'): dict(q50=3, q80=4, q90=5, q95=6),
         ('S2', 'B'): dict(q50=0, q80=0, q90=0, q95=0)}


def make_cfg(budget=1e6, store_min=1, dc_min=1):
    om = deepcopy(MAIN_CFG['order_model'])
    om['solver'].update(time_limit_s=15, feasibility_tolerance=1e-6)
    om.update(store_min_order_cases=store_min, dc_min_order_pallets=dc_min, truck_trip_cost=1.0,
              order_line_cost=0.1)
    item = lambda upc, level, price: dict(units_per_case=upc, cases_per_pallet=2, price=price,
                                          buy_cost=1.0, holding_cost_per_unit_day=0.01,
                                          service_level=level, shelf_life_days=None)
    om['parameters'] = dict(items={'A': item(10, 0.9, 2.0), 'B': item(5, 0.8, 3.0)},
                            store_averages={'S1': {'A': 10.0, 'B': 3.0}, 'S2': {'A': 20.0, 'B': 0.0}},
                            dc_budgets={'W': budget})
    rp = deepcopy(MAIN_CFG['replay'])
    rp['warehouse'] = 'on'              # these tests check the full chain (kept for later, 2026-11-02);
                                        #   store-only mode is tested in test_store_only.py
    return dict(data=deepcopy(MAIN_CFG['data']), network=dict(dcs=['W'], home_dc={'S1': 'W', 'S2': 'W'}),
                order_model=om, replay=rp)


def make_state(day=D1, s1a=15, s2a=30, s1b=1, s2b=0, wa=200, wb=20, pipeline=True):
    pipe = pd.DataFrame(columns=PIPELINE_COLUMNS)
    if pipeline:
        pipe = pd.DataFrame([
            dict(order_date=day - timedelta(days=1), arrival_date=day, location_type='store',
                 location_id='S1', item_id='A', units=10.0),
            dict(order_date=day - timedelta(days=3), arrival_date=day + timedelta(days=2),
                 location_type='dc', location_id='W', item_id='A', units=40.0)], columns=PIPELINE_COLUMNS)
    store = pd.DataFrame([dict(store_id=s, item_id=p, units=float(u)) for (s, p), u in
                          [(('S1', 'A'), s1a), (('S1', 'B'), s1b), (('S2', 'A'), s2a), (('S2', 'B'), s2b)]])
    dc = pd.DataFrame([dict(dc_id='W', item_id='A', units=float(wa)), dict(dc_id='W', item_id='B', units=float(wb))])
    return State(day, store, dc, pipe)


def make_sales(first, last):
    days = pd.date_range(first, last).date
    return pd.DataFrame([dict(date=d, store_id=s, item_id=p, units=demand(s, p, d))
                         for d in days for s in STORES for p in ITEMS])


def make_forecasts(first, last):
    rows = []
    for o in pd.date_range(first, last).date:
        for t in range(7):
            day = o + timedelta(days=t)
            for (s, p), q in LINES.items():
                q = {k: 0.0 if day.strftime('%m-%d') == '12-25' else float(v) for k, v in q.items()}
                rows.append(dict(origin=o, date=day, store_id=s, item_id=p, **q))
    return pd.DataFrame(rows)


def run_fixture(method, days=3, start=D1, cfg=None, state=None, sales=None, forecasts=None, **kw):
    cfg = cfg or make_cfg()
    end = start + timedelta(days=days - 1)
    sales = make_sales(start - timedelta(days=60), end) if sales is None else sales
    forecasts = make_forecasts(start, end) if forecasts is None else forecasts
    return run(make_policy(method), sales, start, None, cfg, state=state or make_state(start),
               forecasts=forecasts, end=end, method=method, **kw)


def orders_by_day(decisions):
    """{day: (store cartons [S1A, S2A, S1B, S2B], supplier pallets [A, B])}"""
    out = {}
    for d in decisions:
        so = d.store_orders.set_index(['store_id', 'item_id']).cases
        po = d.dc_orders.set_index(['dc_id', 'item_id']).pallets
        out[d.date] = ([int(so[k]) for k in [('S1', 'A'), ('S2', 'A'), ('S1', 'B'), ('S2', 'B')]],
                       [int(po[('W', 'A')]), int(po[('W', 'B')])])
    return out


# ---------------------------------------------------------------- config

def test_replay_section_present_and_other_sections_untouched():
    cfg = MAIN_CFG
    fast = yaml.safe_load((ROOT / 'configs' / 'validation_fast.yaml').read_text())
    for section in ['data', 'network', 'stubs']:
        assert cfg[section] == fast[section]
    om = deepcopy(cfg['order_model']); om['solver'].pop('time_limit_s'); om['solver'].pop('parallel_time_reserve_s')
    fom = deepcopy(fast['order_model']); fom['solver'].pop('time_limit_s'); fom['solver'].pop('parallel_time_reserve_s')
    assert om == fom
    r = cfg['replay']
    assert r['methods'] == ['last_week', 'topup_avg', 'topup_forecast', 'store_model', 'store_ilp']
    assert r['warehouse'] in (False, 'off') and r['forecast_days'] == MAIN_CFG['forecast']['horizon_days']
    assert {lvl for lvl, _ in r['service_line_by_level']} == {0.5, 0.8, 0.9, 0.95}
    assert set(r['z_by_line']) == {'q50', 'q80', 'q90', 'q95'}
    from forecast.data import load_config
    from forecast.freeze import config_sha256
    manifest = json.loads((ROOT / 'outputs' / 'forecast' / 'model' / 'manifest.json').read_text())
    assert config_sha256(load_config(ROOT / 'config.yaml')) == manifest['config_sha256']


# ---------------------------------------------------------------- pure helpers

def test_cover_windows_normal_and_around_christmas():
    cfg = make_cfg()
    assert store_cover_window(D1, cfg) == (True, [D1, D1 + timedelta(days=1)])
    can, w = store_cover_window(date(2015, 12, 23), cfg)
    assert can and w == [date(2015, 12, 23), date(2015, 12, 24), date(2015, 12, 25)]
    can, _ = store_cover_window(date(2015, 12, 24), cfg)
    assert not can                                   # it would arrive on 25 Dec
    assert store_cover_window(date(2015, 12, 25), cfg) == (True, [date(2015, 12, 25), date(2015, 12, 26)])
    can, w = warehouse_cover_window(D1, cfg)
    assert can and w == [D1 + timedelta(days=k) for k in range(6)]


def test_split_whole_cartons_largest_remainder():
    assert split_whole_cartons([1, 2], 5) == [1, 2]          # enough: unchanged
    assert split_whole_cartons([1, 2], 2) == [1, 1]          # 0.667 / 1.333 -> 0,1 + remainder to S1
    assert split_whole_cartons([3, 3, 1], 4) == [2, 2, 0]    # 1.71,1.71,0.57 -> 1,1,0 + 2 remainders
    assert split_whole_cartons([2, 2], 1) == [1, 0]          # tie: first store
    assert split_whole_cartons([4, 0], 0) == [0, 0]


def test_scale_to_budget_whole_pallets():
    # spend 3*20 + 2*10 = 80 > 50: factor 0.625 -> 1.875, 1.25 -> floors 1, 1 (spend 30);
    # leftover 20 handed out by largest remainder: line 0 (0.875) +1 -> spend 50; line 1 (+10) no longer fits.
    assert scale_to_budget([3, 2], [20.0, 10.0], 50.0) == ([2, 1], 0.625)
    assert scale_to_budget([3, 2], [20.0, 10.0], 80.0) == ([3, 2], 1.0)
    assert scale_to_budget([1], [20.0], 5.0) == ([0], 0.25)
    # every line 1 pallet, budget for 2 of 3: floors all 0, remainders tie -> first lines that fit
    assert scale_to_budget([1, 1, 1], [10.0, 30.0, 10.0], 25.0) == ([1, 0, 1], 0.5)
    # never above the request, never above the budget
    got, _ = scale_to_budget([2, 5, 1], [7.0, 3.0, 11.0], 20.0)
    assert all(g <= r for g, r in zip(got, [2, 5, 1])) and sum(g * c for g, c in zip(got, [7.0, 3.0, 11.0])) <= 20.0


def test_past_sales_refuses_today_and_later():
    sales = make_sales(D1 - timedelta(days=40), D1 + timedelta(days=3))
    past = PastSales(sales, D1, days=35, missing_units=0)
    assert past.matrix([D1 - timedelta(days=1)], [('S1', 'A')]).tolist() == [[10.0]]
    with pytest.raises(ValueError, match='not before'):
        past.matrix([D1], [('S1', 'A')])
    with pytest.raises(ValueError, match='older than'):
        past.matrix([D1 - timedelta(days=36)], [('S1', 'A')])
    gap = sales[~((sales.date == D1 - timedelta(days=2)) & (sales.store_id == 'S1') & (sales.item_id == 'A'))]
    assert PastSales(gap, D1, 35, 0).matrix([D1 - timedelta(days=2)], [('S1', 'A')]).tolist() == [[0.0]]


# ---------------------------------------------------------------- 3-day hand-worked case

HAND = {
    # last_week: a day's demand = sales 7 days earlier. Store cover = today+tomorrow (2 days);
    # warehouse cover = 6 days (today..today+5). Position = stock + everything on the way.
    # d1: S1A target 10+10=20, pos 15+10=25 -> 0.  S2A 40 vs 30 -> need 10 -> 1 carton.
    #     S1B b(d1-7)+b(d1-6)=4+2=6 vs 1 -> need 5 -> 1 carton.  S2B 0.
    #     W,A pos 200-10+40=230 vs 6*30=180 -> 0.  W,B pos 20-5=15 vs b(d1-7..d1-2)=18 -> need 3 -> 1 pallet.
    #     Sales d1: S1A 25->15, S2A 30->10, S1B 1 (demand 2, 1 lost) -> 0.
    # d2: S1A 20 vs 15 -> 1. S2A 40 vs 10+10 -> 2.  S1B 2+4=6 vs 0+5 -> 1.
    #     W,A 190-30+40=200 vs 180 -> 0.  W,B 15-5+10=20 vs 18 -> 0.
    #     Sales d2: S1A 15->5, S2A 20->0, S1B 5 (demand 4) -> 1.
    # d3: S1A 20 vs 5+10 -> 1. S2A 40 vs 0+20 -> 2. S1B 4+2=6 vs 1+5 -> 0.
    #     W,A (160+40 arrived)-30=170 vs 180 -> need 10 -> 1 pallet.  W,B 10+10=20 vs 18 -> 0.
    'last_week': [([0, 1, 1, 0], [0, 1]), ([1, 2, 1, 0], [0, 0]), ([1, 2, 0, 0], [1, 0])],
    # topup_avg: target = mu*D + z*sigma*sqrt(D), 28 days ending yesterday, sample std.
    # S1A mu 10 sigma 0 -> 20; S2A -> 40; S2B 0.  S1B: 14 twos + 14 fours -> mu 3,
    # sigma = sqrt(28/27) = 1.018350; store D=2: 6 + 0.8416*1.018350*1.414214 = 7.21205.
    # Warehouse D=6: A summed series 30, sigma 0 -> 180;  B: 18 + 0.8416*1.018350*2.449490 = 20.09929.
    # d1: S1A 0, S2A 1, S1B need 7.212-1=6.212 -> 2 cartons (10).  W,A 0.  W,B pos 20-10=10 -> need 10.099 -> 2 pallets.
    #     Sales d1: S1B 1 (1 lost) -> 0, 10 on the way.
    # d2: S1A 1, S2A 2, S1B pos 10 >= 7.212 -> 0.  W,A 200 -> 0.  W,B 10+20=30 -> 0.  Sales: S1B 10-4=6.
    # d3: S1A 1, S2A 2, S1B need 1.212 -> 1.  W,A 170 vs 180 -> 1 pallet.  W,B 10-5+20=25 -> 0.
    'topup_avg': [([0, 1, 2, 0], [0, 2]), ([1, 2, 0, 0], [0, 0]), ([1, 2, 1, 0], [1, 0])],
    # topup_forecast: target = the product's service line summed over the cover days
    # (store 2 days: S1A 24, S2A 48, S1B 8; warehouse 6 days: A (12+24)*6=216, B 4*6=24).
    # d1: S1A 25 -> 0. S2A need 48-30=18 -> 2. S1B need 7 -> 2.  W,A 200-20+40=220 -> 0.  W,B 20-10=10 -> need 14 -> 2.
    #     Sales: S1A 15, S2A 10 (+20 coming), S1B 0 (+10 coming).  W,A 180, W,B 10.
    # d2: S1A need 9 -> 1. S2A 48-30=18 -> 2. S1B 10 -> 0.  W,A 180-30+40=190 -> need 26 -> 2 pallets.  W,B 10+20=30 -> 0.
    #     Sales: S1A 5, S2A 10, S1B 6.  W,A 150 (+40 d3, +40 d7).
    # d3: S1A 24-15 -> 1. S2A 48-30 -> 2. S1B 8-6=2 -> 1.  W,A 150+40-30+40=200 -> need 16 -> 1.  W,B 10-5+20=25 -> 0.
    'topup_forecast': [([0, 2, 2, 0], [0, 2]), ([1, 2, 0, 0], [2, 0]), ([1, 2, 1, 0], [1, 0])],
}


@pytest.mark.parametrize('method', BASELINES)
def test_three_day_hand_worked_case(method):
    assert b(D1) == 2
    daily, kpis = run_fixture(method, keep_decisions=True)
    got = orders_by_day(kpis['_decisions'])
    expected = {D1 + timedelta(days=k): v for k, v in enumerate(HAND[method])}
    assert got == expected
    for d in kpis['_decisions']:
        trucks = d.deliveries.set_index('store_id').trucks
        cartons = d.store_orders.groupby('store_id').cases.sum()
        for s in STORES:
            assert trucks[s] == math.ceil(cartons[s] / 400)


# ---------------------------------------------------------------- whole packs, minimums, limits

@pytest.mark.parametrize('method', BASELINES)
def test_whole_packs_never_negative_and_minimum_order(method):
    cfg = make_cfg(store_min=2, dc_min=2)
    daily, kpis = run_fixture(method, cfg=cfg, days=4, keep_decisions=True)
    om = cfg['order_model']; items = om['parameters']['items']
    for d in kpis['_decisions']:
        for frame, n, pack in [(d.store_orders, 'cases', lambda p: items[p]['units_per_case']),
                               (d.dc_orders, 'pallets', lambda p: items[p]['units_per_case'] * 2)]:
            assert (frame[n] >= 0).all()
            assert (frame[n] == np.round(frame[n])).all()
            assert ((frame[n] == 0) | (frame[n] >= 2)).all()
            assert (frame.units == frame[n] * frame.item_id.map(pack)).all()
    # with a minimum of 2 the 1-carton S2A order of the hand case becomes 2
    first = kpis['_decisions'][0].store_orders.set_index(['store_id', 'item_id']).cases
    assert first[('S2', 'A')] == 2


@pytest.mark.parametrize('method', BASELINES)
def test_warehouse_never_dispatches_more_than_it_has(method):
    # W,A holds 25 units = 2 whole cartons; both stores want A on d1/d2.
    state = make_state(s1a=0, s2a=0, wa=25, pipeline=False)
    daily, kpis = run_fixture(method, state=state, days=2, keep_decisions=True)
    first = kpis['_decisions'][0].store_orders.set_index(['store_id', 'item_id']).cases
    assert first[('S1', 'A')] + first[('S2', 'A')] == 2       # all whole cartons it has, no more
    assert first[('S1', 'A')] == 1 and first[('S2', 'A')] == 1   # proportional (S2 asked twice as much)
    assert kpis['rules']['passed'] and kpis['rules']['dispatch_within_warehouse_stock'] > 0
    assert kpis['_decisions'][0].audit['dc_short_lines'] == 1


@pytest.mark.parametrize('method', BASELINES)
def test_budget_never_exceeded(method):
    cfg = make_cfg(budget=15.0)        # one B pallet costs 10, one A pallet 20
    state = make_state(wa=0, wb=0, pipeline=False)
    daily, kpis = run_fixture(method, cfg=cfg, state=state, days=2, keep_decisions=True)
    for d in kpis['_decisions']:
        spend = (d.dc_orders.units * 1.0).sum()
        assert spend <= 15.0
        assert d.audit['budget_factor']['W'] < 1.0
    assert kpis['budget_scaled_days'] == 2


@pytest.mark.parametrize('method', BASELINES)
def test_no_store_arrival_on_christmas(method):
    start = date(2015, 12, 22)
    daily, kpis = run_fixture(method, start=start, days=5, state=make_state(start), keep_decisions=True)
    by_day = {d.date: d for d in kpis['_decisions']}
    assert by_day[date(2015, 12, 24)].store_orders.cases.sum() == 0
    assert by_day[date(2015, 12, 24)].deliveries.trucks.sum() == 0
    if method == 'last_week':
        # 23 Dec covers 23-25 Dec; 25 Dec counts 0: S1A target = sales(16 Dec)+sales(17 Dec) = 20
        assert store_cover_window(date(2015, 12, 23), make_cfg())[1][-1] == date(2015, 12, 25)
    assert bool(daily.set_index('date').loc['2015-12-25', 'closed'])
    assert daily.set_index('date').loc['2015-12-25', 'store_receipts'] == 0


# ---------------------------------------------------------------- fairness

@pytest.mark.parametrize('method', BASELINES + ['optimizer'])
def test_future_poison(method):
    """Scramble every sale from d3 onward (and every forecast with another origin than
    the day): d1-d3 decisions must not change."""
    sales = make_sales(D1 - timedelta(days=60), D1 + timedelta(days=2))
    poisoned = sales.copy()
    rng = np.random.default_rng(7)
    late = poisoned.date >= D1 + timedelta(days=2)
    poisoned.loc[late, 'units'] = rng.integers(0, 500, late.sum())
    _, clean = run_fixture(method, sales=sales, keep_decisions=True)
    _, dirty = run_fixture(method, sales=poisoned, keep_decisions=True)
    for a, c in zip(clean['_decisions'], dirty['_decisions']):
        pd.testing.assert_frame_equal(a.store_orders, c.store_orders)
        pd.testing.assert_frame_equal(a.dc_orders, c.dc_orders)


def test_identical_start_state_for_all_methods(tmp_path):
    state = make_state()
    saved = deepcopy(state)
    hashes = set()
    for method in BASELINES + ['optimizer']:
        _, k = run_fixture(method, days=2, state=state)
        hashes.add(k['start_state_sha256'])
    assert len(hashes) == 1
    for name in ['store_stock', 'dc_stock', 'pipeline']:
        pd.testing.assert_frame_equal(getattr(state, name), getattr(saved, name))


def test_outside_test_window_refused():
    with pytest.raises(ValueError, match='holdout'):
        run_fixture('last_week', start=date(2015, 12, 25), days=3, state=make_state(date(2015, 12, 25)))
    with pytest.raises(ValueError, match='test window'):
        run_fixture('last_week', start=date(2015, 10, 30), days=2, state=make_state(date(2015, 10, 30)))
    sales = make_sales(D1 - timedelta(days=60), D1 + timedelta(days=2))
    leak = pd.concat([sales, sales.assign(date=date(2015, 12, 28))], ignore_index=True)
    with pytest.raises(ValueError, match='holdout'):
        run_fixture('last_week', sales=leak)


def test_weeks_interface_and_test_run_log(tmp_path):
    cfg = make_cfg()
    start = date(2015, 11, 1)
    sales = make_sales(start - timedelta(days=60), start + timedelta(days=6))
    log = tmp_path / 'test_runs.log'
    daily, kpis = run(make_policy('last_week'), sales, start, 1, cfg, state=make_state(start),
                      forecasts=make_forecasts(start, start + timedelta(days=6)), method='last_week',
                      output_dir=tmp_path / 'last_week', run_log=log)
    assert len(daily) == 7 and kpis['days'] == 7
    line = json.loads(log.read_text().splitlines()[-1])
    assert line['start'] == '2015-11-01' and line['end'] == '2015-11-07'
    assert len(line['config_sha256']) == 64 and line['method'] == 'last_week' and line['git_commit']
    assert (tmp_path / 'last_week' / 'execution_audit.json').exists()      # her independent audit passed
    for name in ['daily_metrics.csv', 'inventory_ledger.csv', 'order_plan.csv', 'dc_orders.csv', 'deliveries.csv']:
        assert (tmp_path / 'last_week' / name).exists()


# ---------------------------------------------------------------- scorecard

def test_kpis_hand_computed_christmas_out_of_service_kept_in_cost():
    """One store S1, one product A (buy 1.0, 10/carton), warehouse W; 3 days 24-26 Dec.
    day   closed demand served lost waste closing(store,W) cash  trucks
    24 Dec  no    10     8     2    1     (5, 20)         7.0   1
    25 Dec  yes    0     0     0    0     (5, 20)         3.0   0
    26 Dec  no    10    10     0    0     (0, 10)         20.0  1
    fill_rate = (8+10)/(10+10) = 0.9; lost 2; stockout_days 1 (25 Dec excluded); waste 1;
    stock value per day 25, 25, 10 -> avg 20 (stores 10/3=3.333, W 50/3=16.667);
    days_of_stock = mean closing units 20 / mean open-day demand 10 = 2.0;
    turnover = (18 served * 1.0 / 3 days * 365) / 20 = 109.5; total_cost = 30;
    start value 30 (store 5 + W 25), end value 10 + 10 on the way = 20 -> net_cost = 30 - (20-30) = 40;
    orders: 2 store lines with cartons > 0, 1 supplier line; trucks 2.
    Service: S1,A open days 2, 1 with lost sales -> 0.5 < 0.9 target -> 0 of 1 store-items meet it.
    """
    cfg = make_cfg()
    cfg['order_model']['parameters']['items'] = {'A': cfg['order_model']['parameters']['items']['A']}
    days = ['2015-12-24', '2015-12-25', '2015-12-26']
    daily = pd.DataFrame(dict(date=days, closed=[False, True, False], actual_demand=[10., 0., 10.],
                              served_units=[8., 0., 10.], lost_units=[2., 0., 0.], stockout_store_items=[1, 0, 0],
                              waste_proxy_units=[1., 0., 0.], cash_cost=[7., 3., 20.], purchase_cash=[0., 0., 10.],
                              truck_cash=[1., 0., 1.], order_cash=[.1, 0., .1], holding_cash=[5.9, 3., 8.9],
                              disposal_cash=[0., 0., 0.], lost_sales_penalty=[6., 0., 0.], supplier_units=[0., 0., 10.]))
    ledger = pd.DataFrame([
        dict(date=d, location_type=t, location_id=l, item_id='A', closing_units=c, served_units=sv,
             lost_units=lo, actual_demand=de)
        for d, rows in zip(days, [[('store', 'S1', 5, 8, 2, 10), ('dc', 'W', 20, 0, 0, 0)],
                                  [('store', 'S1', 5, 0, 0, 0), ('dc', 'W', 20, 0, 0, 0)],
                                  [('store', 'S1', 0, 10, 0, 10), ('dc', 'W', 10, 0, 0, 0)]])
        for t, l, c, sv, lo, de in rows])
    store_orders = pd.DataFrame(dict(date=days, store_id='S1', item_id='A', cases=[1., 0., 1.], units=[10., 0., 10.]))
    dc_orders = pd.DataFrame(dict(date=days, dc_id='W', item_id='A', pallets=[0., 0., 1.], units=[0., 0., 20.]))
    deliveries = pd.DataFrame(dict(date=days, store_id='S1', dc_id='W', cases_total=[1., 0., 1.], trucks=[1., 0., 1.]))
    start = State(date(2015, 12, 24), pd.DataFrame([dict(store_id='S1', item_id='A', units=5.)]),
                  pd.DataFrame([dict(dc_id='W', item_id='A', units=25.)]), pd.DataFrame(columns=PIPELINE_COLUMNS))
    end = State(date(2015, 12, 27), pd.DataFrame([dict(store_id='S1', item_id='A', units=0.)]),
                pd.DataFrame([dict(dc_id='W', item_id='A', units=10.)]),
                pd.DataFrame([dict(order_date=date(2015, 12, 26), arrival_date=date(2015, 12, 31), location_type='dc',
                                   location_id='W', item_id='A', units=10.)]))
    k = compute_kpis(daily, ledger, store_orders, dc_orders, deliveries, start, end, cfg)
    assert set(ROADMAP_KEYS) <= set(k)
    assert k['fill_rate'] == pytest.approx(0.9)
    assert k['lost_units'] == 2 and k['stockout_days'] == 1 and k['waste_units'] == 1
    assert k['avg_stock_value'] == pytest.approx(20.0)
    assert k['avg_store_stock_value'] == pytest.approx(10 / 3)
    assert k['avg_warehouse_stock_value'] == pytest.approx(50 / 3)
    assert k['days_of_stock'] == pytest.approx(2.0)
    assert k['turnover'] == pytest.approx(109.5)
    assert k['total_cost'] == pytest.approx(30.0)
    assert k['net_cost'] == pytest.approx(40.0)
    assert k['orders'] == 2 and k['supplier_order_lines'] == 1 and k['trucks'] == 2
    assert k['lost_sales_penalty'] == pytest.approx(6.0)
    assert k['denominators']['open_days'] == 2 and k['denominators']['days'] == 3
    assert k['denominators']['demand_units_open_days'] == 20
    assert k['service_target']['store_items'] == 1 and k['service_target']['share_meeting_target'] == 0.0
    assert k['service_target']['mean_no_loss_day_share'] == pytest.approx(0.5)


def test_rules_check_catches_broken_ledgers():
    cfg = make_cfg()
    good = dict(ledger=pd.DataFrame([dict(location_type='store', location_id='S1', item_id='A', morning_units=5,
                                          closing_units=2, dispatch_units=0),
                                     dict(location_type='dc', location_id='W', item_id='A', morning_units=10,
                                          closing_units=0, dispatch_units=10)]),
                store_orders=pd.DataFrame([dict(store_id='S1', item_id='A', cases=1., units=10.)]),
                dc_orders=pd.DataFrame([dict(dc_id='W', item_id='A', pallets=1., units=20.)]),
                deliveries=pd.DataFrame([dict(store_id='S1', dc_id='W', cases_total=1., trucks=1.)]),
                residual=0.0)
    assert check_rules(cfg=cfg, **good)['passed']
    bad = deepcopy(good); bad['ledger'].loc[0, 'closing_units'] = -1
    with pytest.raises(ValueError, match='negative'):
        check_rules(cfg=cfg, **bad)
    bad = deepcopy(good); bad['ledger'].loc[1, 'dispatch_units'] = 11
    with pytest.raises(ValueError, match='warehouse'):
        check_rules(cfg=cfg, **bad)
    bad = deepcopy(good); bad['deliveries'].loc[0, 'trucks'] = 0
    with pytest.raises(ValueError, match='truck'):
        check_rules(cfg=cfg, **bad)
    bad = deepcopy(good); bad['dc_orders'].loc[0, 'units'] = 2e6; bad['dc_orders'].loc[0, 'pallets'] = 1e5
    with pytest.raises(ValueError, match='budget'):
        check_rules(cfg=cfg, **bad)
    bad = deepcopy(good); bad['residual'] = 0.5
    with pytest.raises(ValueError, match='conservation'):
        check_rules(cfg=cfg, **bad)


@pytest.mark.parametrize('method', BASELINES + ['optimizer'])
def test_rules_never_broken_on_fixture_runs(method):
    _, k = run_fixture(method, days=3)
    assert k['rules']['passed']
    assert k['rules']['max_conservation_residual'] <= 1e-6
    assert set(ROADMAP_KEYS) <= set(k)


def test_registry_has_all_four_methods():
    # store_model / store_ilp are added to POLICIES on first use, so other tests may have added them
    assert set(POLICIES) - {'store_model', 'store_ilp'} == {'last_week', 'topup_avg', 'topup_forecast', 'optimizer'}
    assert type(make_policy('store_ilp')).__name__ == 'StoreILP'
    assert isinstance(make_policy('optimizer'), OptimizerPolicy)
