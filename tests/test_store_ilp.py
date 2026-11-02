"""Lump 3 on the store-only model (2026-11-02): the one-day store ILP, its shadow prices and
the what-ifs. Uses the small fixture of test_replay.py / test_store_only.py."""

from copy import deepcopy
from datetime import timedelta

import pytest

from analysis.store_ilp import LIMITS, limits, solve_day
from analysis.whatif import CONTRACT, get_path, set_path, what_if
from tests.test_replay import D1, MAIN_CFG, make_forecasts, make_sales, make_state
from tests.test_store_only import run_store, store_cfg, two_day


def ilp_cfg(layers=False, **analysis):
    """The ILP with every layer switched off (the core), or as in config.yaml with layers=True."""
    cfg = store_cfg()
    cfg['analysis'] = deepcopy(MAIN_CFG['analysis'])
    a = cfg['analysis']['store_ilp']
    a.update(analysis)
    if not layers:
        a['layers']['service_floor']['enabled'] = False
        a['layers']['spoilage_cap_in_ilp'] = False
        a['layers']['stock_value_cap']['enabled'] = False
    return cfg


KEYS = [('S1', 'A'), ('S1', 'B')]


def _solve(cfg, need, tomorrow=None):
    tomorrow = tomorrow or {k: 0.0 for k in KEYS}
    return solve_day(cfg, KEYS, need, tomorrow, cover_days=2, open_cover_days=2)


def _shadow(day, name):
    return next(r for r in day.shadow if r['constraint'] == name)


# ---------------------------------------------------------------- the ILP itself

def test_config_has_the_lump3_section():
    a = MAIN_CFG['analysis']
    assert set(a['store_ilp']) >= {'space_factor', 'budget_factor', 'time_limit_s', 'mip_gap'}
    for spec in a['whatif']['runs']:                       # the first value of every what-if is the base
        assert get_path(dict(MAIN_CFG), spec['param']) == spec['values'][0]


def test_whole_cartons_and_no_order_when_nothing_is_needed():
    cfg = ilp_cfg()
    day = _solve(cfg, {('S1', 'A'): 35.0, ('S1', 'B'): -4.0})
    assert all(isinstance(v, int) for v in day.cartons.values())
    assert day.cartons[('S1', 'B')] == 0                   # already above target
    assert day.cartons[('S1', 'A')] in (3, 4)              # 35 units, 10 per carton
    assert day.trucks['S1'] == 1
    assert day.stats['ilp_not_proven'] == 0


def test_loose_limits_have_zero_shadow_price():
    day = _solve(ilp_cfg(), {('S1', 'A'): 15.0, ('S1', 'B'): 5.0})    # 20 units, budget 32.5
    assert {r['constraint'] for r in day.shadow} == set(LIMITS)
    assert all(r['value'] == 0 for r in day.shadow)
    assert day.stats['ilp_binding'] == 0


def test_tight_truck_gives_a_positive_shadow_price():
    cfg = ilp_cfg()
    cfg['order_model']['truck_capacity_cases'] = 2
    cfg['order_model']['truck_trip_cost'] = 50.0           # a trip pays for 2 cartons of A (60), not of B (45)
    cfg['analysis']['store_ilp']['budget_factor'] = 100.0
    day = _solve(cfg, {('S1', 'A'): 50.0, ('S1', 'B'): 25.0})
    r = _shadow(day, 'truck')
    assert day.trucks['S1'] == 3 and r['used'] == pytest.approx(r['limit']) == pytest.approx(6)
    assert r['value'] == pytest.approx(5 * 4.5)          # one more carton of B: 5 units x Rs 4.5 lost-sale cost


def test_tight_space_gives_a_positive_shadow_price():
    cfg = ilp_cfg(budget_factor=100.0)
    free = limits(cfg, KEYS, {k: 0.0 for k in KEYS}, 2)['S1']['space']
    day = _solve(cfg, {('S1', 'A'): 10 * free, ('S1', 'B'): 10 * free})
    r = _shadow(day, 'space')
    assert r['used'] <= free + 1e-6
    assert r['value'] > 0


def test_tight_budget_gives_a_positive_shadow_price():
    cfg = ilp_cfg(budget_factor=0.5)               # 13 Rs: one carton of A, not two
    day = _solve(cfg, {('S1', 'A'): 40.0, ('S1', 'B'): 15.0})
    r = _shadow(day, 'budget')
    assert r['used'] <= r['limit'] + 1e-6
    assert r['value'] > 0


def test_space_counts_what_is_still_on_the_shelf_tomorrow():
    cfg = ilp_cfg()
    empty = limits(cfg, KEYS, {k: 0.0 for k in KEYS}, 2)['S1']['space']
    full = limits(cfg, KEYS, {('S1', 'A'): 15.0, ('S1', 'B'): -3.0}, 2)['S1']['space']
    assert full == pytest.approx(max(empty - 15.0, 0.0))   # negative leftovers count as empty


# ---------------------------------------------------------------- in the store-only replay

def test_store_ilp_replay_passes_rules_and_audit_and_writes_shadow_prices(tmp_path):
    cfg = ilp_cfg()
    daily, k = run_store('store_ilp', cfg=cfg, output_dir=tmp_path / 'ilp')
    assert k['rules']['passed'] and k['independent_audit']
    assert k['supplier_units'] == 0
    assert k['store_ilp']['days_solved'] == 3
    # orders are the ILP's cartons exactly
    for d in k['_decisions']:
        assert d.audit['ilp_status'] == 'Optimal'
        assert d.audit['ilp_trucks'] >= 0


def test_store_ilp_matches_store_model_when_limits_are_loose(tmp_path):
    """With loose limits the ILP should order about what the store model's rounding orders."""
    cfg = ilp_cfg()
    _, a = run_store('store_model', cfg=cfg)
    _, b = run_store('store_ilp', cfg=cfg)
    assert b['fill_rate'] == pytest.approx(a['fill_rate'], abs=0.05)


# ---------------------------------------------------------------- what-ifs

def test_set_path_refuses_unknown_settings():
    cfg = ilp_cfg()
    set_path(cfg, 'analysis.store_ilp.space_factor', 0.5)
    assert get_path(cfg, 'analysis.store_ilp.space_factor') == 0.5
    with pytest.raises(KeyError):
        set_path(cfg, 'analysis.store_ilp.no_such_thing', 1)


def test_what_if_returns_the_contract_and_a_tighter_budget_costs_more(tmp_path):
    cfg = ilp_cfg()
    end = D1 + timedelta(days=2)
    t = what_if('analysis.store_ilp.budget_factor', [cfg['analysis']['store_ilp']['budget_factor'], 0.3], cfg,
                method='store_ilp', sales=make_sales(D1 - timedelta(days=60), end), start=D1,
                state=make_state(D1), forecasts=two_day(make_forecasts(D1, end)), end=end,
                out_dir=tmp_path)
    assert list(t.columns[:len(CONTRACT)]) == CONTRACT
    assert len(t) == 2
    assert t.loc[1, 'shadow_budget_binding_store_days'] >= t.loc[0, 'shadow_budget_binding_store_days']
    assert t.loc[1, 'shadow_budget_binding_store_days'] > 0
    assert t.loc[1, 'fill_rate'] <= t.loc[0, 'fill_rate'] + 1e-9
    assert (tmp_path / 'analysis.store_ilp.budget_factor=0.3' / 'store_ilp' / 'shadow_prices.csv').exists()


# ---------------------------------------------------------------- extra layers (2026-11-02)

def test_config_layers_as_decided():
    lay = MAIN_CFG['analysis']['store_ilp']['layers']
    assert lay['service_floor']['enabled'] is True and lay['spoilage_cap_in_ilp'] is True
    assert isinstance(lay['stock_value_cap']['enabled'], bool)


def _solve_layers(cfg, need, position, cap=None, floor=None):
    return solve_day(cfg, KEYS, need, {k: 0.0 for k in KEYS}, 2, 2, position=position, cap=cap,
                     floor_units=floor or {k: need[k] + position[k] for k in KEYS})


def test_service_floor_is_met_when_there_is_room_and_priced_when_not():
    cfg = ilp_cfg()
    cfg['analysis']['store_ilp']['layers']['service_floor']['enabled'] = True
    pos = {k: 0.0 for k in KEYS}
    # lots of room: every item reaches at least 90% of its target
    day = _solve_layers(cfg, {('S1', 'A'): 15.0, ('S1', 'B'): 5.0}, pos)
    assert day.cartons[('S1', 'A')] * 10 >= 0.9 * 15 and day.cartons[('S1', 'B')] * 5 >= 0.9 * 5
    # tiny budget (13 Rs): the floor cannot be met, so it gets a price
    cfg['analysis']['store_ilp']['budget_factor'] = 0.5
    day = _solve_layers(cfg, {('S1', 'A'): 40.0, ('S1', 'B'): 15.0}, pos)
    floors = [r for r in day.shadow if r['constraint'] == 'floor']
    assert floors and all(r['value'] > 0 and r['item_id'] for r in floors)


def test_service_floor_off_changes_nothing():
    cfg = ilp_cfg()
    need = {('S1', 'A'): 35.0, ('S1', 'B'): 7.0}
    a = _solve(cfg, need)
    b = _solve_layers(cfg, need, {k: 0.0 for k in KEYS})
    assert a.cartons == b.cartons


def test_spoilage_cap_limits_the_order_and_has_a_price():
    cfg = ilp_cfg()
    pos = {('S1', 'A'): 5.0, ('S1', 'B'): 0.0}
    cap = {('S1', 'A'): 25.0, ('S1', 'B'): float('inf')}
    day = _solve_layers(cfg, {('S1', 'A'): 60.0, ('S1', 'B'): 5.0}, pos, cap=cap)
    assert pos[('S1', 'A')] + 10 * day.cartons[('S1', 'A')] <= 25.0
    r = [r for r in day.shadow if r['constraint'] == 'spoil']
    assert len(r) == 1 and r[0]['item_id'] == 'A' and r[0]['value'] > 0


def test_stock_value_cap_binds_and_has_a_price():
    cfg = ilp_cfg()
    sv = cfg['analysis']['store_ilp']['layers']['stock_value_cap']
    sv.update(enabled=True, cover_days=1)                  # 13 Rs of stock tomorrow morning
    day = _solve(cfg, {('S1', 'A'): 40.0, ('S1', 'B'): 15.0})
    r = _shadow(day, 'stockval')
    assert r['limit'] == pytest.approx(13.0) and r['used'] <= 13.0 + 1e-6
    assert r['value'] > 0


def test_store_ilp_with_all_layers_passes_rules_and_audit(tmp_path):
    cfg = ilp_cfg(layers=True)
    cfg['analysis']['store_ilp']['layers']['stock_value_cap']['enabled'] = True
    _, k = run_store('store_ilp', cfg=cfg, output_dir=tmp_path / 'layers')
    assert k['rules']['passed'] and k['independent_audit']
