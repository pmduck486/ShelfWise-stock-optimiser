"""Core-spec tests that were listed in docs/order_model_spec.md section 10 but had
no dedicated test, plus the L1/L2 service-line contract check.

Added by the alignment check (31 Oct 2026). These tests only read the model;
they change no code path, input file or config value of the order model.
"""

from copy import deepcopy
import json
from pathlib import Path

import pulp
import pytest
import yaml

from examples.fixture import load_tiny
from optimize import plan_orders

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = ['q50', 'q80', 'q90', 'q95']


def _fresh_case(disposal_share):
    """FRESH (shelf life 2, 6 per carton) sells 8 on day 2 only.

    One carton loses 2 sales; two cartons leave 4 units for expected disposal.
    """
    state, fc, cfg = load_tiny()
    fc.loc[fc.item_id == 'FRESH', QUANTILES] = [[0] * 4, [8] * 4, [0] * 4]
    state.dc_stock.loc[state.dc_stock.item_id == 'FRESH', 'units'] = 24
    cfg['order_model']['waste_disposal_share'] = disposal_share
    return state, fc, cfg


def _firm_fresh_cases(plan, state):
    o = plan.orders
    return float(o.loc[(o.item_id == 'FRESH') & (o.date == state.date), 'cases'].sum())


def test_waste_cost_works_raising_it_lowers_perishable_stock():
    # Spec row "Waste cost works": a constructed sensitivity case. Cheap waste
    # -> cover all demand (2 cartons); expensive waste -> accept 2 lost sales (1 carton).
    cases = []
    for share in [0.05, 1.0, 5.0, 20.0, 50.0]:
        state, fc, cfg = _fresh_case(share)
        plan = plan_orders(state, fc, cfg)
        assert plan.status == 'Optimal'
        cases.append(_firm_fresh_cases(plan, state))
    assert cases[0] == pytest.approx(2)
    assert cases[-1] == pytest.approx(1)
    # Weakly decreasing along the sweep (the spec allows ties between steps).
    assert all(a >= b for a, b in zip(cases, cases[1:]))


def test_zero_sales_product_gets_no_order_and_no_error():
    # Spec row "Zero-sales product": forecast 0 and store average 0 -> no order.
    state, fc, cfg = load_tiny()
    assert (fc.loc[fc.item_id == 'ZERO', QUANTILES] == 0).all().all()
    plan = plan_orders(state, fc, cfg)
    assert plan.status == 'Optimal'
    assert plan.orders.loc[plan.orders.item_id == 'ZERO', 'units'].sum() == pytest.approx(0)
    assert plan.dc_orders.loc[plan.dc_orders.item_id == 'ZERO', 'units'].sum() == pytest.approx(0)


def test_size_log_is_filled_and_printed(capsys):
    # Spec row "Size log": variable/integer/constraint counts, time and status.
    state, fc, cfg = load_tiny()
    plan = plan_orders(state, fc, cfg)
    for key in ['n_vars', 'n_int', 'n_constraints', 'solve_seconds', 'status']:
        assert plan.stats[key] is not None
    assert plan.stats['n_vars'] > plan.stats['n_int'] > 0
    assert plan.stats['n_constraints'] > 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()
               if line.startswith('{') and '"n_vars"' in line]
    assert printed and printed[-1]['n_vars'] == plan.stats['n_vars']


def test_integer_vs_lp_milk_example():
    # Spec row "Integer vs LP (viva demo)": 18 units needed, 12 per carton.
    # The LP relaxation orders 1.5 cartons; the integer program orders 2.
    state, fc, cfg = load_tiny()
    plan = plan_orders(state, fc, cfg)
    firm = plan.orders[plan.orders.date == state.date].set_index('item_id')
    assert firm.loc['STAPLE', 'cases'] == pytest.approx(2)
    relaxed = deepcopy(plan.model)
    store_order_vars = [v for v in relaxed.variables() if v.isInteger() and v.name.startswith('x_')]
    assert store_order_vars
    for v in relaxed.variables():
        if v.isInteger():
            v.cat = pulp.LpContinuous
    relaxed.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[relaxed.status] == 'Optimal'
    assert sum(v.varValue for v in store_order_vars) == pytest.approx(1.5)
    assert pulp.value(relaxed.objective) <= plan.objective + 1e-9


LEVEL_TO_LINE = {0.95: 'q95', 0.90: 'q90', 0.80: 'q80', 0.50: 'q50'}


def test_forecaster_service_lines_match_order_model_service_levels():
    # L1 scores each product on the line L2 stocks it to. When both sections are
    # in the shared config.yaml they must say the same thing. Skips until merged.
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    if 'forecast' not in cfg or 'service_lines' not in cfg.get('forecast', {}):
        pytest.skip('forecast section not merged into this config yet')
    lines = cfg['forecast']['service_lines']
    levels = cfg['order_model']['service_levels']
    assert lines['perishable'] == LEVEL_TO_LINE[levels['perishable']]
    assert lines['default'] == LEVEL_TO_LINE[levels['default']]
    assert lines['top_n_staples']['n'] == levels['top_n_staples']['n']
    assert lines['top_n_staples']['line'] == LEVEL_TO_LINE[levels['top_n_staples']['level']]
    assert (lines['slow_movers']['below_avg_daily_units']
            == levels['slow_movers']['below_avg_daily_units'])
    assert lines['slow_movers']['line'] == LEVEL_TO_LINE[levels['slow_movers']['level']]
