from dataclasses import replace
from pathlib import Path

import pytest

from optimize import plan_orders
from optimize.audit import audit_plan
from optimize.parameters import load_small_case


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def solved_case():
    case = load_small_case(ROOT/'data'/'checkpoint2', ROOT/'config.yaml')
    plan = plan_orders(case.state, case.forecasts, case.parameters.config)
    return case, plan


def test_real_small_plan_has_a_validated_integer_incumbent(solved_case):
    case, plan = solved_case
    assert plan.stats['incumbent_validated']
    assert plan.status in {'Optimal', 'GapSatisfied-Feasible', 'TimeLimit-Feasible', 'Feasible'}
    assert plan.orders.shape[0] == 140
    assert plan.dc_orders.shape[0] == 140
    assert plan.deliveries.shape[0] == 14
    assert plan.stats['n_int'] == 62
    assert list(plan.orders) == ['date', 'store_id', 'item_id', 'cases', 'units']
    assert list(plan.dc_orders) == ['date', 'dc_id', 'item_id', 'pallets', 'units']
    assert list(plan.deliveries) == ['date', 'store_id', 'dc_id', 'cases_total', 'trucks']


def test_independent_export_audit_checks_all_physical_and_soft_rules(solved_case):
    case, plan = solved_case
    audit = audit_plan(case.state, case.forecasts, case.parameters.config, plan)
    assert audit['passed']
    assert audit['store_balance_checks'] == 140
    assert audit['dc_balance_checks'] == 140
    assert audit['truck_checks'] == 14
    assert audit['terminal_target_checks'] == 40
    assert audit['max_balance_residual_units'] <= case.parameters.config['order_model']['solver']['feasibility_tolerance']


def test_export_audit_rejects_mismatched_carton_units(solved_case):
    case, plan = solved_case
    bad = plan.orders.copy()
    bad.loc[bad.index[0], 'units'] += 1
    corrupted = replace(plan, orders=bad)
    with pytest.raises(ValueError, match='carton units'):
        audit_plan(case.state, case.forecasts, case.parameters.config, corrupted)


def test_common_initialization_fits_correct_store_specific_shelves(solved_case):
    case, _ = solved_case
    stocks = case.state.store_stock.merge(case.parameters.store_parameters,
                                         on=['store_id', 'item_id'], validate='one_to_one')
    assert all(stocks.units <= stocks.shelf_capacity_units)
    assert all(case.initialization_log.clipped_units >= 0)
    assert set(case.initialization_log.accounting_category) == {'initial_state_adjustment'}
