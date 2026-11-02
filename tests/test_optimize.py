from copy import deepcopy
from datetime import date, timedelta

import pandas as pd
import pytest

from examples.fixture import load_tiny
from optimize import plan_orders


def test_hand_checked_three_product_flow():
    state, fc, cfg = load_tiny()
    plan = plan_orders(state, fc, cfg)
    assert plan.status == 'Optimal'
    firm = plan.orders[plan.orders.date == state.date]
    assert firm.set_index('item_id').loc['STAPLE', 'cases'] == pytest.approx(2)
    assert firm.set_index('item_id').loc['STAPLE', 'units'] == pytest.approx(24)
    assert plan.objective == pytest.approx(0.32)
    stocks = pd.DataFrame(plan.stats['store_inventory'])
    assert list(stocks[stocks.item_id == 'STAPLE'].units) == pytest.approx([0, 6, 6])
    assert plan.dc_orders.units.sum() == pytest.approx(0)
    assert plan.stats['incumbent_validated']


def test_firm_integrality_minimum_orders_and_trucks():
    state, fc, cfg = load_tiny()
    plan = plan_orders(state, fc, cfg)
    tol = cfg['order_model']['solver']['feasibility_tolerance']
    for frame, column in [(plan.orders, 'cases'), (plan.dc_orders, 'pallets'),
                           (plan.deliveries, 'trucks')]:
        values = frame.loc[frame.date == state.date, column]
        assert all(abs(x-round(x)) <= tol for x in values)
    firm = plan.orders[plan.orders.date == state.date]
    assert all((x <= tol) or x >= cfg['order_model']['store_min_order_cases']-tol
               for x in firm.cases)
    assert all(plan.deliveries.cases_total <=
               cfg['order_model']['truck_capacity_cases'] * plan.deliveries.trucks + tol)
    assert min(x['units'] for x in plan.stats['store_inventory']) >= -tol
    assert min(x['units'] for x in plan.stats['dc_inventory']) >= -tol


def test_empty_supply_reports_terminal_deficits_without_inventing_stock():
    state, fc, cfg = load_tiny()
    state.dc_stock['units'] = 0
    cfg['order_model']['terminal_store_days'] = 1
    cfg['order_model']['terminal_dc_days'] = 5
    plan = plan_orders(state, fc, cfg)
    assert plan.status == 'Optimal'
    assert plan.orders.units.sum() == pytest.approx(0)
    assert plan.dc_orders.units.sum() == pytest.approx(0)
    assert plan.stats['terminal_store_shortfall']['SHOP|STAPLE'] == pytest.approx(6)
    assert plan.stats['terminal_dc_shortfall']['DC|STAPLE'] == pytest.approx(30)
    assert plan.stats['lost_sales_total'] == pytest.approx(18)


def test_overflow_stays_in_inventory_and_is_separately_priced():
    state, fc, cfg = load_tiny()
    fc.loc[:, ['q50', 'q80', 'q90', 'q95']] = 0
    state.dc_stock['units'] = 0
    state.store_stock.loc[state.store_stock.item_id == 'STAPLE', 'units'] = 30
    cfg['order_model']['parameters']['store_averages']['SHOP']['STAPLE'] = 0
    # Base capacity = 2 cartons = 24; six units stay in the backroom each morning.
    plan = plan_orders(state, fc, cfg)
    assert plan.status == 'Optimal'
    assert plan.stats['overflow_unit_mornings'] == pytest.approx(18)
    assert plan.stats['waste_units_total'] == pytest.approx(0)
    stocks = [r['units'] for r in plan.stats['store_inventory'] if r['item_id'] == 'STAPLE']
    assert stocks == pytest.approx([30, 30, 30])
    assert plan.stats['objective_breakdown']['overflow'] == pytest.approx(36)


def test_known_christmas_closure_prohibits_arrivals_without_removing_date():
    state, fc, cfg = load_tiny()
    state.date = date(2015, 12, 24)
    fc['origin'] = state.date
    fc['date'] = [state.date + timedelta(days=i % cfg['order_model']['horizon_days'])
                  for i in range(len(fc))]
    plan = plan_orders(state, fc, cfg)
    assert plan.status == 'Optimal'
    # Today's dispatch would arrive on Christmas, so it is zero.
    assert plan.orders.loc[plan.orders.date == state.date, 'units'].sum() == pytest.approx(0)
    assert len(plan.stats['store_inventory']) == len(fc)
    assert plan.stats['effective_closed_forecast_rows'] == 3


def test_origin_filter_does_not_sum_other_forecast_vintages():
    state, fc, cfg = load_tiny()
    extra = fc.copy()
    extra['origin'] = state.date - timedelta(days=1)
    extra.loc[:, ['q50', 'q80', 'q90', 'q95']] = 1000
    plan = plan_orders(state, pd.concat([fc, extra], ignore_index=True), cfg)
    assert plan.objective == pytest.approx(0.32)


def test_missing_forecast_is_rejected_before_solve():
    state, fc, cfg = load_tiny()
    with pytest.raises(ValueError, match='complete'):
        plan_orders(state, fc.iloc[:-1], cfg)


def test_initial_stock_is_not_silently_clipped_on_every_planning_day():
    state, fc, cfg = load_tiny()
    state.store_stock.loc[state.store_stock.item_id == 'ZERO', 'units'] = 20
    plan = plan_orders(state, fc, cfg)
    rows = [r for r in plan.stats['store_inventory'] if r['item_id'] == 'ZERO']
    assert [r['units'] for r in rows] == pytest.approx([20, 20, 20])


def test_unapproved_layers_are_rejected():
    state, fc, cfg = load_tiny()
    cfg = deepcopy(cfg)
    cfg['order_model']['layers']['age_tracking'] = True
    with pytest.raises(NotImplementedError):
        plan_orders(state, fc, cfg)


def test_day1_pipeline_arrival_is_counted_exactly_once():
    state, fc, cfg = load_tiny()
    fc.loc[:, ['q50', 'q80', 'q90', 'q95']] = 0
    fc.loc[(fc.item_id == 'STAPLE') & (fc.date == state.date),
           ['q50', 'q80', 'q90', 'q95']] = 12
    state.dc_stock['units'] = 0
    state.pipeline = pd.DataFrame([dict(order_date=state.date-timedelta(days=1),
                                       arrival_date=state.date, location_type='store',
                                       location_id='SHOP', item_id='STAPLE', units=12)])
    plan = plan_orders(state, fc, cfg)
    assert plan.stats['lost_sales_total'] == pytest.approx(0)
    assert sum(r['units'] for r in plan.stats['store_inventory']) == pytest.approx(0)


def test_retained_waste_proxy_removes_six_excess_units():
    state, fc, cfg = load_tiny()
    fc.loc[:, ['q50', 'q80', 'q90', 'q95']] = 0
    fc.loc[fc.item_id == 'FRESH', ['q50', 'q80', 'q90', 'q95']] = 12
    state.store_stock.loc[state.store_stock.item_id == 'FRESH', 'units'] = 30
    state.dc_stock['units'] = 0
    plan = plan_orders(state, fc, cfg)
    assert plan.stats['waste_units_total'] == pytest.approx(6)
    fresh = [r['units'] for r in plan.stats['store_inventory'] if r['item_id'] == 'FRESH']
    # Only day 1 has work-conserving sales fixed exactly. Future quantities are
    # explicitly relaxed planning allocations; do not assert a physical replay.
    assert fresh[0] == pytest.approx(12)
    assert plan.stats['lost_sales_total'] == pytest.approx(12)
    # This particular shortage case demonstrates the declared future allocation
    # relaxation. The diagnostic must expose it, not present it as a replay.
    assert plan.stats['future_sales_rationing_units'] == pytest.approx(12)


def test_committed_christmas_arrival_requires_explicit_rescheduling():
    state, fc, cfg = load_tiny()
    state.pipeline = pd.DataFrame([dict(order_date=state.date-timedelta(days=1),
                                       arrival_date=date(2015, 12, 25), location_type='store',
                                       location_id='SHOP', item_id='STAPLE', units=12)])
    with pytest.raises(ValueError, match='rescheduling'):
        plan_orders(state, fc, cfg)


def test_supplier_purchase_arrives_at_dc_before_onward_store_delivery():
    state, fc, cfg = load_tiny()
    state.dc_stock['units'] = 0
    fc.loc[:, ['q50', 'q80', 'q90', 'q95']] = 0
    last = state.date+timedelta(days=cfg['order_model']['horizon_days']-1)
    fc.loc[(fc.item_id == 'STAPLE') & (fc.date == last), ['q50', 'q80', 'q90', 'q95']] = 18
    cfg['order_model']['lead_time_supplier_days'] = 1
    cfg['order_model']['parameters']['dc_budgets']['DC'] = 10000
    plan = plan_orders(state, fc, cfg)
    first = plan.dc_orders[(plan.dc_orders.date == state.date) & (plan.dc_orders.item_id == 'STAPLE')]
    assert first.pallets.iloc[0] == pytest.approx(2)
    assert first.units.iloc[0] == pytest.approx(24)
    assert plan.orders.loc[plan.orders.date == state.date, 'units'].sum() == pytest.approx(0)
    second = plan.orders[(plan.orders.date == state.date+timedelta(days=1)) & (plan.orders.item_id == 'STAPLE')]
    # All 24 purchased units may be forwarded: this baseline charges holding at
    # stores/DCs but not in transit. What matters here is the arrival sequence and
    # that onward shipment covers demand without exceeding purchased supply.
    assert 18 <= second.units.iloc[0] <= 24
    assert plan.stats['lost_sales_total'] == pytest.approx(0)
