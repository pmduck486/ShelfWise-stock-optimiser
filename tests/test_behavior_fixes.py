from copy import deepcopy
from datetime import timedelta

import pandas as pd
import pytest

from examples.fixture import load_tiny
from optimize import plan_orders
from optimize.behavior import service_priority_weights, work_conserving_path


def priority_config(cfg):
    cfg = deepcopy(cfg)
    cfg['order_model'].update(fulfillment_policy='earlier_service_priority',
                             early_service_margin_share=0.01,
                             postsolve_objective_tolerance=0.00001)
    return cfg


def test_priority_weights_dominate_deferring_a_sale():
    weights = service_priority_weights(3, 1.5, 0.5, 0.5, 0.01)
    assert weights == pytest.approx([2.52, 2.01, 1.5])
    assert weights[-1] > 0.5
    assert all(weights[t]-weights[t+1] > 0.5 for t in range(len(weights)-1))


def test_unsupported_priority_costs_are_rejected():
    with pytest.raises(ValueError, match='terminal'):
        service_priority_weights(3, 0.4, 0.5, 0.5, 0.01)


def test_physical_path_serves_available_demand():
    path = work_conserving_path(30, [0,0,0], [12,12,12], [6,0,0])
    assert [r['closing'] for r in path] == pytest.approx([12,0,0])
    assert [r['lost_sales'] for r in path] == pytest.approx([0,0,12])
    assert [r['waste'] for r in path] == pytest.approx([6,0,0])


def test_future_withholding_is_fixed_without_extra_integers():
    state, fc, original = load_tiny()
    fc.loc[:, ['q50','q80','q90','q95']] = 0
    fc.loc[fc.item_id == 'FRESH', ['q50','q80','q90','q95']] = 12
    state.store_stock.loc[state.store_stock.item_id == 'FRESH', 'units'] = 30
    state.dc_stock['units'] = 0
    old = plan_orders(state,fc,original)
    fixed = plan_orders(state,fc,priority_config(original))
    assert old.stats['future_sales_rationing_units'] == pytest.approx(12)
    assert fixed.stats['future_sales_rationing_units'] == pytest.approx(0)
    assert old.stats['n_int'] == fixed.stats['n_int'] == 10
    assert fixed.stats['waste_units_total'] == pytest.approx(6)
    rows = pd.DataFrame(fixed.stats['store_inventory'])
    assert rows[rows.item_id == 'FRESH'].units.tolist() == pytest.approx([12,0,0])
    assert fixed.stats['physical_projection']['objective_change'] <= 0.00001


def warehouse_case():
    state,fc,cfg = load_tiny()
    om = cfg['order_model']
    om.update(horizon_days=7,lead_time_supplier_days=5,
              service_shortfall_penalty_share=0.5,lost_sale_penalty_share=1.5,
              terminal_store_days=1,terminal_dc_days=5,overflow_penalty_share=0.5)
    p = deepcopy(om['parameters']['items']['STAPLE'])
    p.update(units_per_case=1,cases_per_pallet=1,buy_cost=0.85,
             holding_cost_per_unit_day=0.85*0.25/365)
    om['parameters'] = dict(items={'STAPLE':p},store_averages={'SHOP':{'STAPLE':2}},
                            dc_budgets={'DC':1.25*7*2*0.85})
    state.store_stock = pd.DataFrame([dict(store_id='SHOP',item_id='STAPLE',units=14)])
    state.dc_stock = pd.DataFrame([dict(dc_id='DC',item_id='STAPLE',units=0)])
    fc = pd.DataFrame([dict(origin=state.date,date=state.date+timedelta(days=t),
                            store_id='SHOP',item_id='STAPLE',q50=2,q80=2,q90=2,q95=2)
                       for t in range(7)])
    return state,fc,cfg


def test_daily_dc_cover_counts_real_current_orders_not_future_promises():
    state,fc,cfg = warehouse_case()
    old = plan_orders(state,fc,cfg)
    changed = priority_config(cfg)
    changed['order_model'].update(warehouse_reserve_policy='daily_inventory_position',
                                 dc_position_penalty_share=0.5,
                                 terminal_dc_penalty_cost_multiplier=1.10)
    fixed = plan_orders(state,fc,changed)
    assert old.dc_orders.pallets.sum() == pytest.approx(0)
    firm = fixed.dc_orders[fixed.dc_orders.date == state.date]
    assert firm.units.sum() > 0
    first = fixed.stats['dc_protection'][0]
    assert first['inventory_position'] == pytest.approx(firm.units.sum())
    assert first['target_units'] == pytest.approx(10)
    assert first['inventory_position']+first['shortfall_units'] >= 10-0.00001
    assert fixed.stats['n_int'] == old.stats['n_int']
