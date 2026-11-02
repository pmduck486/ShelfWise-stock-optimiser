import pulp
import pytest

from examples.fixture import load_tiny
from optimize import plan_orders


def no_incumbent(solver, model, **kwargs):
    for variable in model.variables():
        variable.varValue = 0.5
    model.status = pulp.LpStatusNotSolved
    model.sol_status = pulp.LpSolutionNoSolutionFound
    return model.status


def test_no_incumbent_returns_only_the_separately_validated_start(monkeypatch):
    state,fc,cfg = load_tiny()
    cfg['order_model']['solver'].update(warm_start='verified_plan',start_plan_fallback=True)
    monkeypatch.setattr(pulp.PULP_CBC_CMD,'actualSolve',no_incumbent)
    plan = plan_orders(state,fc,cfg)
    assert plan.status == 'Fallback-Feasible'
    assert plan.stats['incumbent_validated']
    assert plan.stats['used_start_fallback']
    assert plan.stats['verified_start_source'] == 'zero_orders'
    assert plan.orders.cases.sum() == pytest.approx(0)
    assert plan.stats['lost_sales_total'] == pytest.approx(18)


def test_previous_order_plan_is_a_valid_start_not_a_fixed_solver_constraint(monkeypatch):
    state,fc,cfg = load_tiny()
    baseline = plan_orders(state,fc,cfg)
    cfg['order_model']['solver'].update(warm_start='verified_plan',start_plan_fallback=True)
    cfg['order_model']['warm_start_plan'] = dict(orders=baseline.orders,dc_orders=baseline.dc_orders)
    monkeypatch.setattr(pulp.PULP_CBC_CMD,'actualSolve',no_incumbent)
    plan = plan_orders(state,fc,cfg)
    assert plan.status == 'Fallback-Feasible'
    assert plan.stats['verified_start_source'] == 'supplied_orders'
    assert plan.stats['incumbent_validated']
    first = plan.orders[(plan.orders.date == state.date)&(plan.orders.item_id == 'STAPLE')]
    assert first.cases.iloc[0] == pytest.approx(2)


def test_cbc_can_improve_the_zero_order_start():
    state,fc,cfg = load_tiny()
    cfg['order_model']['solver'].update(warm_start='verified_plan',start_plan_fallback=True)
    plan = plan_orders(state,fc,cfg)
    first=plan.orders[(plan.orders.date==state.date)&(plan.orders.item_id=='STAPLE')]
    assert plan.stats['verified_start_source']=='zero_orders'
    assert plan.status=='Optimal'
    assert first.cases.iloc[0]==pytest.approx(2)


def test_actual_tiny_time_limit_still_returns_a_valid_plan():
    state,fc,cfg = load_tiny()
    cfg['order_model']['solver'].update(warm_start='verified_plan',start_plan_fallback=True,
                                        time_limit_s=0.000001)
    plan=plan_orders(state,fc,cfg)
    assert plan.stats['incumbent_validated']
    assert plan.status in {'Optimal','TimeLimit-Feasible','Fallback-Feasible','GapSatisfied-Feasible','Feasible'}
    firm=plan.orders[plan.orders.date==state.date]
    assert all(abs(n-round(n))<0.000001 for n in firm.cases)


def test_replenishment_hint_keeps_useful_firm_orders_when_cbc_has_no_incumbent(monkeypatch):
    state,fc,cfg=load_tiny()
    cfg['order_model']['solver'].update(warm_start='verified_plan',start_plan_fallback=True,
                                        replenishment_start=True)
    monkeypatch.setattr(pulp.PULP_CBC_CMD,'actualSolve',no_incumbent)
    plan=plan_orders(state,fc,cfg)
    assert plan.stats['verified_start_source']=='forecast_replenishment'
    assert plan.stats['incumbent_validated'] and plan.stats['used_start_fallback']
    firm=plan.orders[(plan.orders.date==state.date)&(plan.orders.item_id=='STAPLE')]
    assert firm.cases.iloc[0]==2
    assert plan.stats['lost_sales_total']==pytest.approx(0)


def test_duplicate_commitments_are_aggregated_by_indexed_model_and_audit():
    from datetime import timedelta
    import pandas as pd
    from optimize.audit import audit_plan
    state,fc,cfg=load_tiny()
    state.dc_stock.loc[state.dc_stock.item_id=='STAPLE','units']=0
    row=dict(order_date=state.date-timedelta(days=5),arrival_date=state.date,
             location_type='dc',location_id='DC',item_id='STAPLE',units=12)
    state.pipeline=pd.DataFrame([row,row])
    plan=plan_orders(state,fc,cfg)
    assert plan.orders[(plan.orders.date==state.date)&(plan.orders.item_id=='STAPLE')].units.iloc[0]==24
    assert audit_plan(state,fc,cfg,plan)['passed']


def test_cold_dc_replenishment_hint_keeps_a_real_supplier_commitment(monkeypatch):
    state,fc,cfg=load_tiny();om=cfg['order_model']
    om['parameters']['items']={'STAPLE':om['parameters']['items']['STAPLE']}
    om['parameters']['items']['STAPLE'].update(units_per_case=1,cases_per_pallet=1,buy_cost=.85)
    om['parameters']['store_averages']={'SHOP':{'STAPLE':2}}
    om['parameters']['dc_budgets']={'DC':100}
    om.update(warehouse_reserve_policy='daily_inventory_position',dc_position_penalty_share=.5,
              terminal_dc_penalty_cost_multiplier=1.10,terminal_dc_days=5)
    om['solver'].update(warm_start='verified_plan',start_plan_fallback=True,replenishment_start=True)
    state.store_stock=state.store_stock[state.store_stock.item_id=='STAPLE'].copy();state.store_stock['units']=14
    state.dc_stock=state.dc_stock[state.dc_stock.item_id=='STAPLE'].copy();state.dc_stock['units']=0
    fc=fc[fc.item_id=='STAPLE'].copy();fc[['q50','q80','q90','q95']]=2
    monkeypatch.setattr(pulp.PULP_CBC_CMD,'actualSolve',no_incumbent)
    plan=plan_orders(state,fc,cfg)
    assert plan.stats['verified_start_source']=='forecast_replenishment'
    assert plan.stats['incumbent_validated']
    assert plan.dc_orders.loc[plan.dc_orders.date==state.date,'units'].sum()==10


def test_hint_respects_activation_when_available_supply_is_less_than_one_carton(monkeypatch):
    state,fc,cfg=load_tiny()
    state.dc_stock.loc[state.dc_stock.item_id=='STAPLE','units']=1
    cfg['order_model']['solver'].update(warm_start='verified_plan',start_plan_fallback=True,replenishment_start=True)
    monkeypatch.setattr(pulp.PULP_CBC_CMD,'actualSolve',no_incumbent)
    plan=plan_orders(state,fc,cfg)
    assert plan.stats['incumbent_validated']
    assert not plan.stats['replenishment_start_issues']
    assert plan.orders.loc[plan.orders.item_id=='STAPLE','cases'].sum()==0
