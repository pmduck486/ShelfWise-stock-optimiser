from copy import deepcopy
from datetime import date, timedelta

import pandas as pd
import pytest

from examples.fixture import load_tiny
from optimize import Decision, OptimizerPolicy, State
from optimize.policy import PIPELINE_COLUMNS, reschedule_pipeline
from optimize.replay import execute_day, run_replay
from optimize.replay_audit import audit_replay
from optimize.sensitivity import fixed_integer_lp


def physical_case(today=date(2015,11,1),perishable=False):
    _,_,cfg=load_tiny()
    om=cfg['order_model'];p=deepcopy(om['parameters']['items']['STAPLE'])
    p.update(units_per_case=1,cases_per_pallet=1,buy_cost=2.0,
             holding_cost_per_unit_day=0.1,shelf_life_days=2 if perishable else None)
    om['parameters']=dict(items={'SKU':p},store_averages={'SHOP':{'SKU':2}},dc_budgets={'DC':100})
    om.update(store_min_order_cases=1,dc_min_order_pallets=1)
    state=State(today,pd.DataFrame([dict(store_id='SHOP',item_id='SKU',units=5)]),
                pd.DataFrame([dict(dc_id='DC',item_id='SKU',units=10)]),pd.DataFrame(columns=PIPELINE_COLUMNS))
    fc=pd.DataFrame([dict(origin=today,date=today+timedelta(days=t),store_id='SHOP',item_id='SKU',
                         q50=2,q80=2,q90=2,q95=2) for t in range(3)])
    return state,fc,cfg


def manual_decision(state,cfg,store=0,supplier=0):
    day=state.date
    pipeline,adjustments=reschedule_pipeline(state.pipeline,cfg)
    return Decision(day,pd.DataFrame([dict(date=day,store_id='SHOP',item_id='SKU',cases=store,units=store)]),
                    pd.DataFrame([dict(date=day,dc_id='DC',item_id='SKU',pallets=supplier,units=supplier)]),
                    pd.DataFrame([dict(date=day,store_id='SHOP',dc_id='DC',cases_total=store,trucks=1 if store else 0)]),
                    pipeline,adjustments,None,{})


def observed(units):
    return pd.DataFrame([dict(store_id='SHOP',item_id='SKU',units=units)])


def test_policy_returns_both_firm_tables_and_preserves_inputs():
    state,fc,cfg=physical_case();original=deepcopy(state);saved_cfg=deepcopy(cfg)
    policy=OptimizerPolicy();d=policy.decide(state,fc,cfg)
    assert d.audit['passed']
    assert len(d.store_orders)==len(d.dc_orders)==len(d.deliveries)==1
    assert set(d.store_orders.date)==set(d.dc_orders.date)=={state.date}
    assert len(d.plan.orders)==3
    pd.testing.assert_frame_equal(state.store_stock,original.store_stock)
    pd.testing.assert_frame_equal(state.pipeline,original.pipeline)
    assert cfg==saved_cfg


def test_supplier_and_store_commitments_enter_separate_pipelines_once():
    state,fc,cfg=physical_case()
    result=execute_day(state,manual_decision(state,cfg,store=3,supplier=4),fc,observed(2),cfg)
    assert result.state.store_stock.units.sum()==3
    assert result.state.dc_stock.units.sum()==7
    pipe=result.state.pipeline.set_index('location_type')
    assert pipe.loc['store','arrival_date']==state.date+timedelta(days=1)
    assert pipe.loc['dc','arrival_date']==state.date+timedelta(days=5)
    assert pipe.loc['dc','units']==4
    assert result.metrics['purchase_cash']==8
    assert result.metrics['cash_cost']==pytest.approx(8+.1+10*.1+.1) # trip + holding + order-line
    assert result.metrics['conservation_residual_units']==pytest.approx(0)


def test_due_receipts_are_available_but_later_receipts_are_not():
    state,fc,cfg=physical_case()
    state.dc_stock['units']=0
    state.pipeline=pd.DataFrame([dict(order_date=state.date-timedelta(days=5),arrival_date=state.date,
        location_type='dc',location_id='DC',item_id='SKU',units=3)])
    r=execute_day(state,manual_decision(state,cfg,store=3),fc,observed(2),cfg)
    assert r.state.dc_stock.units.sum()==0
    state.pipeline['arrival_date']=state.date+timedelta(days=1)
    with pytest.raises(ValueError,match='physically available'):
        execute_day(state,manual_decision(state,cfg,store=3),fc,observed(2),cfg)


def test_committed_christmas_arrival_postponed_and_logged_without_mutation():
    state,fc,cfg=physical_case(date(2015,12,24))
    state.pipeline=pd.DataFrame([dict(order_date=date(2015,12,23),arrival_date=date(2015,12,25),
        location_type='store',location_id='SHOP',item_id='SKU',units=3)])
    original=state.pipeline.copy(deep=True)
    d=OptimizerPolicy().decide(state,fc,cfg)
    assert d.effective_pipeline.arrival_date.iloc[0]==date(2015,12,26)
    assert len(d.pipeline_adjustments)==1
    assert d.store_orders.units.sum()==0
    pd.testing.assert_frame_equal(state.pipeline,original)


def test_closure_still_consumes_calendar_shelf_life_and_holding_cash():
    state,fc,cfg=physical_case(date(2015,12,24),perishable=True)
    # A life-2 item left on 24 Dec cannot be kept for 26 Dec: the next day is closed.
    result=execute_day(state,manual_decision(state,cfg),fc,observed(1),cfg)
    assert result.metrics['waste_proxy_units']==4
    assert result.state.store_stock.units.sum()==0
    assert result.metrics['disposal_cash']==pytest.approx(4*2*.05)
    next_fc=fc.copy();next_fc['origin']=date(2015,12,25);next_fc['date']+=timedelta(days=1)
    closed=execute_day(result.state,manual_decision(result.state,cfg),next_fc,observed(10),cfg)
    assert closed.metrics['actual_demand']==closed.metrics['lost_units']==0
    assert closed.metrics['ignored_closed_day_recorded_units']==10
    assert closed.metrics['holding_cash']==1
    assert closed.state.date==date(2015,12,26)


def test_fractional_firm_orders_and_missing_actuals_fail_loudly():
    state,fc,cfg=physical_case()
    with pytest.raises(ValueError,match='Fractional'):
        execute_day(state,manual_decision(state,cfg,store=.5),fc,observed(2),cfg)
    with pytest.raises(ValueError,match='complete store-item'):
        execute_day(state,manual_decision(state,cfg),fc,observed(2).iloc[:0],cfg)


def test_two_day_replay_and_future_poison(tmp_path):
    state,fc,cfg=physical_case()
    second=fc.copy();second['origin']+=timedelta(days=1);second['date']+=timedelta(days=1)
    forecasts=pd.concat([fc,second],ignore_index=True)
    actual=pd.DataFrame([dict(date=state.date+timedelta(days=t),store_id='SHOP',item_id='SKU',units=2) for t in range(2)])
    first,_=run_replay(state,forecasts,actual,cfg,end_date=state.date+timedelta(days=1),output_dir=tmp_path/'a')
    actual.loc[1,'units']=200
    second,_=run_replay(state,forecasts,actual,cfg,end_date=state.date+timedelta(days=1),output_dir=tmp_path/'b')
    a=pd.read_csv(tmp_path/'a'/'order_plan.csv');b=pd.read_csv(tmp_path/'b'/'order_plan.csv')
    pd.testing.assert_frame_equal(a.iloc[:1],b.iloc[:1])
    assert first['days']==second['days']==2
    assert first['all_plan_audits_passed']
    assert first['max_conservation_residual_units']<cfg['order_model']['solver']['feasibility_tolerance']
    actual.loc[1,'units']=2
    audit=audit_replay(state,forecasts,actual,cfg,tmp_path/'a')
    assert audit['passed'] and audit['store_balance_checks']==2
    daily=pd.read_csv(tmp_path/'a'/'daily_metrics.csv')
    daily.loc[0,'cash_cost']+=1
    daily.to_csv(tmp_path/'a'/'daily_metrics.csv',index=False)
    with pytest.raises(ValueError,match='cash_cost'):
        audit_replay(state,forecasts,actual,cfg,tmp_path/'a')


def test_replay_requires_every_calendar_day_and_never_holdout(tmp_path):
    state,fc,cfg=physical_case()
    actual=observed(2).assign(date=state.date)
    with pytest.raises(ValueError,match='Incomplete observed'):
        run_replay(state,fc,actual,cfg,end_date=state.date+timedelta(days=1),output_dir=tmp_path/'missing')
    cfg['data']=dict(windows=dict(test_start='2015-11-01',test_end='2015-12-26'))
    with pytest.raises(ValueError,match='holdout'):
        run_replay(state,fc,actual,cfg,end_date=date(2015,12,27),output_dir=tmp_path/'holdout')


def test_fixed_integer_lp_preserves_milp_and_tuple_constraints():
    state,fc,cfg=physical_case()
    plan=OptimizerPolicy().decide(state,fc,cfg).plan
    original={v.name:(v.cat,v.lowBound,v.upBound,v.varValue) for v in plan.model.variables()}
    lp,constraints,summary=fixed_integer_lp(plan,cfg)
    assert summary['fixed_integer_count']==plan.stats['n_int']
    assert set(constraints)==set(plan.constraints)
    assert not any(v.isInteger() for v in lp.variables())
    assert summary['objective']<=plan.objective+cfg['order_model']['solver']['feasibility_tolerance']
    assert original=={v.name:(v.cat,v.lowBound,v.upBound,v.varValue) for v in plan.model.variables()}


def test_shadow_price_predicts_a_small_shelf_rhs_change():
    import pulp
    from optimize import plan_orders
    state,fc,cfg=load_tiny()
    state.store_stock.loc[state.store_stock.item_id=='FRESH','units']=30
    plan=plan_orders(state,fc,cfg)
    lp,constraints,summary=fixed_integer_lp(plan,cfg)
    shelf=constraints['shelf','SHOP','FRESH',1]
    dual=shelf.pi
    # With constant morning stock PuLP stores this as overflow >= morning-cap.
    # More shelf capacity decreases that stored RHS; its raw dual is positive.
    assert dual==cfg['order_model']['overflow_penalty_share']
    delta=.1
    shelf.changeRHS(-shelf.constant-delta)
    lp.solve(pulp.PULP_CBC_CMD(msg=False,mip=False))
    assert pulp.value(lp.objective)-summary['objective']==pytest.approx(-dual*delta)
