from copy import deepcopy

import pandas as pd
import pytest
import pulp

from examples.fixture import load_tiny
from optimize import plan_orders
from optimize.dc_solver import solve_dc_blocks


def two_dc_case():
    state,fc,cfg = load_tiny()
    cfg['network'] = dict(dcs=['DC','OTHER_DC'],home_dc={'SHOP':'DC','OTHER':'OTHER_DC'})
    extra = state.store_stock.copy();extra['store_id']='OTHER'
    state.store_stock = pd.concat([state.store_stock,extra],ignore_index=True)
    extra = state.dc_stock.copy();extra['dc_id']='OTHER_DC'
    state.dc_stock = pd.concat([state.dc_stock,extra],ignore_index=True)
    extra = fc.copy();extra['store_id']='OTHER'
    fc = pd.concat([fc,extra],ignore_index=True)
    params=cfg['order_model']['parameters']
    params['store_averages']['OTHER']=deepcopy(params['store_averages']['SHOP'])
    params['dc_budgets']['OTHER_DC']=0
    return state,fc,cfg


def test_independent_dc_solves_match_the_joint_optimum_and_integer_count():
    state,fc,cfg = two_dc_case()
    joint = plan_orders(state,fc,cfg)
    cfg['order_model']['solver'].update(decompose_by_dc=True,parallel_time_reserve_s=1,
                                        warm_start='verified_plan',start_plan_fallback=True)
    split = plan_orders(state,fc,cfg)
    assert joint.status == split.status == 'Optimal'
    assert split.objective == pytest.approx(joint.objective)
    assert split.objective == pytest.approx(0.64)
    assert split.stats['n_int'] == joint.stats['n_int'] == 20
    assert len(split.stats['dc_solves']) == 2
    assert split.stats['incumbent_validated']


def test_dc_split_rejects_a_cross_dc_constraint(tmp_path):
    model=pulp.LpProblem('coupled',pulp.LpMinimize)
    a=pulp.LpVariable('a',0);b=pulp.LpVariable('b',0)
    constraint=a+b>=1
    model+=constraint,'cross'
    model+=a+b
    settings=dict(time_limit_s=15,mip_gap=0,feasibility_tolerance=0.0001,
                   parallel_time_reserve_s=1,log=False,start_plan_fallback=False)
    with pytest.raises(ValueError,match='cross-DC'):
        solve_dc_blocks(model,{('test',):constraint},{a:'A',b:'B'},['A','B'],settings,None,tmp_path)
