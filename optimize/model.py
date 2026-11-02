"""Deterministic replenishment MILP with continuous tentative future decisions.

This checkpoint accepts compiled item/store parameters in cfg. Historical parameter
calibration, replay policy integration and exact age tracking are separate steps.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
import json
import math
from pathlib import Path
import re
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import pulp

from .behavior import service_priority_weights, work_conserving_path
from .cbc import cbc_solver
from .dc_solver import solve_dc_blocks


ORDER_COLUMNS = ['date', 'store_id', 'item_id', 'cases', 'units']
DC_COLUMNS = ['date', 'dc_id', 'item_id', 'pallets', 'units']
DELIVERY_COLUMNS = ['date', 'store_id', 'dc_id', 'cases_total', 'trucks']
QUANTILES = ['q50', 'q80', 'q90', 'q95']
LAYERS = ['presentation_stock', 'days_of_supply_cap', 'order_calendar',
          'receiving_capacity', 'fair_share', 'age_tracking', 'supply_scenarios']


@dataclass
class State:
    date: date
    store_stock: pd.DataFrame
    dc_stock: pd.DataFrame
    pipeline: pd.DataFrame


@dataclass
class Plan:
    orders: pd.DataFrame
    dc_orders: pd.DataFrame
    deliveries: pd.DataFrame
    objective: float
    status: str
    model: pulp.LpProblem
    constraints: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def _nonnegative(value, label, *, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{label} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def _stock_table(frame, location_column, locations, items):
    required = [location_column, 'item_id', 'units']
    if not set(required) <= set(frame):
        raise ValueError(f'Stock table needs {required}')
    frame = frame[required].copy()
    if frame.duplicated(required[:-1]).any():
        raise ValueError('Duplicate stock keys')
    expected = {(location, item) for location in locations for item in items}
    values = {(r[0], r[1]): _nonnegative(r[2], 'stock units')
              for r in frame.itertuples(index=False, name=None)}
    if set(values) != expected:
        raise ValueError('Stock table must contain exactly the complete active location/item grid')
    return values


def _solver_gap(log, status, objective, tolerance):
    """CBC prints its gap to only two decimals; retain the bound-based estimate."""
    lower = re.search(r'Lower bound:\s*([-+\deE.]+)', log)
    rounded = re.search(r'^Gap:\s*([-+\deE.]+)', log, flags=re.MULTILINE)
    bound = float(lower[1]) if lower else None
    reported = float(rounded[1]) if rounded else None
    if objective is not None and bound is not None:
        gap = max(float(objective)-bound, 0)/max(abs(float(objective)), tolerance)
        source = 'incumbent_vs_printed_lower_bound'
    elif status == 'Optimal':
        gap, source = 0.0, 'proven_optimal_status'
    elif objective is not None and reported is not None and reported > 0:
        gap, source = reported, 'rounded_CBC_log'
    else:
        gap, source = None, 'unavailable'
    return dict(best_bound=bound, mip_gap=gap, cbc_reported_gap=reported,
                mip_gap_source=source)


def plan_orders(state: State, fc: pd.DataFrame, cfg: dict) -> Plan:
    started = perf_counter()
    om = cfg['order_model']
    for layer in LAYERS:
        if om.get('layers', {}).get(layer, om.get(layer, False)):
            raise NotImplementedError(f'Layer {layer} is not implemented in the core checkpoint')
    if om['integer_days'] != 1:
        raise ValueError('Only day 1 may be integer in this core')
    horizon = om['horizon_days']
    ls, lv = om['lead_time_store_days'], om['lead_time_supplier_days']
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in [horizon, ls, lv]):
        raise ValueError('Horizon and lead times must be positive calendar-day integers')
    origin = pd.Timestamp(state.date).date()
    dates = [origin + timedelta(days=t) for t in range(horizon)]
    params = om['parameters']
    item_params = params['items']
    items = list(item_params)
    stores = list(state.store_stock.store_id.drop_duplicates())
    dcs = list(state.dc_stock.dc_id.drop_duplicates())
    if not items or not stores or not dcs:
        raise ValueError('Active stores, DCs and items must not be empty')
    homes = cfg['network']['home_dc']
    if any(s not in homes or homes[s] not in dcs for s in stores):
        raise ValueError('Every active store needs an active home DC')
    if not set(dcs) <= set(cfg['network']['dcs']):
        raise ValueError('Unknown DC')
    initial_store = _stock_table(state.store_stock, 'store_id', stores, items)
    initial_dc = _stock_table(state.dc_stock, 'dc_id', dcs, items)
    budgets = {d: _nonnegative(params['dc_budgets'][d], 'DC budget') for d in dcs}
    settings = om['solver']
    tol = _nonnegative(settings['feasibility_tolerance'], 'feasibility tolerance', positive=True)
    limit = _nonnegative(settings['time_limit_s'], 'solver time limit', positive=True)
    gap = _nonnegative(settings['mip_gap'], 'solver relative gap')
    truck_cap = _nonnegative(om['truck_capacity_cases'], 'truck capacity', positive=True)
    trip_cost = _nonnegative(om['truck_trip_cost'], 'trip cost')
    line_cost = _nonnegative(om['order_line_cost'], 'order-line cost')
    minimum = _nonnegative(om['store_min_order_cases'], 'minimum cartons', positive=True)
    if minimum != int(minimum):
        raise ValueError('Firm minimum order must be whole cartons')
    if om['dc_min_order_pallets'] != 1:
        raise NotImplementedError('Supplier minimum above one pallet needs its own activation model')
    pi_share = _nonnegative(om['service_shortfall_penalty_share'], 'service penalty')
    loss_share = _nonnegative(om['lost_sale_penalty_share'], 'lost-sale penalty')
    overflow_share = _nonnegative(om['overflow_penalty_share'], 'overflow penalty', positive=True)
    disposal_share = _nonnegative(om['waste_disposal_share'], 'disposal share')
    terminal_store_days = _nonnegative(om['terminal_store_days'], 'terminal store days')
    terminal_dc_days = _nonnegative(om['terminal_dc_days'], 'terminal DC days')
    fulfillment_policy = om.get('fulfillment_policy', 'legacy_allocation')
    reserve_policy = om.get('warehouse_reserve_policy', 'terminal_only')
    if fulfillment_policy not in {'legacy_allocation', 'earlier_service_priority'}:
        raise ValueError('Unknown fulfillment policy')
    if reserve_policy not in {'terminal_only', 'daily_inventory_position'}:
        raise ValueError('Unknown warehouse reserve policy')
    priority_margin = (_nonnegative(om['early_service_margin_share'], 'early service margin', positive=True)
                       if fulfillment_policy == 'earlier_service_priority' else 0)
    projection_tolerance = (_nonnegative(om['postsolve_objective_tolerance'], 'postsolve objective tolerance')
                            if fulfillment_policy == 'earlier_service_priority' else 0)
    dc_cover_share = (_nonnegative(om['dc_position_penalty_share'], 'DC protection penalty', positive=True)
                      if reserve_policy == 'daily_inventory_position' else 0)
    dc_terminal_multiplier = (_nonnegative(om['terminal_dc_penalty_cost_multiplier'], 'DC terminal cost multiplier', positive=True)
                              if reserve_policy == 'daily_inventory_position' else None)
    if dc_terminal_multiplier is not None and dc_terminal_multiplier <= 1:
        raise ValueError('DC terminal penalty multiplier must exceed procurement cost')
    backroom = om['backroom_capacity_units']
    if backroom is not None:
        backroom = _nonnegative(backroom, 'backroom capacity')
    store_closed = set(om['calendar']['closed_store_month_days'])
    dc_closed = set(om['calendar']['closed_dc_month_days'])
    closed = lambda when: when.strftime('%m-%d') in store_closed

    if not set(['origin', 'date', 'store_id', 'item_id'] + QUANTILES) <= set(fc):
        raise ValueError('Forecast columns do not match the agreed contract')
    forecast = fc.copy()
    forecast['origin'] = pd.to_datetime(forecast.origin).dt.date
    forecast['date'] = pd.to_datetime(forecast.date).dt.date
    forecast = forecast[(forecast.origin == origin) & forecast.date.isin(dates)]
    keys = ['store_id', 'item_id', 'date']
    if forecast.duplicated(keys).any():
        raise ValueError('Duplicate forecast keys for this origin')
    expected = {(s, p, dt) for s in stores for p in items for dt in dates}
    if set(forecast[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('Forecast must contain the complete active store/item/horizon grid')
    qs = forecast[QUANTILES].to_numpy(dtype=float)
    if not np.isfinite(qs).all() or (qs < 0).any() or (np.diff(qs, axis=1) < 0).any():
        raise ValueError('Forecast quantiles must be finite, nonnegative and noncrossing')
    closure_mask = forecast.date.map(closed)
    changed_closed_rows = int((closure_mask & (forecast[QUANTILES].max(axis=1) > 0)).sum())
    forecast.loc[closure_mask, QUANTILES] = 0
    lookup = forecast.set_index(keys)

    case, pallet_cases, buy, price, holding, life, qcol = {}, {}, {}, {}, {}, {}, {}
    for p, info in item_params.items():
        case[p] = _nonnegative(info['units_per_case'], 'units per carton', positive=True)
        pallet_cases[p] = _nonnegative(info['cases_per_pallet'], 'cartons per pallet', positive=True)
        if case[p] != int(case[p]) or pallet_cases[p] != int(pallet_cases[p]):
            raise ValueError('Pack sizes must be positive integers')
        price[p] = _nonnegative(info['price'], 'price', positive=True)
        buy[p] = _nonnegative(info['buy_cost'], 'purchase cost', positive=True)
        holding[p] = _nonnegative(info['holding_cost_per_unit_day'], 'holding cost')
        life[p] = info['shelf_life_days']
        if life[p] is not None and (not isinstance(life[p], int) or life[p] <= 0):
            raise ValueError('Shelf life must be a positive integer or null')
        qcol[p] = {0.5: 'q50', 0.8: 'q80', 0.9: 'q90', 0.95: 'q95'}[info['service_level']]
    demand = {(s, p, t): float(lookup.loc[(s, p, dt), 'q50'])
              for s in stores for p in items for t, dt in enumerate(dates)}
    safety = {(s, p, t): float(lookup.loc[(s, p, dt), qcol[p]])
              for s in stores for p in items for t, dt in enumerate(dates)}
    mean = {(s, p): sum(demand[s, p, t] for t in range(horizon))/horizon
            for s in stores for p in items}
    lost_weights = {p: (service_priority_weights(horizon, loss_share*price[p], pi_share*price[p],
                                               pi_share*price[p], priority_margin*price[p])
                        if fulfillment_policy == 'earlier_service_priority' else
                        [loss_share*price[p]]*horizon) for p in items}
    cap = {}
    for s in stores:
        for p in items:
            avg = _nonnegative(params['store_averages'][s][p], 'frozen store average')
            days = om['shelf_days_perishable'] if life[p] is not None else om['shelf_days']
            cap[s, p] = max(_nonnegative(days, 'shelf days') * avg,
                            _nonnegative(om['shelf_min_cases'], 'shelf minimum cases') * case[p])

    pipeline = state.pipeline.copy()
    required_pipe = ['order_date', 'arrival_date', 'location_type', 'location_id', 'item_id', 'units']
    if not set(required_pipe) <= set(pipeline):
        raise ValueError('Pipeline columns do not match the agreed contract')
    pipe_rows = []
    for row in pipeline[required_pipe].itertuples(index=False, name=None):
        ordered, arrived = pd.Timestamp(row[0]).date(), pd.Timestamp(row[1]).date()
        kind, location, item = str(row[2]).lower(), row[3], row[4]
        units = _nonnegative(row[5], 'pipeline units')
        valid_locations = stores if kind == 'store' else dcs if kind == 'dc' else []
        if location not in valid_locations or item not in items or arrived < origin or ordered >= origin or arrived < ordered:
            raise ValueError('Invalid pipeline location, item or date')
        if kind == 'store' and closed(arrived):
            raise ValueError('Committed delivery on store closure needs L4 rescheduling before solving')
        pipe_rows.append((arrived, kind, location, item, units))
    # Rolling states have many commitments. Index them once rather than scan
    # the entire pipeline for every store/item/date constraint.
    arrival_totals = {}
    dc_pipeline = {}
    for arrived,kind,location,item,units in pipe_rows:
        key=(kind,location,item,arrived)
        arrival_totals[key]=arrival_totals.get(key,0.0)+units
        if kind=='dc':
            dc_pipeline.setdefault((location,item),[]).append((arrived,units))
    arrivals = lambda kind, loc, item, dt: arrival_totals.get((kind,loc,item,dt),0.0)

    model = pulp.LpProblem('RetailOps', pulp.LpMinimize)
    constraints = {}

    def add(key, expression):
        model.addConstraint(expression, name=f'C_{len(constraints)}')
        constraints[key] = expression

    x, y, inventory, service, lost, waste, overflow, w, dc_inventory, trucks = [{} for _ in range(10)]
    max_cases = {}
    for si, s in enumerate(stores):
        d = homes[s]
        for pi, p in enumerate(items):
            for t, dt in enumerate(dates):
                firm = t == 0
                cumulative = initial_dc[d, p] + sum(n for a,n in dc_pipeline.get((d,p),[]) if a<=dt)
                if t >= lv:
                    cumulative += budgets[d] / buy[p]
                upper = cumulative / case[p]
                if firm:
                    upper = math.floor(upper)
                # Tail store dispatches arrive outside the model and have no terminal value.
                if closed(dt + timedelta(days=ls)) or dt.strftime('%m-%d') in dc_closed or t+ls >= horizon:
                    upper = 0
                key = (s, p, t)
                max_cases[key] = upper
                suffix = f'{si}_{pi}_{t}'
                category = pulp.LpInteger if firm else pulp.LpContinuous
                x[key] = pulp.LpVariable('x_'+suffix, 0, upper, cat=category)
                y[key] = pulp.LpVariable('y_'+suffix, 0, 1, cat=pulp.LpBinary if firm else pulp.LpContinuous)
                inventory[key] = pulp.LpVariable('I_'+suffix, 0)
                service[key] = pulp.LpVariable('u_'+suffix, 0)
                lost[key] = pulp.LpVariable('lost_'+suffix, 0, demand[key])
                overflow[key] = pulp.LpVariable('overflow_'+suffix, 0, backroom)
                waste[key] = pulp.LpVariable('waste_'+suffix, 0) if life[p] is not None else 0
    for di, d in enumerate(dcs):
        for pi, p in enumerate(items):
            upper = budgets[d] / (buy[p] * case[p] * pallet_cases[p])
            for t in range(horizon):
                suffix = f'{di}_{pi}_{t}'
                w[d, p, t] = pulp.LpVariable('w_'+suffix, 0, math.floor(upper) if t == 0 else upper,
                                            cat=pulp.LpInteger if t == 0 else pulp.LpContinuous)
                dc_inventory[d, p, t] = pulp.LpVariable('J_'+suffix, 0)
    for si, s in enumerate(stores):
        for t in range(horizon):
            upper = math.ceil(sum(max_cases[s, p, t] for p in items)/truck_cap)
            trucks[s, t] = pulp.LpVariable(f'trucks_{si}_{t}', 0, upper,
                                          cat=pulp.LpInteger if t == 0 else pulp.LpContinuous)
    zs = {(s, p): pulp.LpVariable(f'z_store_{si}_{pi}', 0)
          for si, s in enumerate(stores) for pi, p in enumerate(items)}
    zd = {(d, p): pulp.LpVariable(f'z_dc_{di}_{pi}', 0)
          for di, d in enumerate(dcs) for pi, p in enumerate(items)}
    zcover = {(d,p,t): pulp.LpVariable(f'z_protect_{di}_{pi}_{t}',0)
              for di,d in enumerate(dcs) for pi,p in enumerate(items) for t in range(horizon-1)
              if reserve_policy == 'daily_inventory_position'}
    protection_expressions = {}

    for s in stores:
        for p in items:
            for t, dt in enumerate(dates):
                key = (s, p, t)
                previous = inventory[s, p, t-1] if t else initial_store[s, p]
                received = arrivals('store', s, p, dt) + (case[p]*x[s, p, t-ls] if t >= ls else 0)
                morning = previous + received
                add(('store_balance', s, p, t+1), inventory[key] == morning-demand[key]+lost[key]-waste[key])
                add(('safety', s, p, t+1), morning+service[key] >= safety[key])
                add(('shelf', s, p, t+1), morning <= cap[s, p]+overflow[key])
                add(('minimum', s, p, t+1), x[key] >= minimum*y[key])
                add(('activation', s, p, t+1), x[key] <= max_cases[key]*y[key])
                if t == 0:
                    # Day-1 availability is fixed: no artificial refusal to sell existing stock.
                    fixed_morning = initial_store[s, p] + arrivals('store', s, p, dt)
                    add(('physical_day1_sales', s, p), lost[key] == max(demand[key]-fixed_morning, 0))
                if life[p] is not None:
                    cover = 0
                    for offset in range(1, life[p]):
                        future = dt+timedelta(days=offset)
                        if not closed(future):
                            cover += demand[s, p, t+offset] if t+offset < horizon else mean[s, p]
                    add(('waste', s, p, t+1), waste[key] >= morning-demand[key]+lost[key]-cover)
            add(('terminal_store', s, p), inventory[s, p, horizon-1]+zs[s, p] >= terminal_store_days*mean[s, p])
        for t in range(horizon):
            add(('truck', s, t+1), pulp.lpSum(x[s, p, t] for p in items) <= truck_cap*trucks[s, t])
    for d in dcs:
        for p in items:
            for t, dt in enumerate(dates):
                previous = dc_inventory[d, p, t-1] if t else initial_dc[d, p]
                received = arrivals('dc', d, p, dt) + (case[p]*pallet_cases[p]*w[d, p, t-lv] if t >= lv else 0)
                sent = pulp.lpSum(case[p]*x[s, p, t] for s in stores if homes[s] == d)
                add(('dc_balance', d, p, t+1), dc_inventory[d, p, t] == previous+received-sent)
            outstanding = pulp.lpSum(case[p]*pallet_cases[p]*w[d, p, t]
                                      for t in range(horizon) if t+lv >= horizon)
            pending = sum(n for a,n in dc_pipeline.get((d,p),[]) if a>dates[-1])
            target = terminal_dc_days*sum(mean[s, p] for s in stores if homes[s] == d)
            add(('terminal_dc', d, p), dc_inventory[d, p, horizon-1]+outstanding+pending+zd[d, p] >= target)
            if reserve_policy == 'daily_inventory_position':
                for t,dt in enumerate(dates[:-1]):
                    # Only orders already placed by t count. A future tentative
                    # purchase cannot satisfy today's protection target.
                    on_order = pulp.lpSum(case[p]*pallet_cases[p]*w[d,p,order_t]
                                           for order_t in range(t+1) if order_t+lv > t)
                    existing = sum(n for a,n in dc_pipeline.get((d,p),[]) if dt<a<=dt+timedelta(days=lv))
                    position = dc_inventory[d,p,t]+on_order+existing
                    coverage = 0.0
                    for offset in range(1,lv+1):
                        target_date = dt+timedelta(days=offset)
                        if not closed(target_date):
                            coverage += sum(demand[s,p,t+offset] if t+offset < horizon else mean[s,p]
                                            for s in stores if homes[s] == d)
                    protection_expressions[d,p,t] = (position,coverage)
                    add(('dc_protection',d,p,t+1),position+zcover[d,p,t] >= coverage)
        add(('budget', d), pulp.lpSum(buy[p]*case[p]*pallet_cases[p]*w[d, p, t]
                                      for p in items for t in range(horizon)) <= budgets[d])

    costs = {
        'purchase': pulp.lpSum(buy[p]*case[p]*pallet_cases[p]*v for (d, p, t), v in w.items()),
        'order_lines': line_cost*pulp.lpSum(y.values()),
        'trucks': trip_cost*pulp.lpSum(trucks.values()),
        'holding': pulp.lpSum(holding[p]*v for (s, p, t), v in inventory.items()) +
                   pulp.lpSum(holding[p]*v for (d, p, t), v in dc_inventory.items()),
        'service': pulp.lpSum(pi_share*price[p]*v for (s, p, t), v in service.items()),
        'lost_sales': pulp.lpSum(loss_share*price[p]*v for (s, p, t), v in lost.items()),
        'early_service_priority': pulp.lpSum((lost_weights[p][t]-loss_share*price[p])*v
                                             for (s,p,t),v in lost.items()),
        'waste_penalty': pulp.lpSum(buy[p]*(1+disposal_share)*v for (s, p, t), v in waste.items()),
        'overflow': pulp.lpSum(overflow_share*price[p]*v for (s, p, t), v in overflow.items()),
        'terminal': pulp.lpSum(pi_share*price[p]*v for (s, p), v in zs.items())+
                    pulp.lpSum((dc_terminal_multiplier*buy[p] if dc_terminal_multiplier is not None
                                else pi_share*price[p])*v for (d, p), v in zd.items()),
        'dc_protection': pulp.lpSum(dc_cover_share*price[p]*v for (d,p,t),v in zcover.items()),
    }
    model.setObjective(pulp.lpSum(costs.values()))
    variables = model.variables()
    def candidate_valid():
        for v in variables:
            n = v.varValue
            if n is None or not math.isfinite(n):
                return False
            if v.lowBound is not None and n < v.lowBound-tol:
                return False
            if v.upBound is not None and n > v.upBound+tol:
                return False
            if v.isInteger() and abs(n-round(n)) > tol:
                return False
        for c in constraints.values():
            residual = c.value()
            if residual is None or (c.sense == pulp.LpConstraintEQ and abs(residual) > tol):
                return False
            if (c.sense == pulp.LpConstraintLE and residual > tol) or (c.sense == pulp.LpConstraintGE and residual < -tol):
                return False
        return True

    def populate_start(seed=None):
        for v in variables:
            v.varValue = 0.0
        if seed is not None:
            for table,label,location,quantity in [(x,'orders','store_id','cases'),(w,'dc_orders','dc_id','pallets')]:
                supplied = seed[label].copy()
                supplied['date'] = pd.to_datetime(supplied.date).dt.date
                records = supplied.set_index([location,'item_id','date'])[quantity]
                for (loc,p,t),v in table.items():
                    n = float(records.get((loc,p,dates[t]),0))
                    v.varValue = float(round(n)) if t == 0 else n
            for key,v in y.items():
                amount = x[key].varValue
                v.varValue = float(amount > 0) if key[-1] == 0 else amount/max_cases[key] if max_cases[key] else 0
            for (s,t),v in trucks.items():
                load = sum(x[s,p,t].varValue for p in items)/truck_cap
                v.varValue = float(math.ceil(load)) if t == 0 else load
        for d in dcs:
            for p in items:
                previous = initial_dc[d,p]
                for t,dt in enumerate(dates):
                    incoming = arrivals('dc',d,p,dt)+(case[p]*pallet_cases[p]*w[d,p,t-lv].varValue if t >= lv else 0)
                    outgoing = sum(case[p]*x[s,p,t].varValue for s in stores if homes[s] == d)
                    previous = previous+incoming-outgoing
                    dc_inventory[d,p,t].varValue = max(previous,0)
        for s in stores:
            for p in items:
                previous = initial_store[s,p]
                for t,dt in enumerate(dates):
                    incoming = arrivals('store',s,p,dt)+(case[p]*x[s,p,t-ls].varValue if t >= ls else 0)
                    morning = previous+incoming
                    missing = max(demand[s,p,t]-morning,0)
                    remaining = max(morning-demand[s,p,t],0)
                    cover = 0.0
                    if life[p] is not None:
                        for step in range(1,life[p]):
                            future = dt+timedelta(days=step)
                            if not closed(future):
                                cover += demand[s,p,t+step] if t+step < horizon else mean[s,p]
                        waste[s,p,t].varValue = max(remaining-cover,0)
                    previous = remaining-(waste[s,p,t].varValue if life[p] is not None else 0)
                    inventory[s,p,t].varValue = previous
                    lost[s,p,t].varValue = missing
                    service[s,p,t].varValue = max(safety[s,p,t]-morning,0)
                    overflow[s,p,t].varValue = max(morning-cap[s,p],0)
                zs[s,p].varValue = max(terminal_store_days*mean[s,p]-previous,0)
        for d in dcs:
            for p in items:
                pending = sum(n for a,n in dc_pipeline.get((d,p),[]) if a>dates[-1])
                pending += sum(case[p]*pallet_cases[p]*w[d,p,t].varValue for t in range(horizon) if t+lv >= horizon)
                target = terminal_dc_days*sum(mean[s,p] for s in stores if homes[s] == d)
                zd[d,p].varValue = max(target-dc_inventory[d,p,horizon-1].varValue-pending,0)
        for key,(position,target) in protection_expressions.items():
            zcover[key].varValue = max(target-float(pulp.value(position)),0)

    def replenishment_start():
        """Construct a forecast-only feasible hint; never fix its decisions.

        Follow physical q50 stock, top up toward the existing service line on
        arrival, and restore DC lead-time position subject to the same budget.
        All candidate constraints are independently checked before use.
        """
        for v in variables:
            v.varValue=0.0
        store_closing=dict(initial_store)
        warehouse=dict(initial_dc)
        remaining_budget=dict(budgets)
        for t,dt in enumerate(dates):
            # Physical current-day store stock, after sales and expected disposal.
            for s in stores:
                for p in items:
                    incoming=arrivals('store',s,p,dt)
                    if t>=ls:incoming+=case[p]*x[s,p,t-ls].varValue
                    closing=max(store_closing[s,p]+incoming-demand[s,p,t],0)
                    if life[p] is not None:
                        cover=sum(demand[s,p,t+step] if t+step<horizon else mean[s,p]
                                  for step in range(1,life[p]) if not closed(dt+timedelta(days=step)))
                        closing=min(closing,cover)
                    store_closing[s,p]=closing
            for d in dcs:
                home_stores=[s for s in stores if homes[s]==d]
                for p in items:
                    available=warehouse[d,p]+arrivals('dc',d,p,dt)
                    if t>=lv:available+=case[p]*pallet_cases[p]*w[d,p,t-lv].varValue
                    wanted={}
                    for s in home_stores:
                        projected=store_closing[s,p]
                        arrival_t=t+ls
                        if arrival_t>=horizon or max_cases[s,p,t]<minimum:
                            wanted[s]=0.0
                            continue
                        # Account for earlier commitments already on the way.
                        for future_t in range(t+1,arrival_t+1):
                            projected+=arrivals('store',s,p,dates[future_t])
                            dispatch_t=future_t-ls
                            if 0<=dispatch_t<t:projected+=case[p]*x[s,p,dispatch_t].varValue
                            if future_t<arrival_t:
                                projected=max(projected-demand[s,p,future_t],0)
                                if life[p] is not None:
                                    cover=sum(demand[s,p,future_t+step] if future_t+step<horizon else mean[s,p]
                                              for step in range(1,life[p]) if not closed(dates[future_t]+timedelta(days=step)))
                                    projected=min(projected,cover)
                        quantity=max(safety[s,p,arrival_t]-projected,0)/case[p]
                        if t==0:
                            quantity=max(minimum,math.ceil(quantity)) if quantity>0 else 0
                        wanted[s]=min(quantity,max_cases[s,p,t])
                    load=sum(case[p]*n for n in wanted.values())
                    scale=min(available/load,1.0) if load>0 else 1.0
                    for s,n in wanted.items():
                        quantity=n*scale
                        if t==0:
                            quantity=math.floor(quantity)
                            if quantity<minimum:quantity=0.0
                        x[s,p,t].varValue=quantity
                    warehouse[d,p]=max(available-sum(case[p]*x[s,p,t].varValue for s in home_stores),0)
                    if reserve_policy=='daily_inventory_position':
                        target=sum(sum(demand[s,p,t+step] if t+step<horizon else mean[s,p]
                                       for s in home_stores) for step in range(1,lv+1)
                                   if not closed(dt+timedelta(days=step)))
                        position=warehouse[d,p]+sum(n for a,n in dc_pipeline.get((d,p),[]) if dt<a<=dt+timedelta(days=lv))
                        position+=sum(case[p]*pallet_cases[p]*w[d,p,placed].varValue
                                      for placed in range(t) if placed+lv>t)
                        unit_pack=case[p]*pallet_cases[p]
                        needed=max(target-position,0)/unit_pack
                        quantity=math.ceil(needed) if t==0 else needed
                        affordable=remaining_budget[d]/(buy[p]*unit_pack)
                        if t==0:affordable=math.floor(affordable)
                        quantity=min(quantity,affordable,w[d,p,t].upBound)
                        if t==0 and quantity<om['dc_min_order_pallets']:quantity=0.0
                        w[d,p,t].varValue=max(quantity,0)
                        remaining_budget[d]-=quantity*unit_pack*buy[p]
        seed=dict(orders=pd.DataFrame([dict(date=dates[t],store_id=s,item_id=p,cases=v.varValue)
                                      for (s,p,t),v in x.items()]),
                  dc_orders=pd.DataFrame([dict(date=dates[t],dc_id=d,item_id=p,pallets=v.varValue)
                                          for (d,p,t),v in w.items()]))
        populate_start(seed)

    start_values, start_objective, start_source = None, None, None
    replenishment_start_issues=[]
    if settings.get('warm_start') == 'verified_plan':
        seed = om.get('warm_start_plan')
        populate_start(seed)
        if not candidate_valid():
            populate_start()
            seed = None
        if candidate_valid():
            start_values = {v:v.varValue for v in variables}
            start_objective = float(pulp.value(model.objective))
            start_source = 'supplied_orders' if seed is not None else 'zero_orders'
        if settings.get('replenishment_start',False):
            replenishment_start()
            if candidate_valid():
                objective=float(pulp.value(model.objective))
                if start_objective is None or objective<start_objective:
                    start_values={v:v.varValue for v in variables}
                    start_objective=objective
                    start_source='forecast_replenishment'
            else:
                for key,c in constraints.items():
                    value=c.value()
                    if value is None or (c.sense==pulp.LpConstraintEQ and abs(value)>tol) or (c.sense==pulp.LpConstraintLE and value>tol) or (c.sense==pulp.LpConstraintGE and value<-tol):
                        replenishment_start_issues.append(dict(constraint=str(key),residual=value))
                        if len(replenishment_start_issues)>=10:break
            if start_values is not None:
                for v,value in start_values.items():v.varValue=value
    build_seconds = perf_counter()-started
    dc_results = None
    with tempfile.TemporaryDirectory(prefix='retailops-cbc-') as temp:
        log_file = Path(temp)/'cbc.log'
        if settings.get('decompose_by_dc',False) and len(dcs)>1:
            owner = {}
            for table in [x,y,inventory,service,lost,waste,overflow,trucks,zs]:
                for key,v in table.items():
                    if isinstance(v,pulp.LpVariable):owner[v]=homes[key[0]]
            for table in [w,dc_inventory,zd,zcover]:
                for key,v in table.items():owner[v]=key[0]
            dc_results,solve_seconds = solve_dc_blocks(model,constraints,owner,dcs,settings,start_values,temp)
            valid_blocks=all(r['validated'] for r in dc_results)
            exact=valid_blocks and all(r['status']=='Optimal' for r in dc_results)
            model.status=pulp.LpStatusOptimal if exact else pulp.LpStatusNotSolved
            model.sol_status=(pulp.LpSolutionOptimal if exact else pulp.LpSolutionIntegerFeasible
                              if valid_blocks else pulp.LpSolutionNoSolutionFound)
            solver_error=None if valid_blocks else 'No validated incumbent for one or more DC blocks'
            log='\n'.join('=== '+str(r['dc_id'])+' ===\n'+r['log'] for r in dc_results)
        else:
            solver = cbc_solver(msg=bool(settings['log']),timeLimit=limit,gapRel=gap,
                               logPath=str(log_file),warmStart=start_values is not None)
            if not solver.available():
                raise RuntimeError('CBC unavailable: install the pinned requirements.txt')
            solve_started = perf_counter()
            try:
                model.solve(solver)
                solver_error = None
            except pulp.PulpSolverError as exc:
                solver_error = str(exc)
            solve_seconds = perf_counter()-solve_started
            log = log_file.read_text(errors='replace') if log_file.exists() else ''
    incumbent_reported = model.sol_status in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible)
    valid = incumbent_reported and not solver_error and candidate_valid()
    raw_solver_objective=float(pulp.value(model.objective)) if valid else None
    used_start_fallback = bool(dc_results and any(r['used_start_fallback'] for r in dc_results))
    if (not valid and start_values is not None and settings.get('start_plan_fallback',False)):
        for v,n in start_values.items():
            v.varValue = n
        valid = candidate_valid()
        used_start_fallback = bool(valid)
    solver_incumbent_objective = raw_solver_objective
    projection_info = dict(applied=False, objective_change=0.0, shipments_unchanged=True)
    if valid and fulfillment_policy == 'earlier_service_priority':
        value = lambda v: float(pulp.value(v))
        fixed = {v: v.varValue for table in [x,y,w,trucks] for v in table.values()}
        before_objective = value(model.objective)
        changed_paths = 0
        for s in stores:
            for p in items:
                received = [arrivals('store',s,p,dt)+(case[p]*value(x[s,p,t-ls]) if t >= ls else 0)
                            for t,dt in enumerate(dates)]
                original_lost = [value(lost[s,p,t]) for t in range(horizon)]
                path = work_conserving_path(initial_store[s,p],received,
                                            [demand[s,p,t] for t in range(horizon)],
                                            [value(waste[s,p,t]) for t in range(horizon)])
                if any(abs(r['lost_sales']-original_lost[t]) > tol for t,r in enumerate(path)):
                    changed_paths += 1
                for t,r in enumerate(path):
                    inventory[s,p,t].varValue = r['closing']
                    lost[s,p,t].varValue = r['lost_sales']
                    service[s,p,t].varValue = max(safety[s,p,t]-r['morning'],0)
                    overflow[s,p,t].varValue = max(r['morning']-cap[s,p],0)
                    if life[p] is not None:
                        waste[s,p,t].varValue = r['waste']
                zs[s,p].varValue = max(terminal_store_days*mean[s,p]-path[-1]['closing'],0)
        after_objective = value(model.objective)
        if not candidate_valid() or any(v.varValue != original for v,original in fixed.items()):
            raise RuntimeError('Physical-sales correction failed feasibility or changed order decisions')
        if after_objective > before_objective+projection_tolerance:
            raise RuntimeError('Physical-sales correction violated the service-priority dominance bound')
        projection_info = dict(applied=True, changed_store_item_paths=changed_paths,
                               objective_before=before_objective, objective_after=after_objective,
                               objective_change=after_objective-before_objective,
                               shipments_unchanged=True)
    timed_out = 'time limit' in log.lower() or 'stopped on time' in log.lower()
    if used_start_fallback:
        status = 'TimeLimit-Feasible' if timed_out else 'Fallback-Feasible'
    elif valid:
        status = 'TimeLimit-Feasible' if timed_out else 'GapSatisfied-Feasible' if 'gap tolerance' in log.lower() else 'Optimal' if model.sol_status == pulp.LpSolutionOptimal else 'Feasible'
    else:
        status = 'SolverError' if solver_error else 'TimeLimit-NoIncumbent' if timed_out else 'Infeasible' if model.status == pulp.LpStatusInfeasible else 'NoValidatedIncumbent'
    gap_report = _solver_gap(log, status, float(pulp.value(model.objective)) if valid else None, tol)
    if dc_results is not None:
        bounds_available=all(r['best_bound'] is not None for r in dc_results)
        bound=sum(r['best_bound'] for r in dc_results) if bounds_available else None
        final_objective=float(pulp.value(model.objective)) if valid else None
        combined_gap=(max(final_objective-bound,0)/max(abs(final_objective),tol)
                      if final_objective is not None and bound is not None else None)
        gap_report=dict(best_bound=bound,mip_gap=combined_gap,cbc_reported_gap=None,
                        mip_gap_source='sum_of_independent_dc_bounds' if bounds_available else 'unavailable')
    stats = dict(n_vars=len(variables), n_int=sum(v.isInteger() for v in variables),
                 n_constraints=len(constraints), build_seconds=build_seconds,
                 solve_seconds=solve_seconds, status=status,
                 **gap_report,
                 incumbent_validated=bool(valid), effective_closed_forecast_rows=int(closure_mask.sum()),
                 nonzero_closed_forecast_rows=changed_closed_rows,
                 solver_error=solver_error, solver_log=log,
                 solver_incumbent_objective=solver_incumbent_objective,
                 verified_start_source=start_source,verified_start_objective=start_objective,
                 replenishment_start_issues=replenishment_start_issues,
                 used_start_fallback=used_start_fallback,
                 dc_solves=[] if dc_results is None else [{k:v for k,v in r.items() if k!='log'} for r in dc_results],
                 decomposition='exact_fixed_home_dc' if dc_results is not None else None,
                 physical_projection=projection_info, fulfillment_policy=fulfillment_policy,
                 warehouse_reserve_policy=reserve_policy,
                 approximation=('Work-conserving deterministic forecast sales with explicit early-service priority; exact aging is not implemented.'
                                if fulfillment_policy == 'earlier_service_priority' else
                                'Future lost sales are continuous planning allocations; exact aging is not implemented.'))
    print(json.dumps({k: stats[k] for k in ['n_vars', 'n_int', 'n_constraints', 'build_seconds',
                                          'solve_seconds', 'status', 'mip_gap']}))
    if not valid:
        # No executable rows are exported without a validated integer incumbent.
        return Plan(pd.DataFrame(columns=ORDER_COLUMNS), pd.DataFrame(columns=DC_COLUMNS),
                    pd.DataFrame(columns=DELIVERY_COLUMNS), math.nan, status, model, constraints, stats)
    val = lambda v: float(pulp.value(v))
    def firm_value(variable, t):
        value = val(variable)
        return float(round(value)) if t == 0 else max(value, 0)
    orders = pd.DataFrame([dict(date=dates[t], store_id=s, item_id=p,
                               cases=firm_value(v, t), units=case[p]*firm_value(v, t))
                           for (s, p, t), v in x.items()], columns=ORDER_COLUMNS)
    dc_orders = pd.DataFrame([dict(date=dates[t], dc_id=d, item_id=p,
                                  pallets=firm_value(v, t), units=case[p]*pallet_cases[p]*firm_value(v, t))
                              for (d, p, t), v in w.items()], columns=DC_COLUMNS)
    deliveries = pd.DataFrame([dict(date=dates[t], store_id=s, dc_id=homes[s],
                                    cases_total=sum(firm_value(x[s, p, t], t) for p in items),
                                    trucks=firm_value(v, t))
                                for (s, t), v in trucks.items()], columns=DELIVERY_COLUMNS)
    stats.update(
        store_inventory=[dict(date=dates[t].isoformat(), store_id=s, item_id=p, units=val(v))
                         for (s, p, t), v in inventory.items()],
        dc_inventory=[dict(date=dates[t].isoformat(), dc_id=d, item_id=p, units=val(v))
                      for (d, p, t), v in dc_inventory.items()],
        terminal_store_shortfall={s+'|'+p: val(v) for (s, p), v in zs.items()},
        terminal_dc_shortfall={d+'|'+p: val(v) for (d, p), v in zd.items()},
        overflow_unit_mornings=sum(val(v) for v in overflow.values()),
        waste_units_total=sum(val(v) for v in waste.values()),
        lost_sales_total=sum(val(v) for v in lost.values()),
        objective_breakdown={k: val(v) for k, v in costs.items()},
        dc_protection=[dict(date=dates[t].isoformat(),dc_id=d,item_id=p,
                            inventory_position=val(position),target_units=target,
                            shortfall_units=val(zcover[d,p,t]))
                       for (d,p,t),(position,target) in protection_expressions.items()],
        store_flows=[dict(date=dates[t].isoformat(), store_id=s, item_id=p,
                          demand=demand[s, p, t], safety_target=safety[s, p, t],
                          lost_sales=val(lost[s, p, t]), waste=val(waste[s, p, t]),
                          overflow=val(overflow[s, p, t]),
                          service_shortfall=val(service[s, p, t]),
                          shelf_capacity=cap[s, p])
                     for s in stores for p in items for t in range(horizon)],
    )
    # Diagnostic for the declared future-sales relaxation: positive planned lost
    # sales can coexist with saleable stock, although actual day 1 is fixed.
    rationing = [dict(date=dates[t].isoformat(), store_id=s, item_id=p,
                      withheld_units=min(val(lost[s, p, t]), val(v)+val(waste[s, p, t])))
                 for (s, p, t), v in inventory.items() if t > 0 and
                 min(val(lost[s, p, t]), val(v)+val(waste[s, p, t])) > tol]
    stats['future_sales_rationing'] = rationing
    stats['future_sales_rationing_units'] = sum(r['withheld_units'] for r in rationing)
    return Plan(orders, dc_orders, deliveries, val(model.objective), status, model, constraints, stats)
