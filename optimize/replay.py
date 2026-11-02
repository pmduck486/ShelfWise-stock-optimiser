"""L2 integration validation: execute firm decisions against observed sales.

This is a deterministic core validation runner, not L4's baseline comparison.
Disposal uses the approved forecast-cover approximation, not exact age cohorts.
"""

from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .model import State
from .policy import OptimizerPolicy, PIPELINE_COLUMNS


@dataclass
class DayResult:
    state: State
    ledger: pd.DataFrame
    metrics: dict


def _finite_nonnegative(values, label):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f'{label} must be finite and nonnegative')


def execute_day(state, decision, forecasts, observed, cfg):
    """Execute after deciding; observed sales are never provided to the policy.

    Receipts are available before dispatch/sales. Supplier purchases enter DC
    pipeline; DC dispatch is removed immediately and enters store pipeline.
    Cash charges purchase once, plus incremental disposal and actual holding.
    """
    today = pd.Timestamp(state.date).date()
    if decision.date != today:
        raise ValueError('Decision date differs from execution date')
    om = cfg['order_model']
    tol = om['solver']['feasibility_tolerance']
    params = om['parameters']['items']
    homes = cfg['network']['home_dc']
    closed = set(om['calendar']['closed_store_month_days'])
    store_closed = today.strftime('%m-%d') in closed
    pipeline = decision.effective_pipeline.copy()
    for col in ['arrival_date', 'order_date']:
        pipeline[col] = pd.to_datetime(pipeline[col]).dt.date
    if not pipeline.empty and (pipeline.arrival_date < today).any():
        raise ValueError('Unprocessed pipeline arrival before execution date')
    arrived = pipeline[pipeline.arrival_date == today]
    remaining = pipeline[pipeline.arrival_date > today].to_dict('records')

    stores = state.store_stock.copy().rename(columns={'units': 'opening_units'})
    dcs = state.dc_stock.copy().rename(columns={'units': 'opening_units'})
    obs = observed[['store_id', 'item_id', 'units']].rename(columns={'units': 'recorded_demand'})
    if obs.duplicated(['store_id', 'item_id']).any():
        raise ValueError('Duplicate observed-sales key')
    if set(obs.set_index(['store_id','item_id']).index) != set(stores.set_index(['store_id','item_id']).index):
        raise ValueError('Observed sales must cover the complete store-item grid')
    _finite_nonnegative(obs.recorded_demand, 'Observed sales')
    stores = stores.merge(obs, on=['store_id', 'item_id'], validate='one_to_one')

    for frame, kind, location in [(stores, 'store', 'store_id'), (dcs, 'dc', 'dc_id')]:
        receipts = arrived[arrived.location_type == kind].groupby(['location_id', 'item_id']).units.sum()
        frame['receipts'] = [float(receipts.get((getattr(r,location),r.item_id),0)) for r in frame.itertuples()]
        frame['morning_units'] = frame.opening_units + frame.receipts
    if store_closed and stores.receipts.sum() > tol:
        raise ValueError('Store receipt on closure')

    orders = decision.store_orders.copy()
    purchases = decision.dc_orders.copy()
    deliveries = decision.deliveries.copy()
    for frame in [orders, purchases, deliveries]:
        if set(pd.to_datetime(frame.date).dt.date) != {today}:
            raise ValueError('Only firm-day rows may be executed')
    for frame, location, count, pack in [(orders,'store_id','cases',lambda p: params[p]['units_per_case']),
                                        (purchases,'dc_id','pallets',lambda p: params[p]['units_per_case']*params[p]['cases_per_pallet'])]:
        if frame.duplicated([location,'item_id']).any():
            raise ValueError('Duplicate firm order')
        _finite_nonnegative(frame[count], 'Firm pack counts')
        rounded = np.rint(frame[count].to_numpy(dtype=float))
        if np.max(np.abs(rounded-frame[count].to_numpy(dtype=float)),initial=0)>tol:
            raise ValueError('Fractional firm order')
        exact = np.array([pack(p) for p in frame.item_id])*rounded
        if np.max(np.abs(exact-frame.units.to_numpy(dtype=float)),initial=0)>tol:
            raise ValueError('Firm units do not equal whole packs')
        frame[count] = rounded
        frame['units'] = exact
        minimum=om['store_min_order_cases'] if count=='cases' else om['dc_min_order_pallets']
        if ((frame[count]>0)&(frame[count]<minimum)).any():
            raise ValueError('Firm order below minimum pack count')
    if set(orders.set_index(['store_id','item_id']).index) != set(stores.set_index(['store_id','item_id']).index):
        raise ValueError('Incomplete firm store grid')
    if set(purchases.set_index(['dc_id','item_id']).index) != set(dcs.set_index(['dc_id','item_id']).index):
        raise ValueError('Incomplete firm supplier grid')
    orders['dc_id'] = orders.store_id.map(homes)
    dispatched = orders.groupby(['dc_id','item_id']).units.sum()
    dcs['dispatch_units'] = [float(dispatched.get((r.dc_id,r.item_id),0)) for r in dcs.itertuples()]
    dcs['closing_units'] = dcs.morning_units-dcs.dispatch_units
    if dcs.closing_units.min() < -tol:
        raise ValueError('Firm dispatch exceeds physically available DC stock')
    dcs['closing_units'] = dcs.closing_units.clip(lower=0)
    if len(deliveries)!=stores.store_id.nunique() or deliveries.store_id.duplicated().any():
        raise ValueError('Incomplete firm delivery grid')
    if set(deliveries.store_id)!=set(stores.store_id):
        raise ValueError('Unknown delivery store')
    _finite_nonnegative(deliveries.trucks,'Trucks')
    for r in deliveries.itertuples():
        cases = orders.loc[orders.store_id==r.store_id,'cases'].sum()
        if r.dc_id!=homes[r.store_id] or abs(cases-r.cases_total)>tol or abs(r.trucks-round(r.trucks))>tol or cases>r.trucks*om['truck_capacity_cases']+tol:
            raise ValueError('Invalid firm truck load')
    for d, frame in purchases.groupby('dc_id'):
        spend=sum(r.units*params[r.item_id]['buy_cost'] for r in frame.itertuples())
        if spend>om['parameters']['dc_budgets'][d]+tol:
            raise ValueError('Firm purchases exceed DC budget')

    for frame, kind, location, lead in [(orders,'store','store_id',om['lead_time_store_days']),
                                      (purchases,'dc','dc_id',om['lead_time_supplier_days'])]:
        arrival=today+timedelta(days=lead)
        calendar=om['calendar']['closed_store_month_days' if kind=='store' else 'closed_dc_month_days']
        if arrival.strftime('%m-%d') in calendar and frame.units.sum()>tol:
            raise ValueError('New firm order would arrive on closure')
        for r in frame[frame.units>tol].itertuples():
            remaining.append(dict(order_date=today,arrival_date=arrival,location_type=kind,
                                  location_id=getattr(r,location),item_id=r.item_id,units=float(r.units)))

    f=forecasts.copy()
    f['date']=pd.to_datetime(f.date).dt.date
    f=f[pd.to_datetime(f.origin).dt.date==today].set_index(['store_id','item_id','date'])
    horizon=om['horizon_days']
    def cover(s,p,life):
        horizon_values=[0.0 if (today+timedelta(days=t)).strftime('%m-%d') in closed
                        else float(f.loc[(s,p,today+timedelta(days=t)),'q50']) for t in range(horizon)]
        mean=sum(horizon_values)/horizon
        return sum((horizon_values[t] if t<horizon else mean)
                   for t in range(1,life) if (today+timedelta(days=t)).strftime('%m-%d') not in closed)
    stores['actual_demand']=0.0 if store_closed else stores.recorded_demand
    stores['served_units']=np.minimum(stores.morning_units,stores.actual_demand)
    stores['lost_units']=stores.actual_demand-stores.served_units
    leftover=stores.morning_units-stores.served_units
    covers=[cover(r.store_id,r.item_id,params[r.item_id]['shelf_life_days'])
            if params[r.item_id]['shelf_life_days'] is not None else np.inf for r in stores.itertuples()]
    stores['waste_proxy_units']=np.maximum(leftover-np.asarray(covers),0)
    stores['closing_units']=leftover-stores.waste_proxy_units
    for frame in [stores,dcs]:
        frame['holding_cost']=[r.closing_units*params[r.item_id]['holding_cost_per_unit_day'] for r in frame.itertuples()]
    stores['disposal_cash']=[r.waste_proxy_units*params[r.item_id]['buy_cost']*om['waste_disposal_share'] for r in stores.itertuples()]
    stores['lost_sales_penalty']=[r.lost_units*params[r.item_id]['price']*om['lost_sale_penalty_share'] for r in stores.itertuples()]
    stores['overflow_units']=[max(r.morning_units-max(
        (om['shelf_days_perishable'] if params[r.item_id]['shelf_life_days'] is not None else om['shelf_days'])*om['parameters']['store_averages'][r.store_id][r.item_id],
        om['shelf_min_cases']*params[r.item_id]['units_per_case']),0) for r in stores.itertuples()]
    if om['backroom_capacity_units'] is not None and stores.overflow_units.max()>om['backroom_capacity_units']+tol:
        raise ValueError('Observed replay requires more than configured backroom capacity')

    purchase_cash=sum(r.units*params[r.item_id]['buy_cost'] for r in purchases.itertuples())
    truck_cash=float(deliveries.trucks.sum()*om['truck_trip_cost'])
    order_cash=float((orders.cases>tol).sum()*om['order_line_cost'])
    holding_cash=float(stores.holding_cost.sum()+dcs.holding_cost.sum())
    disposal_cash=float(stores.disposal_cash.sum())
    cash=purchase_cash+truck_cash+order_cash+holding_cash+disposal_cash
    # Chain conservation counts both in-transit stages, including arrivals after the replay.
    opening_total=float(state.store_stock.units.sum()+state.dc_stock.units.sum()+pipeline.units.sum())
    closing_total=float(stores.closing_units.sum()+dcs.closing_units.sum()+sum(r['units'] for r in remaining))
    residual=opening_total+float(purchases.units.sum())-float(stores.served_units.sum()+stores.waste_proxy_units.sum())-closing_total
    if abs(residual)>tol:
        raise ValueError(f'Physical chain conservation failed: {residual}')
    metrics=dict(date=today.isoformat(),closed=store_closed,actual_demand=float(stores.actual_demand.sum()),
                 ignored_closed_day_recorded_units=float(stores.recorded_demand.sum()) if store_closed else 0.0,
                 served_units=float(stores.served_units.sum()),lost_units=float(stores.lost_units.sum()),
                 stockout_store_items=int((stores.lost_units>tol).sum()),
                 waste_proxy_units=float(stores.waste_proxy_units.sum()),overflow_unit_mornings=float(stores.overflow_units.sum()),
                 store_receipts=float(stores.receipts.sum()),dc_receipts=float(dcs.receipts.sum()),
                 store_dispatch_units=float(orders.units.sum()),supplier_units=float(purchases.units.sum()),
                 purchase_cash=purchase_cash,truck_cash=truck_cash,order_cash=order_cash,
                 holding_cash=holding_cash,disposal_cash=disposal_cash,cash_cost=cash,
                 lost_sales_penalty=float(stores.lost_sales_penalty.sum()),
                 operating_score=cash+float(stores.lost_sales_penalty.sum()),
                 ending_stock_and_pipeline_units=closing_total,conservation_residual_units=residual,
                 pipeline_adjustments=len(decision.pipeline_adjustments))
    stores['location_type']='store';stores['location_id']=stores.store_id
    dcs['location_type']='dc';dcs['location_id']=dcs.dc_id
    ledger=pd.concat([stores,dcs],ignore_index=True).fillna(0)
    ledger.insert(0,'date',today.isoformat())
    next_state=State(today+timedelta(days=1),
                     stores[['store_id','item_id','closing_units']].rename(columns={'closing_units':'units'}),
                     dcs[['dc_id','item_id','closing_units']].rename(columns={'closing_units':'units'}),
                     pd.DataFrame(remaining,columns=PIPELINE_COLUMNS))
    return DayResult(next_state,ledger,metrics)


def run_replay(state, forecasts, sales, cfg, *, end_date, output_dir, policy=None):
    """Strict calendar replay, with append-only daily evidence and restart state."""
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'daily_metrics.csv').exists():
        raise ValueError('Replay output already exists; choose a fresh directory')
    first=pd.Timestamp(state.date).date();end=pd.Timestamp(end_date).date()
    if end<first:
        raise ValueError('Replay end precedes start')
    windows=cfg.get('data',{}).get('windows')
    if windows and (first<pd.Timestamp(windows['test_start']).date() or end>pd.Timestamp(windows['test_end']).date()):
        raise ValueError('Validation replay must stay inside test; holdout remains sealed')
    forecasts=forecasts.copy();sales=sales.copy()
    forecasts['origin']=pd.to_datetime(forecasts.origin).dt.date
    forecasts['date']=pd.to_datetime(forecasts.date).dt.date
    sales['date']=pd.to_datetime(sales.date).dt.date
    days=pd.date_range(first,end).date
    store_ids=list(state.store_stock.store_id.unique());items=list(state.store_stock.item_id.unique())
    forecasts=forecasts[forecasts.origin.isin(days)&forecasts.store_id.isin(store_ids)&forecasts.item_id.isin(items)]
    sales=sales[sales.date.isin(days)&sales.store_id.isin(store_ids)&sales.item_id.isin(items)]
    expected={(s,p) for s in store_ids for p in items}
    if sales.duplicated(['date','store_id','item_id']).any() or forecasts.duplicated(['origin','date','store_id','item_id']).any():
        raise ValueError('Duplicate replay source key')
    fc_groups={day:f for day,f in forecasts.groupby('origin')}
    sales_groups={day:f for day,f in sales.groupby('date')}
    for day in days:
        if day not in sales_groups or set(sales_groups[day].set_index(['store_id','item_id']).index)!=expected:
            raise ValueError(f'Incomplete observed-sales grid for {day}')
        expected_fc={(s,p,day+timedelta(days=t)) for s,p in expected for t in range(cfg['order_model']['horizon_days'])}
        if day not in fc_groups or set(fc_groups[day].set_index(['store_id','item_id','date']).index)!=expected_fc:
            raise ValueError(f'Incomplete forecast horizon for {day}')
    policy=policy or OptimizerPolicy();rows=[];started=perf_counter()
    for day in days:
        daily_started=perf_counter()
        # Only this origin reaches decide; actual sales are consumed afterwards.
        decision=policy.decide(state,fc_groups[day],cfg)
        result=execute_day(state,decision,fc_groups[day],sales_groups[day],cfg)
        row=result.metrics
        stats=decision.plan.stats
        row.update(status=decision.plan.status,objective=decision.plan.objective,
                   mip_gap=stats['mip_gap'],solve_seconds=stats['solve_seconds'],build_seconds=stats['build_seconds'],
                   used_start_fallback=stats['used_start_fallback'],
                   planned_future_withholding=stats['future_sales_rationing_units'],
                   n_vars=stats['n_vars'],n_int=stats['n_int'],
                   plan_audit_passed=decision.audit['passed'],
                   physical_sales_checks=decision.audit['physical_sales_checks'],
                   elapsed_seconds=perf_counter()-daily_started)
        rows.append(row)
        pd.DataFrame([row]).to_csv(output/'daily_metrics.csv',mode='a',header=len(rows)==1,index=False)
        result.ledger.to_csv(output/'inventory_ledger.csv',mode='a',header=len(rows)==1,index=False)
        for frame,name in [(decision.store_orders,'order_plan.csv'),(decision.dc_orders,'dc_orders.csv'),
                           (decision.deliveries,'deliveries.csv'),(decision.pipeline_adjustments,'pipeline_adjustments.csv')]:
            frame.to_csv(output/name,mode='a',header=len(rows)==1,index=False)
        logs=output/'solver_logs';logs.mkdir(exist_ok=True)
        (logs/(day.isoformat()+'.log')).write_text(stats['solver_log'])
        state=result.state
        state.store_stock.to_csv(output/'final_store_stock.csv',index=False)
        state.dc_stock.to_csv(output/'final_dc_stock.csv',index=False)
        state.pipeline.to_csv(output/'final_pipeline.csv',index=False)
        (output/'progress.json').write_text(json.dumps(dict(last_completed_day=day.isoformat(),next_state_date=state.date.isoformat(),completed_days=len(rows)),indent=2)+'\n')
        print(f'{day}: {decision.plan.status}, gap={stats["mip_gap"]}, lost={row["lost_units"]:.0f}, supplier={row["supplier_units"]:.0f}, elapsed={row["elapsed_seconds"]:.2f}s',flush=True)
    frame=pd.DataFrame(rows);open_days=frame[~frame.closed]
    fill=lambda f:float(f.served_units.sum()/f.actual_demand.sum()) if f.actual_demand.sum()>0 else None
    daily_fills=1-open_days.lost_units/open_days.actual_demand.replace(0,np.nan)
    finite_gaps=frame.mip_gap.dropna()
    summary=dict(start=first.isoformat(),end=end.isoformat(),days=len(frame),open_days=len(open_days),
                 actual_demand=float(open_days.actual_demand.sum()),served_units=float(open_days.served_units.sum()),
                 lost_units=float(open_days.lost_units.sum()),unit_fill_rate=fill(open_days),
                 mean_open_day_fill_rate=None if daily_fills.dropna().empty else float(daily_fills.mean()),
                 stockout_store_item_days=int(open_days.stockout_store_items.sum()),
                 waste_proxy_units=float(frame.waste_proxy_units.sum()),cash_cost=float(frame.cash_cost.sum()),
                 lost_sales_penalty=float(frame.lost_sales_penalty.sum()),operating_score=float(frame.operating_score.sum()),
                 purchase_cash=float(frame.purchase_cash.sum()),disposal_cash=float(frame.disposal_cash.sum()),
                 holding_cash=float(frame.holding_cash.sum()),truck_cash=float(frame.truck_cash.sum()),order_cash=float(frame.order_cash.sum()),
                 supplier_units=float(frame.supplier_units.sum()),
                 ending_stock_and_pipeline_units=float(frame.ending_stock_and_pipeline_units.iloc[-1]),
                 overflow_unit_mornings=float(frame.overflow_unit_mornings.sum()),
                 fallback_days=int(frame.used_start_fallback.sum()),status_counts=frame.status.value_counts().to_dict(),
                 all_plan_audits_passed=bool(frame.plan_audit_passed.all()),
                 physical_sales_checks=int(frame.physical_sales_checks.sum()),
                 planned_future_withholding=float(frame.planned_future_withholding.sum()),
                 max_conservation_residual_units=float(frame.conservation_residual_units.abs().max()),
                 max_gap=None if finite_gaps.empty else float(finite_gaps.max()),
                 median_gap=None if finite_gaps.empty else float(finite_gaps.median()),
                 gaps_unavailable=int(frame.mip_gap.isna().sum()),
                 median_daily_seconds=float(frame.elapsed_seconds.median()),max_daily_seconds=float(frame.elapsed_seconds.max()),
                 elapsed_seconds=perf_counter()-started,
                 solver_settings=cfg['order_model']['solver'],
                 scope='Observed M5 sales treated as demand; supplied forecasts; optimizer only; no causal/baseline or holdout claim.',
                 disposal='Approved forecast-cover approximation, not exact expiry/age tracking.',
                 cash_accounting='Purchase once plus 5%-configured disposal; excludes safety, reserve, overflow and early-service preference penalties.',
                 closure_scoring='Closed dates excluded from service means; all cash, holding and disposal retained.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary,state
