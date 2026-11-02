"""Reconstruct an exported plan from inputs; never rely on solver status alone."""

from datetime import timedelta
import math

import pandas as pd


def audit_plan(state, fc, cfg, plan):
    om = cfg['order_model']
    params = om['parameters']
    tol = om['solver']['feasibility_tolerance']
    origin = pd.Timestamp(state.date).date()
    dates = [origin+timedelta(days=t) for t in range(om['horizon_days'])]
    stores = state.store_stock.store_id.drop_duplicates().tolist()
    dcs = state.dc_stock.dc_id.drop_duplicates().tolist()
    items = list(params['items'])
    homes = cfg['network']['home_dc']
    closed = set(om['calendar']['closed_store_month_days'])
    is_closed = lambda dt: dt.strftime('%m-%d') in closed
    checks = dict(store_balance_checks=0, dc_balance_checks=0, truck_checks=0,
                  terminal_target_checks=0, dc_protection_checks=0, physical_sales_checks=0,
                  budget_checks=0, max_balance_residual_units=0.0)

    def require(condition, message):
        if not condition:
            raise ValueError('Plan audit failed: '+message)

    def close(actual, expected, message):
        error = abs(float(actual)-float(expected))
        require(math.isfinite(error) and error <= tol, message+f' ({actual} vs {expected})')
        return error

    require(plan.stats.get('incumbent_validated', False), 'no validated incumbent')
    orders, dc_orders, deliveries = plan.orders.copy(), plan.dc_orders.copy(), plan.deliveries.copy()
    for frame, columns in [(orders, ['date','store_id','item_id','cases','units']),
                            (dc_orders, ['date','dc_id','item_id','pallets','units']),
                            (deliveries, ['date','store_id','dc_id','cases_total','trucks'])]:
        require(list(frame) == columns, 'output columns differ from contract')
        frame['date'] = pd.to_datetime(frame.date).dt.date
    for frame, columns, expected in [
        (orders, ['store_id','item_id','date'], {(s,p,d) for s in stores for p in items for d in dates}),
        (dc_orders, ['dc_id','item_id','date'], {(d,p,t) for d in dcs for p in items for t in dates}),
        (deliveries, ['store_id','date'], {(s,t) for s in stores for t in dates})]:
        require(not frame.duplicated(columns).any(), 'duplicate output key')
        require(set(frame[columns].itertuples(index=False, name=None)) == expected, 'incomplete output grid')
    order = orders.set_index(['store_id','item_id','date'])
    purchase = dc_orders.set_index(['dc_id','item_id','date'])
    shipment = deliveries.set_index(['store_id','date'])
    for r in orders.itertuples(index=False):
        info = params['items'][r.item_id]
        require(r.cases >= -tol and r.units >= -tol, 'negative store order')
        close(r.units, r.cases*info['units_per_case'], 'carton units')
        if r.date == origin:
            close(r.cases, round(r.cases), 'firm cartons must be integers')
            require(r.cases <= tol or r.cases >= om['store_min_order_cases']-tol, 'firm MOQ')
        if is_closed(r.date+timedelta(days=om['lead_time_store_days'])):
            require(r.units <= tol, 'arrival on store closure')
    for r in dc_orders.itertuples(index=False):
        info = params['items'][r.item_id]
        require(r.pallets >= -tol and r.units >= -tol, 'negative supplier order')
        close(r.units, r.pallets*info['units_per_case']*info['cases_per_pallet'], 'pallet units')
        if r.date == origin:
            close(r.pallets, round(r.pallets), 'firm pallets must be integers')
    for r in deliveries.itertuples(index=False):
        require(r.dc_id == homes[r.store_id], 'wrong home DC')
        cartons = float(orders[(orders.store_id == r.store_id) & (orders.date == r.date)].cases.sum())
        close(r.cases_total, cartons, 'delivery carton total')
        require(r.trucks >= -tol and cartons <= om['truck_capacity_cases']*r.trucks+tol, 'truck capacity')
        if r.date == origin:
            close(r.trucks, round(r.trucks), 'firm trucks must be integers')
        checks['truck_checks'] += 1

    forecasts = fc.copy()
    forecasts['origin'] = pd.to_datetime(forecasts.origin).dt.date
    forecasts['date'] = pd.to_datetime(forecasts.date).dt.date
    forecasts = forecasts[(forecasts.origin == origin) & forecasts.date.isin(dates)].set_index(['store_id','item_id','date'])
    stocks = pd.DataFrame(plan.stats['store_inventory'])
    warehouse = pd.DataFrame(plan.stats['dc_inventory'])
    flows = pd.DataFrame(plan.stats['store_flows'])
    for f in [stocks, warehouse, flows]:
        f['date'] = pd.to_datetime(f.date).dt.date
    for frame, cols, expected in [
        (stocks, ['store_id','item_id','date'], set(order.index)),
        (flows, ['store_id','item_id','date'], set(order.index)),
        (warehouse, ['dc_id','item_id','date'], set(purchase.index))]:
        require(not frame.duplicated(cols).any(), 'duplicate diagnostic key')
        require(set(frame[cols].itertuples(index=False,name=None)) == expected, 'incomplete inventory/flow diagnostics')
    stock = stocks.set_index(['store_id','item_id','date']).units
    dc_stock = warehouse.set_index(['dc_id','item_id','date']).units
    flow = {(r.store_id,r.item_id,r.date):r for r in flows.itertuples(index=False)}
    initial = state.store_stock.set_index(['store_id','item_id']).units
    dc_initial = state.dc_stock.set_index(['dc_id','item_id']).units
    pipe = state.pipeline.copy()
    if not pipe.empty:
        pipe['arrival_date'] = pd.to_datetime(pipe.arrival_date).dt.date
        pipe['location_type'] = pipe.location_type.str.lower()
    pipeline_by_item = {}
    for row in pipe.itertuples(index=False):
        pipeline_by_item.setdefault((row.location_type,row.location_id,row.item_id),[]).append((row.arrival_date,float(row.units)))
    def pending(kind, location, item, dt=None, after=None):
        return sum(units for arrival,units in pipeline_by_item.get((kind,location,item),[])
                   if (dt is None or arrival==dt) and (after is None or arrival>after))
    effective = lambda s,p,dt,column: 0.0 if is_closed(dt) else float(forecasts.at[(s,p,dt),column])
    means = {(s,p): sum(effective(s,p,dt,'q50') for dt in dates)/len(dates) for s in stores for p in items}
    initial_overflow, peak_overflow = 0.0, 0.0
    for s in stores:
        for p in items:
            info = params['items'][p]
            qcol = {0.5:'q50',0.8:'q80',0.9:'q90',0.95:'q95'}[info['service_level']]
            shelf_days = om['shelf_days_perishable'] if info['shelf_life_days'] is not None else om['shelf_days']
            capacity = max(shelf_days*params['store_averages'][s][p], om['shelf_min_cases']*info['units_per_case'])
            previous = float(initial.at[(s,p)])
            for t, dt in enumerate(dates):
                r = flow[s,p,dt]
                arrival = pending('store',s,p,dt=dt)
                if t >= om['lead_time_store_days']:
                    arrival += float(order.at[(s,p,dates[t-om['lead_time_store_days']]),'units'])
                morning = previous+arrival
                demand = effective(s,p,dt,'q50')
                safety = effective(s,p,dt,qcol)
                close(r.demand,demand,'effective demand')
                close(r.safety_target,safety,'effective safety line')
                close(r.shelf_capacity,capacity,'store-specific shelf capacity')
                require(-tol <= r.lost_sales <= demand+tol, 'lost-sales bounds')
                require(r.waste >= -tol and r.overflow >= -tol and r.service_shortfall >= -tol, 'negative disposal/overflow/shortfall')
                require(morning <= capacity+r.overflow+tol, 'shelf plus backroom')
                if om['backroom_capacity_units'] is not None:
                    require(r.overflow <= om['backroom_capacity_units']+tol, 'backroom limit')
                require(morning+r.service_shortfall >= safety-tol,'soft safety line')
                require(stock.at[(s,p,dt)] >= -tol,'negative store inventory')
                residual = close(stock.at[(s,p,dt)],morning-demand+r.lost_sales-r.waste,'store stock flow')
                checks['max_balance_residual_units'] = max(checks['max_balance_residual_units'],residual)
                if t == 0 or om.get('fulfillment_policy') == 'earlier_service_priority':
                    close(r.lost_sales,max(demand-morning,0),'physical forecast sales')
                    checks['physical_sales_checks'] += 1
                if t == 0:
                    initial_overflow += r.overflow
                if info['shelf_life_days'] is None:
                    close(r.waste,0,'nonperishable waste proxy')
                else:
                    cover = 0.0
                    for step in range(1,info['shelf_life_days']):
                        target = dt+timedelta(days=step)
                        if not is_closed(target):
                            cover += effective(s,p,target,'q50') if target in dates else means[s,p]
                    require(r.waste >= morning-demand+r.lost_sales-cover-tol,'expected disposal bound')
                peak_overflow = max(peak_overflow,float(r.overflow))
                previous = float(stock.at[(s,p,dt)])
                checks['store_balance_checks'] += 1
            shortfall = plan.stats['terminal_store_shortfall'][s+'|'+p]
            require(shortfall >= -tol and previous+shortfall >= om['terminal_store_days']*means[s,p]-tol,'soft store terminal target')
            checks['terminal_target_checks'] += 1
    for d in dcs:
        for p in items:
            previous = float(dc_initial.at[(d,p)])
            for t,dt in enumerate(dates):
                arrival = pending('dc',d,p,dt=dt)
                if t >= om['lead_time_supplier_days']:
                    arrival += float(purchase.at[(d,p,dates[t-om['lead_time_supplier_days']]),'units'])
                sent = sum(float(order.at[(s,p,dt),'units']) for s in stores if homes[s] == d)
                require(sent <= previous+arrival+tol,'shipments exceed available DC supply')
                require(dc_stock.at[(d,p,dt)] >= -tol,'negative DC inventory')
                residual = close(dc_stock.at[(d,p,dt)],previous+arrival-sent,'DC stock flow')
                checks['max_balance_residual_units'] = max(checks['max_balance_residual_units'],residual)
                previous = float(dc_stock.at[(d,p,dt)])
                checks['dc_balance_checks'] += 1
            position = previous+pending('dc',d,p,after=dates[-1])
            position += sum(float(purchase.at[(d,p,dt),'units']) for dt in dates
                            if dt+timedelta(days=om['lead_time_supplier_days']) > dates[-1])
            target = om['terminal_dc_days']*sum(means[s,p] for s in stores if homes[s] == d)
            shortfall = plan.stats['terminal_dc_shortfall'][d+'|'+p]
            require(shortfall >= -tol and position+shortfall >= target-tol,'soft DC terminal target')
            checks['terminal_target_checks'] += 1
        spend = sum(float(purchase.at[(d,p,dt),'units'])*params['items'][p]['buy_cost'] for p in items for dt in dates)
        require(spend <= params['dc_budgets'][d]+tol,'forward purchase budget')
        checks['budget_checks'] += 1
    if om.get('warehouse_reserve_policy') == 'daily_inventory_position':
        protected = pd.DataFrame(plan.stats['dc_protection'])
        protected['date'] = pd.to_datetime(protected.date).dt.date
        cols = ['dc_id','item_id','date']
        expected = {(d,p,dt) for d in dcs for p in items for dt in dates[:-1]}
        require(not protected.duplicated(cols).any() and
                set(protected[cols].itertuples(index=False,name=None)) == expected,
                'incomplete daily DC protection grid')
        protection = {(r.dc_id,r.item_id,r.date):r for r in protected.itertuples(index=False)}
        for d,p,dt in expected:
            r = protection[d,p,dt]
            lead = om['lead_time_supplier_days']
            existing = sum(units for arrival,units in pipeline_by_item.get(('dc',d,p),[])
                           if dt<arrival<=dt+timedelta(days=lead))
            position = float(dc_stock.at[(d,p,dt)])+existing
            position += sum(float(purchase.at[(d,p,ordered),'units']) for ordered in dates
                            if ordered <= dt and ordered+timedelta(days=lead) > dt)
            target = 0.0
            for offset in range(1,lead+1):
                target_date = dt+timedelta(days=offset)
                if not is_closed(target_date):
                    target += sum(effective(s,p,target_date,'q50') if target_date in dates else means[s,p]
                                  for s in stores if homes[s] == d)
            close(r.inventory_position,position,'DC protection counts only already-placed orders')
            close(r.target_units,target,'DC protection forecast window')
            require(r.shortfall_units >= -tol and position+r.shortfall_units >= target-tol,
                    'soft daily DC protection target')
            checks['dc_protection_checks'] += 1
    checks.update(passed=True, initial_overflow_units=float(initial_overflow),
                  largest_single_line_overflow_units=float(peak_overflow),
                  min_store_stock_units=float(stock.min()), min_dc_stock_units=float(dc_stock.min()),
                  scope='One origin and seven tentative dates; independent export reconstruction, not a realized replay.')
    return checks
