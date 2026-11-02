"""Independently reconstruct saved execution from orders, sales and inputs."""

from datetime import timedelta
import json
import math
from pathlib import Path

import pandas as pd


def audit_replay(initial_state, forecasts, sales, cfg, output_dir):
    root=Path(output_dir);om=cfg['order_model'];params=om['parameters']['items']
    tolerance=om['solver']['feasibility_tolerance']
    frame=pd.read_csv(root/'daily_metrics.csv')
    if frame.empty:
        raise ValueError('No executed days')
    days=pd.to_datetime(frame.date).dt.date.tolist()
    if days!=pd.date_range(initial_state.date,days[-1]).date.tolist():
        raise ValueError('Execution omitted or repeated a calendar day')
    tables={name:pd.read_csv(root/(name+'.csv')) for name in ['order_plan','dc_orders','deliveries','inventory_ledger']}
    for table in tables.values():table['date']=pd.to_datetime(table.date).dt.date
    store_stock={(r.store_id,r.item_id):float(r.units) for r in initial_state.store_stock.itertuples()}
    dc_stock={(r.dc_id,r.item_id):float(r.units) for r in initial_state.dc_stock.itertuples()}
    closed=set(om['calendar']['closed_store_month_days'])
    due={};received=0;max_residual=0.0

    def require(condition,message):
        if not condition:raise ValueError('Replay audit failed: '+message)
    def close(a,b,label):
        nonlocal max_residual
        difference=abs(float(a)-float(b))
        require(math.isfinite(difference) and difference<=tolerance,label+f': {a} != {b}')
        max_residual=max(max_residual,difference)
    def schedule(kind,loc,item,arrival,units,committed=False):
        calendar=set(om['calendar']['closed_store_month_days' if kind=='store' else 'closed_dc_month_days'])
        if committed:
            for _ in range(367):
                if arrival.strftime('%m-%d') not in calendar:break
                arrival+=timedelta(days=1)
            else:raise ValueError('No open commitment arrival date')
        require(arrival.strftime('%m-%d') not in calendar,'arrival on closure')
        key=(arrival,kind,loc,item);due[key]=due.get(key,0)+units
    for r in initial_state.pipeline.itertuples():
        schedule(r.location_type.lower(),r.location_id,r.item_id,pd.Timestamp(r.arrival_date).date(),float(r.units),True)
    forecasts=forecasts.copy();forecasts['origin']=pd.to_datetime(forecasts.origin).dt.date
    forecasts['date']=pd.to_datetime(forecasts.date).dt.date
    forecasts=forecasts.set_index(['origin','date','store_id','item_id'])
    sales=sales.copy();sales['date']=pd.to_datetime(sales.date).dt.date
    sales=sales.set_index(['date','store_id','item_id']).units
    counts=dict(days=len(days),store_balance_checks=0,dc_balance_checks=0,firm_pack_checks=0,
                truck_checks=0,cash_checks=0,conservation_checks=0)
    for day,daily in zip(days,frame.itertuples()):
        orders=tables['order_plan'].loc[tables['order_plan'].date==day]
        purchase=tables['dc_orders'].loc[tables['dc_orders'].date==day]
        trucks=tables['deliveries'].loc[tables['deliveries'].date==day]
        ledger=tables['inventory_ledger'].loc[tables['inventory_ledger'].date==day]
        require(not ledger.duplicated(['location_type','location_id','item_id']).any(),'duplicate ledger key')
        require(len(ledger)==len(store_stock)+len(dc_stock),'incomplete ledger')
        ledger=ledger.set_index(['location_type','location_id','item_id'])
        for table,loc,qty,keys in [(orders,'store_id','cases',set(store_stock)),(purchase,'dc_id','pallets',set(dc_stock))]:
            require(not table.duplicated([loc,'item_id']).any(),'duplicate executed order')
            require(set(zip(table[loc],table.item_id))==keys,'incomplete executed order grid')
            for r in table.itertuples():
                n=float(getattr(r,qty));pack=params[r.item_id]['units_per_case']
                if qty=='pallets':pack*=params[r.item_id]['cases_per_pallet']
                require(n>=0 and abs(n-round(n))<=tolerance,'fractional/negative executed pack')
                minimum=om['store_min_order_cases' if qty=='cases' else 'dc_min_order_pallets']
                require(n==0 or n>=minimum-tolerance,'executed MOQ')
                close(r.units,round(n)*pack,'pack conversion')
                if r.units>tolerance:
                    schedule('store' if qty=='cases' else 'dc',getattr(r,loc),r.item_id,
                             day+timedelta(days=om['lead_time_store_days' if qty=='cases' else 'lead_time_supplier_days']),float(r.units))
                counts['firm_pack_checks']+=1
        require(len(trucks)==len({s for s,p in store_stock}) and not trucks.store_id.duplicated().any(),'truck grid')
        for r in trucks.itertuples():
            require(r.dc_id==cfg['network']['home_dc'][r.store_id],'wrong home DC')
            cases=orders.loc[orders.store_id==r.store_id,'cases'].sum()
            close(cases,r.cases_total,'truck load')
            require(r.trucks>=0 and abs(r.trucks-round(r.trucks))<=tolerance and cases<=r.trucks*om['truck_capacity_cases']+tolerance,'truck capacity/integrality')
            counts['truck_checks']+=1
        totals=dict(actual=0.0,served=0.0,lost=0.0,waste=0.0,hold=0.0,disposal=0.0,lost_penalty=0.0,stockout=0,
                    store_receipts=0.0,dc_receipts=0.0,overflow=0.0,ignored=0.0)
        before=sum(store_stock.values())+sum(dc_stock.values())+sum(due.values())-float(orders.units.sum()+purchase.units.sum())
        # Store shipments in due were just added, so undo their addition as an internal transfer.
        for (store,item),opening in list(store_stock.items()):
            r=ledger.loc[('store',store,item)];receipt=due.pop((day,'store',store,item),0.0)
            close(r.opening_units,opening,'store opening');close(r.receipts,receipt,'store receipt')
            if day.strftime('%m-%d') in closed:require(receipt<=tolerance,'closed store receipt')
            actual=0.0 if day.strftime('%m-%d') in closed else float(sales.loc[(day,store,item)])
            available=opening+receipt;served=min(available,actual);lost=actual-served
            life=params[item]['shelf_life_days'];cover=math.inf
            if life is not None:
                values=[0.0 if (day+timedelta(days=t)).strftime('%m-%d') in closed else float(forecasts.loc[(day,day+timedelta(days=t),store,item),'q50']) for t in range(om['horizon_days'])]
                cover=sum(values[t] if t<len(values) else sum(values)/len(values) for t in range(1,life) if (day+timedelta(days=t)).strftime('%m-%d') not in closed)
            waste=max(available-served-cover,0);closing=available-served-waste
            for label,value in [('morning_units',available),('actual_demand',actual),('served_units',served),('lost_units',lost),('waste_proxy_units',waste),('closing_units',closing)]:close(r[label],value,label)
            store_stock[store,item]=closing
            totals['actual']+=actual;totals['served']+=served;totals['lost']+=lost;totals['waste']+=waste
            totals['stockout']+=int(lost>tolerance);totals['store_receipts']+=receipt
            totals['hold']+=closing*params[item]['holding_cost_per_unit_day']
            totals['disposal']+=waste*params[item]['buy_cost']*om['waste_disposal_share']
            totals['lost_penalty']+=lost*params[item]['price']*om['lost_sale_penalty_share']
            capacity=max((om['shelf_days_perishable'] if life is not None else om['shelf_days'])*om['parameters']['store_averages'][store][item],om['shelf_min_cases']*params[item]['units_per_case'])
            totals['overflow']+=max(available-capacity,0)
            if day.strftime('%m-%d') in closed:totals['ignored']+=float(sales.loc[(day,store,item)])
            counts['store_balance_checks']+=1
        ordered_by_dc=orders.assign(dc_id=orders.store_id.map(cfg['network']['home_dc'])).groupby(['dc_id','item_id']).units.sum()
        for (dc,item),opening in list(dc_stock.items()):
            r=ledger.loc[('dc',dc,item)];receipt=due.pop((day,'dc',dc,item),0.0)
            dispatched=float(ordered_by_dc.get((dc,item),0));closing=opening+receipt-dispatched
            require(closing>=-tolerance,'negative DC stock')
            for label,value in [('opening_units',opening),('receipts',receipt),('dispatch_units',dispatched),('closing_units',closing)]:close(r[label],value,label)
            dc_stock[dc,item]=closing;totals['hold']+=closing*params[item]['holding_cost_per_unit_day'];totals['dc_receipts']+=receipt
            counts['dc_balance_checks']+=1
        purchase_cash=sum(r.units*params[r.item_id]['buy_cost'] for r in purchase.itertuples())
        for dc,group in purchase.groupby('dc_id'):
            require(sum(r.units*params[r.item_id]['buy_cost'] for r in group.itertuples())<=om['parameters']['dc_budgets'][dc]+tolerance,'purchase budget')
        truck_cash=float(trucks.trucks.sum()*om['truck_trip_cost'])
        order_cash=float((orders.cases>tolerance).sum()*om['order_line_cost'])
        cash=purchase_cash+truck_cash+order_cash+totals['hold']+totals['disposal']
        for label,value in [('actual_demand',totals['actual']),('served_units',totals['served']),('lost_units',totals['lost']),
                            ('waste_proxy_units',totals['waste']),('holding_cash',totals['hold']),('disposal_cash',totals['disposal']),
                            ('purchase_cash',purchase_cash),('cash_cost',cash),('operating_score',cash+totals['lost_penalty']),
                            ('overflow_unit_mornings',totals['overflow']),('ignored_closed_day_recorded_units',totals['ignored']),
                            ('store_receipts',totals['store_receipts']),('dc_receipts',totals['dc_receipts'])]:close(getattr(daily,label),value,label)
        close(daily.stockout_store_items,totals['stockout'],'stockout count')
        after=sum(store_stock.values())+sum(dc_stock.values())+sum(due.values())
        close(before+float(purchase.units.sum())-totals['served']-totals['waste'],after,'chain conservation')
        close(daily.ending_stock_and_pipeline_units,after,'ending system units')
        counts['cash_checks']+=1;counts['conservation_checks']+=1
    # Final state is independently reconstructed, including unreceived purchases.
    for name,location,expected in [('final_store_stock.csv','store_id',store_stock),('final_dc_stock.csv','dc_id',dc_stock)]:
        saved=pd.read_csv(root/name)
        require(set(zip(saved[location],saved.item_id))==set(expected),'final stock grid')
        for r in saved.itertuples():close(r.units,expected[getattr(r,location),r.item_id],'final stock')
    saved_pipe=pd.read_csv(root/'final_pipeline.csv')
    final_due={}
    for r in saved_pipe.itertuples():
        key=(pd.Timestamp(r.arrival_date).date(),r.location_type,r.location_id,r.item_id)
        final_due[key]=final_due.get(key,0)+r.units
    require(set(final_due)==set(due),'final pipeline keys')
    for key,units in due.items():close(final_due[key],units,'final pipeline units')
    result=dict(passed=True,**counts,max_residual=max_residual,
                scope='Independent chronological reconstruction of saved firm orders, receipts, sales, expected disposal, cash and final pipeline.')
    (root/'execution_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
