"""Synthetic rolling diagnostic; not L4 policy integration or a real-data replay."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import date,timedelta
import io
import json
from pathlib import Path

import pandas as pd
import yaml

from optimize import State,plan_orders
from optimize.audit import audit_plan


def run_synthetic(config_path, *, legacy=False, surprise=False):
    root = Path(__file__).resolve().parents[1]
    fixture = yaml.safe_load((root/'examples'/'behavior.yaml').read_text())
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg['network'] = dict(dcs=['DC'],home_dc={'SHOP':'DC'})
    om = cfg['order_model']
    om.update(horizon_days=fixture['horizon_days'],lead_time_store_days=fixture['store_lead_days'],
              lead_time_supplier_days=fixture['supplier_lead_days'],
              truck_capacity_cases=fixture['truck_capacity_cases'],
              truck_trip_cost=fixture['truck_trip_cost'],order_line_cost=fixture['order_line_cost'])
    if legacy:
        om.update(fulfillment_policy='legacy_allocation',warehouse_reserve_policy='terminal_only')
    price = fixture['price']
    cost = price*om['buy_cost_share_of_price']
    demand = fixture['daily_demand']
    p = dict(units_per_case=fixture['units_per_case'],cases_per_pallet=fixture['cases_per_pallet'],
             price=price,buy_cost=cost,holding_cost_per_unit_day=cost*om['holding_rate_per_year']/om['holding_days_per_year'],
             service_level=om['service_levels']['default'],shelf_life_days=None)
    om['parameters'] = dict(items={'SKU':p},store_averages={'SHOP':{'SKU':demand}},
                            dc_budgets={'DC':om['budget_factor']*om['horizon_days']*demand*cost})
    first = date.fromisoformat(fixture['date'])
    state = State(first,pd.DataFrame([dict(store_id='SHOP',item_id='SKU',units=fixture['opening_store_units'])]),
                  pd.DataFrame([dict(dc_id='DC',item_id='SKU',units=fixture['opening_dc_units'])]),
                  pd.DataFrame(columns=['order_date','arrival_date','location_type','location_id','item_id','units']))
    rows, logs = [], []
    for offset in range(fixture['days']):
        today = first+timedelta(days=offset)
        forecasts = pd.DataFrame([dict(origin=today,date=today+timedelta(days=t),store_id='SHOP',item_id='SKU',
                                       q50=demand,q80=demand,q90=demand,q95=demand)
                                  for t in range(om['horizon_days'])])
        stream = io.StringIO()
        with redirect_stdout(stream):
            plan = plan_orders(state,forecasts,cfg)
        logs.append(dict(date=today.isoformat(),summary=stream.getvalue(),cbc=plan.stats['solver_log']))
        if not plan.stats['incumbent_validated']:
            raise RuntimeError('Synthetic solve has no validated incumbent: '+plan.status)
        audit_plan(state,forecasts,cfg,plan)
        pipe = state.pipeline.copy()
        arriving = pipe[pipe.arrival_date == today] if not pipe.empty else pipe
        remaining = pipe[pipe.arrival_date > today].to_dict('records') if not pipe.empty else []
        store_receipt = float(arriving.loc[arriving.location_type == 'store','units'].sum())
        dc_receipt = float(arriving.loc[arriving.location_type == 'dc','units'].sum())
        firm = plan.orders[plan.orders.date == today]
        firm_dc = plan.dc_orders[plan.dc_orders.date == today]
        units = float(firm.units.sum())
        purchased = float(firm_dc.units.sum())
        warehouse = float(state.dc_stock.units.sum())+dc_receipt-units
        morning = float(state.store_stock.units.sum())+store_receipt
        actual = fixture['surprise_demand'] if surprise and offset+1 == fixture['surprise_day'] else demand
        served = min(actual,morning)
        closing = morning-served
        tol = om['solver']['feasibility_tolerance']
        if min(warehouse,closing) < -tol:
            raise RuntimeError('Physical synthetic state became negative')
        for kind,location,qty,lead in [('store','SHOP',units,om['lead_time_store_days']),
                                       ('dc','DC',purchased,om['lead_time_supplier_days'])]:
            if qty > tol:
                remaining.append(dict(order_date=today,arrival_date=today+timedelta(days=lead),
                                       location_type=kind,location_id=location,item_id='SKU',units=qty))
        on_order = sum(r['units'] for r in remaining if r['location_type'] == 'dc' and
                       r['arrival_date'] <= today+timedelta(days=om['lead_time_supplier_days']))
        position = warehouse+on_order
        target = demand*om['lead_time_supplier_days']
        trucks = float(plan.deliveries.loc[plan.deliveries.date == today,'trucks'].sum())
        ordering = int((firm.cases > tol).sum())*om['order_line_cost']
        cash = purchased*cost+trucks*om['truck_trip_cost']+ordering+(closing+warehouse)*p['holding_cost_per_unit_day']
        rows.append(dict(date=today.isoformat(),actual_demand=actual,served=served,lost_units=actual-served,
                         store_closing_units=closing,dc_closing_units=warehouse,dc_inventory_position=position,
                         protection_target=target,protection_deficit=max(target-position,0),
                         supplier_units=purchased,store_shipped_units=units,cash_cost=cash,
                         planned_future_withholding=plan.stats['future_sales_rationing_units'],
                         solve_seconds=plan.stats['solve_seconds'],status=plan.status))
        state = State(today+timedelta(days=1),pd.DataFrame([dict(store_id='SHOP',item_id='SKU',units=closing)]),
                      pd.DataFrame([dict(dc_id='DC',item_id='SKU',units=warehouse)]),
                      pd.DataFrame(remaining,columns=state.pipeline.columns))
    frame = pd.DataFrame(rows)
    ordered = frame[frame.supplier_units > om['solver']['feasibility_tolerance']]
    store_pipeline=sum(r['units'] for r in remaining if r['location_type']=='store')
    total_inventory=closing+warehouse+sum(r['units'] for r in remaining)
    summary = dict(mode='legacy' if legacy else 'revised',surprise=surprise,days=fixture['days'],
                   lost_units=float(frame.lost_units.sum()),supplier_units=float(frame.supplier_units.sum()),
                   first_supplier_order=None if ordered.empty else ordered.date.iloc[0],
                   dc_protection_deficit_unit_days=float(frame.protection_deficit.sum()),
                   cash_cost=float(frame.cash_cost.sum()),
                   ending_store_units=float(frame.store_closing_units.iloc[-1]),
                   ending_dc_position=float(frame.dc_inventory_position.iloc[-1]),
                   ending_store_pipeline_units=float(store_pipeline),
                   ending_stock_and_pipeline_units=float(total_inventory),
                   scope='Controlled synthetic test; cash cost excludes model preference penalties.')
    return frame,summary,logs


def main():
    root = Path(__file__).resolve().parents[1]
    config = root/'config.yaml'
    output = root/'outputs'/'behavior';output.mkdir(parents=True,exist_ok=True)
    summaries = []
    for surprise in [False,True]:
        for legacy in [True,False]:
            frame,summary,logs = run_synthetic(config,legacy=legacy,surprise=surprise)
            label=('surprise' if surprise else 'constant')+'_'+('legacy' if legacy else 'revised')
            frame.to_csv(output/(label+'.csv'),index=False)
            (output/(label+'_logs.json')).write_text(json.dumps(logs,indent=2)+'\n')
            summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
