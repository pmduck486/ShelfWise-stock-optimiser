"""Scorecard and the rules-never-broken checks (plan/l4-replay-plan.md).

Service measures leave out closed store days (25 Dec); costs keep every day.
Fill rate is aggregate and unit-based: served units / demand units over open days.

Store-only (replay.warehouse off, 2026-11-02): the warehouse is left out of every figure.
Stock value, days of stock and turnover count stores only; the warehouse's (unlimited) stock
and its holding cost are dropped; goods are charged at buy cost when they leave the warehouse
for a store (purchase_cash); net cost values only store stock and store-bound pipeline.
"""

import numpy as np
import pandas as pd

ROADMAP_KEYS = ['fill_rate', 'stockout_days', 'lost_units', 'waste_units', 'avg_stock_value',
                'days_of_stock', 'turnover', 'total_cost', 'orders', 'trucks']

DEFINITIONS = {
    'fill_rate': 'served units / demand units, open store days only (aggregate, unit-based)',
    'stockout_days': 'store-item-days with lost units > 0, open days only',
    'lost_units': 'demand - served, open days only',
    'waste_units': "expected-disposal units, her forecast-cover estimate (q50 for every method), all days",
    'avg_stock_value': 'mean over all days of closing store + warehouse units x buy cost (on hand, not pipeline)',
    'days_of_stock': 'mean closing store + warehouse units / mean daily demand over open days',
    'turnover': '(served units x buy cost / days x days_per_year) / avg_stock_value',
    'total_cost': 'cash over all days: purchases + trucks + order lines + holding + disposal',
    'net_cost': 'total_cost - (ending stock + pipeline value - starting stock + pipeline value), at buy cost',
    'orders': 'store order lines with cartons > 0 (supplier lines reported separately)',
    'trucks': 'truck trips, all days',
    'service_target': 'per store-item: share of open days with no lost sales vs its service level',
}


def _value(state, buy, stores_only=False):
    units = state.store_stock.groupby('item_id').units.sum()
    pipe = state.pipeline
    if stores_only:
        if len(pipe):
            pipe = pipe[pipe.location_type.astype(str).str.lower() == 'store']
    else:
        units = units.add(state.dc_stock.groupby('item_id').units.sum(), fill_value=0)
    if len(pipe):
        units = units.add(pd.to_numeric(pipe.units).groupby(pipe.item_id).sum(), fill_value=0)
    return float(sum(float(u) * buy[p] for p, u in units.items()))


def compute_kpis(daily, ledger, store_orders, dc_orders, deliveries, start_state, final_state, cfg):
    om = cfg['order_model']
    items = om['parameters']['items']
    buy = {p: v['buy_cost'] for p, v in items.items()}
    store_only = cfg['replay'].get('warehouse', 'on') in (False, 'off')
    tol = om['solver']['feasibility_tolerance']
    closed = daily.closed.astype(bool).to_numpy()
    open_days = daily[~closed]
    n_days, n_open = len(daily), int((~closed).sum())
    demand = float(open_days.actual_demand.sum())
    served = float(open_days.served_units.sum())

    led = ledger.copy()
    led['date'] = led.date.astype(str)
    led['value'] = led.closing_units * led.item_id.map(buy)
    closed_dates = set(daily.date.astype(str)[closed])
    by_day = led.groupby(['date', 'location_type']).value.sum().unstack(fill_value=0.0)
    for col in ['store', 'dc']:
        if col not in by_day:
            by_day[col] = 0.0
    store_value = float(by_day['store'].mean())
    dc_value = float(by_day['dc'].mean())
    avg_value = store_value + dc_value
    mean_units = float(led.groupby('date').closing_units.sum().mean())
    breakdown = {c: float(daily[c].sum()) for c in
                 ['purchase_cash', 'truck_cash', 'order_cash', 'holding_cash', 'disposal_cash']}
    if store_only:
        dcl = led[led.location_type == 'dc']
        dc_value, avg_value = None, store_value
        mean_units = float(led[led.location_type == 'store'].groupby('date').closing_units.sum().mean())
        breakdown['holding_cash'] -= float(dcl.holding_cost.sum())
        breakdown['purchase_cash'] = float((dcl.dispatch_units * dcl.item_id.map(buy)).sum())
    mean_demand = demand / n_open if n_open else None
    stores = led[led.location_type == 'store']
    open_store = stores[~stores.date.isin(closed_dates)]
    served_value = float((open_store.served_units * open_store.item_id.map(buy)).sum())
    days_per_year = cfg['replay']['days_per_year']
    start_value, end_value = _value(start_state, buy, store_only), _value(final_state, buy, store_only)
    total_cost = float(sum(breakdown.values())) if store_only else float(daily.cash_cost.sum())

    # Service target per store-item: share of open days with no lost sales.
    per = open_store.groupby(['location_id', 'item_id']).lost_units.apply(lambda x: float((x <= tol).mean()))
    target = per.index.get_level_values('item_id').map(lambda p: items[p]['service_level']).to_numpy(float)
    met = per.to_numpy() >= target - 1e-12
    service = pd.DataFrame(dict(store_id=per.index.get_level_values(0), item_id=per.index.get_level_values(1),
                                no_loss_day_share=per.to_numpy(), target=target, met=met))
    fill_item = open_store.groupby('item_id')[['served_units', 'actual_demand']].sum()

    k = dict(
        fill_rate=served / demand if demand > 0 else None,
        stockout_days=int(open_days.stockout_store_items.sum()),
        lost_units=float(open_days.lost_units.sum()),
        waste_units=float(daily.waste_proxy_units.sum()),
        avg_stock_value=avg_value,
        days_of_stock=mean_units / mean_demand if mean_demand else None,
        turnover=(served_value / n_days * days_per_year) / avg_value if avg_value > 0 else None,
        total_cost=total_cost,
        orders=int((store_orders.cases > tol).sum()),
        trucks=int(round(float(deliveries.trucks.sum()))),
        # --- extra fields (not part of the roadmap contract) ---
        net_cost=total_cost - (end_value - start_value),
        lost_sales_penalty=float(daily.lost_sales_penalty.sum()),
        operating_score=total_cost + float(daily.lost_sales_penalty.sum()),
        avg_store_stock_value=store_value,
        avg_warehouse_stock_value=dc_value,
        supplier_order_lines=int((dc_orders.pallets > tol).sum()),
        supplier_pallets=float(dc_orders.pallets.sum()),
        supplier_units=float(daily.supplier_units.sum()),
        store_cartons=float(store_orders.cases.sum()),
        cost_breakdown=breakdown,
        warehouse='off (store-only: warehouse not counted)' if store_only else 'on',
        stock_and_pipeline_value=dict(start=start_value, end=end_value),
        service_target=dict(store_items=int(len(service)),
                            share_meeting_target=float(met.mean()) if len(service) else None,
                            mean_no_loss_day_share=float(per.mean()) if len(service) else None,
                            by_level={str(lvl): dict(store_items=int((g.target == lvl).sum()),
                                                     share_meeting_target=float(g.met.mean()),
                                                     mean_no_loss_day_share=float(g.no_loss_day_share.mean()))
                                      for lvl, g in service.groupby('target')}),
        fill_rate_by_item={p: (float(r.served_units / r.actual_demand) if r.actual_demand > 0 else None)
                           for p, r in fill_item.iterrows()},
        denominators=dict(days=n_days, open_days=n_open, closed_dates=sorted(closed_dates),
                          demand_units_open_days=demand, served_units_open_days=served,
                          store_item_open_days=int(len(open_store)),
                          mean_daily_demand_open_days=mean_demand, mean_closing_units=mean_units,
                          served_value_open_days=served_value, days_per_year=days_per_year),
        definitions=DEFINITIONS,
    )
    k['_service_table'] = service       # written to service_by_store_item.csv, not to kpis.json
    return k


def check_rules(ledger, store_orders, dc_orders, deliveries, residual, cfg):
    """Raise ValueError on the first broken rule; return counts of checks passed."""
    om = cfg['order_model']
    tol = om['solver']['feasibility_tolerance']
    items = om['parameters']['items']
    stores = ledger[ledger.location_type == 'store']
    dcs = ledger[ledger.location_type == 'dc']
    if (stores.morning_units < -tol).any() or (ledger.closing_units < -tol).any():
        raise ValueError('Rule broken: negative stock')
    if (dcs.dispatch_units > dcs.morning_units + tol).any():
        raise ValueError('Rule broken: stores received more than the warehouse holds')
    for frame, n, pack in [(store_orders, 'cases', lambda p: items[p]['units_per_case']),
                           (dc_orders, 'pallets', lambda p: items[p]['units_per_case'] * items[p]['cases_per_pallet'])]:
        counts = frame[n].to_numpy(float)
        if (counts < -tol).any() or (np.abs(counts - np.round(counts)) > tol).any():
            raise ValueError('Rule broken: not whole cartons/pallets')
        if (np.abs(frame.units.to_numpy(float) - np.round(counts) * frame.item_id.map(pack).to_numpy(float)) > tol).any():
            raise ValueError('Rule broken: units are not whole packs')
    loads = store_orders.groupby('store_id').cases.sum()
    cap = om['truck_capacity_cases']
    for r in deliveries.itertuples():
        if (abs(float(loads.get(r.store_id, 0.0)) - r.cases_total) > tol or abs(r.trucks - round(r.trucks)) > tol
                or r.cases_total > r.trucks * cap + tol):
            raise ValueError('Rule broken: truck capacity / truck load')
    for d, g in dc_orders.groupby('dc_id'):
        spend = float((g.units * g.item_id.map(lambda p: items[p]['buy_cost'])).sum())
        if spend > om['parameters']['dc_budgets'][d] + tol:
            raise ValueError('Rule broken: warehouse budget exceeded')
    if abs(residual) > tol:
        raise ValueError(f'Rule broken: chain conservation residual {residual}')
    return dict(passed=True, stock_nonnegative=int(len(ledger)), dispatch_within_warehouse_stock=int(len(dcs)),
                whole_packs=int(len(store_orders) + len(dc_orders)), trucks_within_capacity=int(len(deliveries)),
                budget_kept=int(dc_orders.dc_id.nunique()), conservation=1,
                max_conservation_residual=float(abs(residual)))
