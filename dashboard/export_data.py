"""Export the saved RetailOps results into small JSON files for the dashboard.

Run from RetailOps/dashboard (or anywhere) with the project venv:

    ../../.venv/Scripts/python export_data.py

Reads only files that already exist under RetailOps/ (replay outputs, forecasts, items,
the M5 calendar, forecaster scores, docs/decisions.md for the layer table) and writes
JSON into dashboard/public/data/. Nothing is re-run, re-solved or re-tuned. The one thing
it recomputes is each store-item's target line and spoilage cap for every replay day, by
calling the project's own code (replay.store_model.StoreModel.lines_and_caps) on the same
forecast file and compiled config the replay used. A self-check rebuilds the rule-only
`store_model` orders from those targets and compares them with that run's saved order plan.

Deterministic: same inputs give byte-identical outputs (sorted keys, fixed rounding).
"""

from __future__ import annotations

import json
import math
import sys
import time
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # RetailOps/
OUT = HERE / 'public' / 'data'
sys.path.insert(0, str(ROOT))

from replay.baselines import store_cover_window, _closed  # noqa: E402
from replay.store_model import StoreModel, stock_level    # noqa: E402

METHOD_LABELS = {
    'last_week': 'Habit (repeat last week)',
    'topup_avg': 'Top-up to 4-week average',
    'topup_forecast': 'Top-up to our forecast',
    'store_model': 'Store model (rule only)',
    'store_ilp': 'Store ILP (main model)',
}
SIMPLE = ['last_week', 'topup_avg', 'topup_forecast']
METHODS = SIMPLE + ['store_model', 'store_ilp']

PERIODS = {
    'test': dict(
        label='Test window', range='1 Nov to 26 Dec 2015', weeks=8,
        run_dirs={'store_ilp': 'outputs/replay/store_ilp_2026-11-02',
                  'store_model': 'outputs/replay/store_ilp_2026-11-02',
                  'last_week': 'outputs/replay/store_only_2day_2026-11-02',
                  'topup_avg': 'outputs/replay/store_only_2day_2026-11-02',
                  'topup_forecast': 'outputs/replay/store_only_2day_2026-11-02'},
        forecasts='data/processed/forecasts.parquet',
        note=('The saved test-window run of store_ilp is the core ILP (service floor and spoilage-cap '
              'layers off). The main model with both layers was run on the test window in a separate '
              'cloud copy; its numbers are in docs/decisions.md #40 (see the Layers page).'),
    ),
    'b1': dict(
        label='Holdout block 1', range='27 Dec 2015 to 13 Feb 2016', weeks=7,
        run_dirs={m: 'outputs/replay_holdout/holdout_block1' for m in METHODS},
        forecasts='data/processed_holdout_b1/forecasts.parquet',
        forecast_scores='data/processed_holdout_b1/forecast_scores.csv',
        note='Run once with frozen code and settings; store_ilp is the main model (service floor + spoilage cap on).',
    ),
    'b2': dict(
        label='Holdout block 2', range='14 Feb to 2 Apr 2016', weeks=7,
        run_dirs={m: 'outputs/replay_holdout/holdout_block2' for m in METHODS},
        forecasts='data/processed_holdout_b2/forecasts.parquet',
        forecast_scores='data/processed_holdout_b2/forecast_scores.csv',
        note='Run once with frozen code and settings; store_ilp is the main model (service floor + spoilage cap on).',
    ),
}

# Layer table: these test-window runs were made in a cloud copy and are not saved in RetailOps/.
# The numbers are copied from docs/decisions.md #40; tests/test_export.py checks every one of
# them still appears in that file, and that the core row equals the saved results_table.csv.
LAYERS_SOURCE = 'docs/decisions.md, decision #40 (runs saved only in the cloud copy)'
LAYERS = [
    dict(id='core', label='Core store ILP (no layers)', floor=False, spoil=False, stockcap=False,
         fill=94.87, lost=26208, waste=8043, stock=35279, total=934088, verdict='baseline',
         fill_txt='94.87', lost_txt='26,208', waste_txt='8,043', stock_txt='35,279', total_txt='934,088'),
    dict(id='floor', label='+ service floor (90%)', floor=True, spoil=False, stockcap=False,
         fill=95.63, lost=22318, waste=11680, stock=35797, total=935257, verdict='tested alone',
         fill_txt='95.63', lost_txt='22,318', waste_txt='11,680', stock_txt='35,797', total_txt='935,257'),
    dict(id='spoil', label='+ spoilage cap in the ILP', floor=False, spoil=True, stockcap=False,
         fill=94.44, lost=28400, waste=6568, stock=35226, total=934465, verdict='tested alone',
         fill_txt='94.44', lost_txt='28,400', waste_txt='6,568', stock_txt='35,226', total_txt='934,465'),
    dict(id='main', label='+ floor + spoilage cap (main model)', floor=True, spoil=True, stockcap=False,
         fill=94.96, lost=25727, waste=7015, stock=35663, total=932094, verdict='kept',
         fill_txt='94.96', lost_txt='25,727', waste_txt='7,015', stock_txt='35,663', total_txt='932,094'),
    dict(id='stockcap', label='+ soft stock cap', floor=True, spoil=True, stockcap=True,
         fill=94.47, lost=28243, waste=6809, stock=33444, total=934553, verdict='dropped by the pre-set test',
         fill_txt='94.47', lost_txt='28,243', waste_txt='6,809', stock_txt='33,444', total_txt='934,553'),
    dict(id='holiday', label='+ floor + spoilage cap + holiday policy', floor=True, spoil=True, stockcap=False, holiday=True,
         fill=94.78, lost=26641, waste=6662, stock=37262, total=932421, verdict='kept: lead-time holiday model',
         fill_txt='94.78', lost_txt='26,641', waste_txt='6,662', stock_txt='37,262', total_txt='932,421'),
]
LAYER_TEST = ('Pre-set keep/drop test for the stock cap (written before the runs): store stock must fall '
              'at least 5%, cost may rise at most 0.2%, fill may fall at most 0.3 points. Result: stock -6.2% '
              '(passes), cost +0.26% and fill -0.49 points (fail), so it was dropped.')


def r(x, nd=2):
    """Round for JSON; None for NaN/inf."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    v = round(x, nd)
    if v == 0:
        v = 0.0
    return int(v) if nd == 0 else v


def write(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(',', ':'), sort_keys=True, ensure_ascii=True, allow_nan=False)
    path.write_text(text, encoding='utf-8', newline='\n')
    return len(text)


def load_cfg(run_dir: str) -> dict:
    with open(ROOT / run_dir / 'compiled_config.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)


def results_table(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / run_dir / 'results_table.csv').set_index('method')


def method_row(period_key: str, method: str) -> dict:
    p = PERIODS[period_key]
    row = results_table(p['run_dirs'][method]).loc[method]
    kp = json.loads((ROOT / p['run_dirs'][method] / 'kpis.json').read_text(encoding='utf-8'))[method]
    audit = kp.get('independent_audit')
    rules = kp.get('rules')
    return dict(
        id=method, label=METHOD_LABELS[method],
        fill=r(row.fill_rate, 6), lost=r(row.lost_units, 1), waste=r(row.waste_units, 1),
        stock=r(row.avg_store_stock_value, 1), net_cost=r(row.net_cost, 1),
        penalty=r(row.lost_sales_penalty, 1), total=r(row.net_cost + row.lost_sales_penalty, 1),
        stockout_days=int(row.stockout_days), orders=int(row.orders), trucks=int(row.trucks),
        purchase=r(row.purchase_cash, 1), turnover=r(row.turnover, 2), days_of_stock=r(row.days_of_stock, 3),
        n_vars=r(row.n_vars, 0) if not pd.isna(row.n_vars) else None,
        n_int=r(row.n_int, 0) if not pd.isna(row.n_int) else None,
        solve_seconds=r(row.solve_seconds_total, 1) if not pd.isna(row.solve_seconds_total) else None,
        audit_passed=bool(audit.get('passed')) if isinstance(audit, dict) else None,
        rules_passed=bool(rules.get('passed')) if isinstance(rules, dict) else None,
        service_met_share=r(row.service_met_share, 4),
        source=f"{p['run_dirs'][method]}/results_table.csv".replace('\\', '/'),
    )


def build_summary():
    periods = {}
    for key, p in PERIODS.items():
        rows = [method_row(key, m) for m in METHODS]
        best = min((x for x in rows if x['id'] in SIMPLE), key=lambda x: x['total'])
        for x in rows:
            x['vs_best_simple'] = r((x['total'] - best['total']) / best['total'] * 100, 3)
        periods[key] = dict(label=p['label'], range=p['range'], weeks=p['weeks'], note=p['note'],
                            best_simple=best['id'], methods=rows,
                            store_ilp_layers='off (core)' if key == 'test' else 'on (service floor + spoilage cap)')
    cfg = load_cfg(PERIODS['test']['run_dirs']['store_ilp'])
    return dict(
        periods=periods, period_order=list(PERIODS),
        layers=dict(source=LAYERS_SOURCE, rows=[{k: v for k, v in x.items() if not k.endswith('_txt')} for x in LAYERS],
                    test=LAYER_TEST),
        money=dict(unit='$', note=('Money values are in M5 shelf-price units: Walmart prices from the M5 '
                                   'sell_prices file, in US dollars (config.yaml calls them "USD-equivalent model '
                                   'price units"; the project notes write them as Rs). Buy cost = 85% of price.')),
        totals_definition='net cost + lost-sale penalty: net cost (cash spent minus the change in stock value) plus '
                          f"{cfg['order_model']['lost_sale_penalty_share']} x price for every lost unit. Lower is better.",
    )


def m5_calendar() -> pd.DataFrame:
    cal = pd.read_csv(ROOT / 'data/m5/calendar.csv')
    cal['date'] = pd.to_datetime(cal['date']).dt.date
    return cal.set_index('date')


def events_for(cal, d):
    if d not in cal.index:
        return [], {}
    row = cal.loc[d]
    ev = []
    for n, t in (('event_name_1', 'event_type_1'), ('event_name_2', 'event_type_2')):
        if isinstance(row[n], str) and row[n]:
            ev.append(dict(name=row[n], type=row[t]))
    snap = {s: int(row[f'snap_{s}']) for s in ('CA', 'TX', 'WI')}
    return ev, snap


def build_items(cfgs: dict):
    items = pd.read_csv(ROOT / 'data/processed/items.csv')
    items = items.sort_values('item_id').reset_index(drop=True)
    cfg0 = cfgs['test']
    out = []
    for row in items.itertuples():
        it = cfg0['order_model']['parameters']['items'][row.item_id]
        out.append(dict(id=row.item_id, dept=row.dept_id, rank=int(row.rank), avg_daily=r(row.avg_daily_units, 2),
                        perishable=bool(row.is_perishable),
                        shelf_life=None if pd.isna(row.shelf_life_days) else int(row.shelf_life_days),
                        upc=int(it['units_per_case'])))
    params = {}
    for key, cfg in cfgs.items():
        om = cfg['order_model']
        rows = []
        for x in out:
            it = om['parameters']['items'][x['id']]
            assert int(it['units_per_case']) == x['upc'], 'case size differs between periods'
            rows.append(dict(price=r(it['price'], 4), buy=r(it['buy_cost'], 4),
                             hold=r(it['holding_cost_per_unit_day'], 7), service_level=it['service_level'],
                             level=r(stock_level(cfg, x['id']), 5)))
        lay = cfg.get('analysis', {}).get('store_ilp', {}).get('layers', {})
        a = cfg.get('analysis', {}).get('store_ilp', {})
        params[key] = dict(
            items=rows,
            lost_sale_penalty_share=om['lost_sale_penalty_share'], waste_disposal_share=om['waste_disposal_share'],
            order_line_cost=om['order_line_cost'], truck_trip_cost=om['truck_trip_cost'],
            truck_capacity_cases=om['truck_capacity_cases'], store_min_order_cases=om['store_min_order_cases'],
            shelf_days=om['shelf_days'], shelf_days_perishable=om['shelf_days_perishable'],
            shelf_min_cases=om['shelf_min_cases'], space_factor=a.get('space_factor'),
            budget_factor=a.get('budget_factor'), time_limit_s=a.get('time_limit_s'),
            service_floor=dict(enabled=bool(lay.get('service_floor', {}).get('enabled', False)),
                               share=lay.get('service_floor', {}).get('share'),
                               gap_penalty_factor=lay.get('service_floor', {}).get('gap_penalty_factor')),
            spoilage_cap_in_ilp=bool(lay.get('spoilage_cap_in_ilp', False)),
            stock_value_cap=bool(lay.get('stock_value_cap', {}).get('enabled', False)),
            closed_days=om['calendar']['closed_store_month_days'],
        )
    return dict(items=out, params=params,
                source='data/processed/items.csv + each run\'s compiled_config.yaml (order_model.parameters)')


def read_method_csv(run_dir, method, name):
    return pd.read_csv(ROOT / run_dir / method / f'{name}.csv')


def forecasts_indexed(path: str) -> pd.DataFrame:
    fc = pd.read_parquet(ROOT / path)
    fc['origin'] = pd.to_datetime(fc.origin).dt.date
    fc['date'] = pd.to_datetime(fc.date).dt.date
    return fc


def build_period(key: str, p: dict, cfg: dict, cal, stores, item_ids, sizes: dict):
    t0 = time.perf_counter()
    rd = p['run_dirs']['store_ilp']
    ledger = read_method_csv(rd, 'store_ilp', 'inventory_ledger')
    ledger = ledger[ledger.location_type == 'store'].copy()
    ledger['date'] = pd.to_datetime(ledger.date).dt.date
    plan = read_method_csv(rd, 'store_ilp', 'order_plan')
    plan['date'] = pd.to_datetime(plan.date).dt.date
    shadow = read_method_csv(rd, 'store_ilp', 'shadow_prices')
    shadow['date'] = pd.to_datetime(shadow.date).dt.date
    shadow['item_id'] = shadow['item_id'].fillna('')
    daily = read_method_csv(rd, 'store_ilp', 'daily_metrics')
    daily['date'] = pd.to_datetime(daily.date).dt.date
    deliveries = read_method_csv(rd, 'store_ilp', 'deliveries')
    deliveries['date'] = pd.to_datetime(deliveries.date).dt.date
    fc = forecasts_indexed(p['forecasts'])

    # other methods: daily KPIs and their own order on each store-item-day
    other_daily, other_cases = {}, {}
    for m in METHODS:
        dm = read_method_csv(p['run_dirs'][m], m, 'daily_metrics')
        dm['date'] = pd.to_datetime(dm.date).dt.date
        other_daily[m] = dm.set_index('date')
        if m != 'store_ilp':
            op = read_method_csv(p['run_dirs'][m], m, 'order_plan')
            op['date'] = pd.to_datetime(op.date).dt.date
            other_cases[m] = op.set_index(['date', 'store_id', 'item_id']).cases

    # Load holiday replay data if available for test period
    holiday_daily, holiday_cases, holiday_units, holiday_trucks = None, None, None, None
    holiday_plan_item_cases, cfg_holiday = None, None
    if key == 'test':
        h_dir = ROOT / 'outputs' / 'replay' / 'store_ilp_holiday' / 'store_ilp'
        if (h_dir / 'daily_metrics.csv').exists():
            h_dm = pd.read_csv(h_dir / 'daily_metrics.csv')
            h_dm['date'] = pd.to_datetime(h_dm.date).dt.date
            holiday_daily = h_dm.set_index('date')
            if (h_dir / 'order_plan.csv').exists():
                h_op = pd.read_csv(h_dir / 'order_plan.csv')
                h_op['date'] = pd.to_datetime(h_op.date).dt.date
                holiday_cases = h_op.groupby('date')['cases'].sum()
                holiday_units = h_op.groupby('date')['units'].sum()
                holiday_plan_item_cases = h_op.set_index(['date', 'store_id', 'item_id']).cases
            if (h_dir / 'deliveries.csv').exists():
                h_deliv = pd.read_csv(h_dir / 'deliveries.csv')
                h_deliv['date'] = pd.to_datetime(h_deliv.date).dt.date
                holiday_trucks = h_deliv.groupby('date')['trucks'].sum()
            main_cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
            cfg_holiday = deepcopy(cfg)
            if 'analysis' in main_cfg and 'store_ilp' in main_cfg['analysis']:
                cfg_holiday['analysis']['store_ilp']['layers'] = deepcopy(main_cfg['analysis']['store_ilp']['layers'])
                cfg_holiday['analysis']['store_ilp']['layers']['holiday_policy']['enabled'] = True

    days = sorted(ledger.date.unique())
    key_order = [(s, i) for s in stores for i in item_ids]
    led = ledger.set_index(['date', 'store_id', 'item_id'])
    cases = plan.set_index(['date', 'store_id', 'item_id']).cases
    fc_idx = fc.set_index(['origin', 'store_id', 'item_id', 'date']).sort_index()
    model = StoreModel()
    lay = cfg.get('analysis', {}).get('store_ilp', {}).get('layers', {})
    floor_on = bool(lay.get('service_floor', {}).get('enabled', False))
    spoil_in_ilp = bool(lay.get('spoilage_cap_in_ilp', False))
    floor_share = lay.get('service_floor', {}).get('share') if floor_on else None
    row_of = {k: j for j, k in enumerate(key_order)}

    calendar = []
    series = {s: {i: {f: [] for f in ('q50', 'q80', 'q90', 'q95', 'act', 'stock', 'order')} for i in item_ids}
              for s in stores}
    floor_matrix = {s: [] for s in stores}
    floor_item_counts = {}
    day_bytes = 0
    ilp_check = [0, 0]
    alt_methods =['last_week', 'topup_avg', 'topup_forecast', 'store_model']
    for d in days:
        can_order, window = store_cover_window(d, cfg)
        fcd = fc_idx.loc[d]                                 # this origin's rows: (store, item, date)
        ev, snap = events_for(cal, d)
        dm = daily.set_index('date').loc[d]
        L = led.loc[d].reindex(key_order)
        q = {}
        dates_fc = sorted(fcd.index.get_level_values('date').unique())
        for t, fd in enumerate(dates_fc[:2]):
            sub = fcd.xs(fd, level='date').reindex(key_order)
            q[t] = sub[['q50', 'q80', 'q90', 'q95']].to_numpy(float)
        if can_order:
            ctx = dict(cfg=cfg, fc=fcd, today=d)
            line, caps = model.lines_and_caps(ctx, key_order, window)
        else:
            line = np.full(len(key_order), np.nan)
            caps = np.full(len(key_order), np.nan)
        dc = cases.loc[d].reindex(key_order).fillna(0).to_numpy(float) if d in cases.index.get_level_values(0) else np.zeros(len(key_order))
        alt = {}
        for m in alt_methods:
            oc = other_cases[m]
            alt[m] = oc.loc[d].reindex(key_order).fillna(0).to_numpy(float) if d in oc.index.get_level_values(0) else np.zeros(len(key_order))
        sh = shadow[shadow.date == d]
        per_store = {}
        for s in stores:
            ent = {}
            for c in ('truck', 'space', 'budget'):
                x = sh[(sh.constraint == c) & (sh.store_id == s)]
                if len(x):
                    x = x.iloc[0]
                    ent[c] = [r(x.limit, 1), r(x.used, 1), r(x.value, 4)]
            dv = deliveries[(deliveries.date == d) & (deliveries.store_id == s)]
            ent['trucks'] = int(dv.trucks.sum()) if len(dv) else 0
            ent['cartons'] = int(round(sum(dc[row_of[(s, i)]] for i in item_ids)))
            per_store[s] = ent
        layer_rows = []
        for x in sh[sh.constraint.isin(['floor', 'spoil'])].itertuples():
            layer_rows.append([row_of[(x.store_id, x.item_id)], 0 if x.constraint == 'floor' else 1,
                               r(x.value, 3), r(x.limit, 2), r(x.used, 1)])
            if x.constraint == 'floor':
                floor_item_counts[(x.store_id, x.item_id)] = floor_item_counts.get((x.store_id, x.item_id), 0) + 1
        layer_rows.sort()
        if can_order:
            m_ok, m_n = ilp_term_check(cfg, key_order, line, caps, L.morning_units.to_numpy(float), dc, len(window),
                                       floor_on, floor_share, spoil_in_ilp)
            ilp_check[0] += m_ok
            ilp_check[1] += m_n
        # Compute holiday line and caps if holiday policy is available
        h_line, h_caps, h_cases_arr = None, None, None
        if cfg_holiday is not None and can_order:
            ctx_h = dict(cfg=cfg_holiday, fc=fcd, today=d)
            h_line, h_caps = model.lines_and_caps(ctx_h, key_order, window)
            if holiday_plan_item_cases is not None and d in holiday_plan_item_cases.index.get_level_values(0):
                h_cases_arr = holiday_plan_item_cases.loc[d].reindex(key_order).fillna(0).to_numpy(float)

        rows = []
        act = L.actual_demand.to_numpy(float)
        morn = L.morning_units.to_numpy(float)
        lost = L.lost_units.to_numpy(float)
        waste = L.waste_proxy_units.to_numpy(float)
        for j in range(len(key_order)):
            rows.append([
                r(q[0][j, 0], 2), r(q[0][j, 1], 2), r(q[0][j, 2], 2), r(q[0][j, 3], 2),
                r(q[1][j, 0], 2) if 1 in q else None, r(q[1][j, 1], 2) if 1 in q else None,
                r(q[1][j, 2], 2) if 1 in q else None, r(q[1][j, 3], 2) if 1 in q else None,
                r(act[j], 2), r(morn[j], 2), r(line[j], 2), r(caps[j], 2), int(round(dc[j])),
                r(lost[j], 2), r(waste[j], 2),
                *[int(round(alt[m][j])) for m in alt_methods],
                r(h_line[j], 2) if h_line is not None else None,
                r(h_caps[j], 2) if h_caps is not None else None,
                int(round(h_cases_arr[j])) if h_cases_arr is not None else None,
            ])
        day_obj = dict(date=d.isoformat(), period=key, can_order=bool(can_order),
                       cover=[w.isoformat() for w in window], fc_dates=[x.isoformat() for x in dates_fc[:2]],
                       closed=bool(_closed(d, cfg)), stores=per_store, layers=layer_rows,
                       floor_on=floor_on, floor_share=floor_share, spoil_in_ilp=spoil_in_ilp,
                       cols=['q50a', 'q80a', 'q90a', 'q95a', 'q50b', 'q80b', 'q90b', 'q95b', 'act', 'pos',
                             'line', 'cap', 'cases', 'lost', 'waste', 'alt_last_week', 'alt_topup_avg',
                             'alt_topup_forecast', 'alt_store_model', 'h_line', 'h_cap', 'h_cases'],
                       rows=rows)
        day_bytes += write(OUT / 'day' / key / f'{d.isoformat()}.json', day_obj)

        # series (same-day forecast line vs actual) for the forecast explorer
        for s in stores:
            for i in item_ids:
                j = row_of[(s, i)]
                ser = series[s][i]
                for t, f in enumerate(('q50', 'q80', 'q90', 'q95')):
                    ser[f].append(r(q[0][j, t], 1))
                ser['act'].append(r(act[j], 1))
                ser['stock'].append(r(morn[j], 1))
                ser['order'].append(int(round(dc[j])))

        fl = sh[sh.constraint == 'floor']
        sp = sh[sh.constraint == 'spoil']
        for s in stores:
            fs = fl[fl.store_id == s]
            floor_matrix[s].append([len(fs), r(fs.value.mean(), 2) if len(fs) else 0.0])
        demand = float(dm.actual_demand)
        served = float(dm.served_units)
        per_m = {}
        for m in METHODS:
            od = other_daily[m].loc[d]
            per_m[m] = [r(od.served_units / od.actual_demand, 5) if od.actual_demand > 0 else None,
                        r(od.lost_units, 1), r(od.waste_proxy_units, 1)]
        limits_used = {}
        for c in ('truck', 'space', 'budget'):
            x = sh[sh.constraint == c]
            if len(x):
                share = (x.used / x.limit.replace(0, np.nan)).dropna()
                limits_used[c] = [r(share.mean(), 4), r(share.max(), 4), int((x.value > 0).sum())]

        holiday_metrics = None
        if holiday_daily is not None and d in holiday_daily.index:
            h_row = holiday_daily.loc[d]
            h_served = float(h_row.served_units)
            h_lost = float(h_row.lost_units)
            h_waste = float(h_row.waste_proxy_units)
            h_dem = float(h_row.actual_demand)
            h_fill = r(h_served / h_dem, 5) if h_dem > 0 else None
            h_cart = int(round(holiday_cases.get(d, 0))) if holiday_cases is not None else int(round(dc.sum()))
            h_un = int(round(holiday_units.get(d, 0))) if holiday_units is not None else int(round(plan[plan.date == d].units.sum()))
            h_trk = int(holiday_trucks.get(d, 0)) if holiday_trucks is not None else int(deliveries[deliveries.date == d].trucks.sum())
            holiday_metrics = dict(
                served=r(h_served, 1),
                lost=r(h_lost, 1),
                waste=r(h_waste, 1),
                fill=h_fill,
                cartons=h_cart,
                units=h_un,
                trucks=h_trk,
            )

        calendar.append(dict(
            date=d.isoformat(), dow=d.weekday(), closed=bool(_closed(d, cfg)), can_order=bool(can_order),
            cover_days=len(window), demand=r(demand, 1), served=r(served, 1), lost=r(dm.lost_units, 1),
            fill=r(served / demand, 5) if demand > 0 else None, waste=r(dm.waste_proxy_units, 1),
            cartons=int(round(dc.sum())), units=int(round(plan[plan.date == d].units.sum())),
            trucks=int(deliveries[deliveries.date == d].trucks.sum()),
            floor_n=int(len(fl)), floor_mean=r(fl.value.mean(), 3) if len(fl) else None,
            floor_max=r(fl.value.max(), 3) if len(fl) else None,
            spoil_n=int(len(sp)), spoil_mean=r(sp.value.mean(), 3) if len(sp) else None,
            binding=int(dm.ilp_binding) if not pd.isna(dm.ilp_binding) else None,
            n_vars=r(dm.ilp_n_vars, 0) if can_order and not pd.isna(dm.ilp_n_vars) else None,
            n_int=r(dm.ilp_n_int, 0) if can_order and not pd.isna(dm.ilp_n_int) else None,
            ilp_seconds=r(dm.ilp_seconds, 3) if can_order and not pd.isna(dm.ilp_seconds) else None,
            events=ev, snap=snap, methods=per_m, limits=limits_used, holiday=holiday_metrics,
        ))

    ser_bytes = 0
    for s in stores:
        ser_bytes += write(OUT / 'series' / key / f'{s}.json',
                           dict(store=s, period=key, dates=[d.isoformat() for d in days], items=series[s]))
    sizes[f'day/{key}'] = day_bytes
    sizes[f'series/{key}'] = ser_bytes

    # shadow price summary
    lim = shadow[shadow.constraint.isin(['truck', 'space', 'budget'])].copy()
    lim['share'] = lim.used / lim.limit.replace(0, np.nan)
    limit_stats = {}
    for c, g in lim.groupby('constraint'):
        limit_stats[c] = dict(store_days=int(len(g)), binding=int((g.value > 0).sum()), value_max=r(g.value.max(), 4),
                              use_mean=r(g.share.mean(), 4), use_max=r(g.share.max(), 4))
    lay_rows = shadow[shadow.constraint.isin(['floor', 'spoil'])]
    layer_stats = {}
    for c, g in lay_rows.groupby('constraint'):
        layer_stats[c] = dict(rows=int(len(g)), mean=r(g.value.mean(), 3), max=r(g.value.max(), 3),
                              median=r(g.value.median(), 3), store_items=int(g.groupby(['store_id', 'item_id']).ngroups))
    top_items = sorted(floor_item_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    shadow_obj = dict(
        limits=limit_stats, layers=layer_stats,
        floor_matrix=dict(stores=stores, dates=[d.isoformat() for d in days], cells=[floor_matrix[s] for s in stores]),
        top_floor_items=[dict(store=k[0], item=k[1], days=v) for k, v in top_items],
        source=f'{rd}/store_ilp/shadow_prices.csv'.replace('\\', '/'),
    )
    shadow_obj['ilp_term_check'] = dict(
        matched=ilp_check[0], total=ilp_check[1],
        what=('store-item-days where the cartons the ILP chose also minimise that item\'s own term of the ILP '
              'objective (the cost curve drawn on the Day decision page); the rest are items whose choice was '
              'shaped by the shared store terms (one truck trip per store) or by the solver\'s gap tolerance'))
    print(f'  {key}: {len(days)} days in {time.perf_counter() - t0:.1f} s; ILP item-term check '
          f'{ilp_check[0]}/{ilp_check[1]}')
    return calendar, shadow_obj, days, ledger, fc


def ilp_term_check(cfg, key_order, line, caps, pos, chosen, cover_days, floor_on, floor_share, spoil_in_ilp):
    """Does each item's chosen carton count minimise its own ILP objective term? (dashboard cost curve)"""
    om = cfg['order_model']
    items = om['parameters']['items']
    gpf = cfg.get('analysis', {}).get('store_ilp', {}).get('layers', {}).get('service_floor', {}).get('gap_penalty_factor', 0)
    ok = n = 0
    for j, (s, p) in enumerate(key_order):
        it = items[p]
        upc = it['units_per_case']
        u = om['lost_sale_penalty_share'] * it['price']
        o = it['holding_cost_per_unit_day'] * cover_days if it['shelf_life_days'] is None else it['buy_cost'] * (1 + om['waste_disposal_share'])
        cap = caps[j]
        target = line[j] if (spoil_in_ilp or not math.isfinite(cap)) else min(line[j], cap)
        need = target - pos[j]
        big = math.ceil(max(need, 0.0) / upc) + om['store_min_order_cases']
        costs = {}
        for k in range(0, big + 1):
            units = k * upc
            if spoil_in_ilp and math.isfinite(cap) and units > max(cap - pos[j], 0.0) + 1e-9:
                continue
            c = u * max(need - units, 0) + o * max(units - need, 0) + (om['order_line_cost'] if k > 0 else 0)
            if floor_on:
                fu = line[j] if not math.isfinite(cap) else min(line[j], cap)
                c += gpf * u * max(floor_share * fu - pos[j] - units, 0)
            costs[k] = c
        k_chosen = int(round(chosen[j]))
        n += 1
        if k_chosen in costs and costs[k_chosen] <= min(costs.values()) + 1e-6:
            ok += 1
    return ok, n


def self_check_store_model(cfg, key_order, d, run_dir, fc) -> tuple[int, int]:
    """Rebuild store_model's orders on day d from the exported targets and compare with its order plan."""
    led = read_method_csv(run_dir, 'store_model', 'inventory_ledger')
    led = led[(led.location_type == 'store')]
    led = led[pd.to_datetime(led.date).dt.date == d].set_index(['store_id', 'item_id']).reindex(key_order)
    plan = read_method_csv(run_dir, 'store_model', 'order_plan')
    plan = plan[pd.to_datetime(plan.date).dt.date == d].set_index(['store_id', 'item_id']).reindex(key_order)
    _, window = store_cover_window(d, cfg)
    fcd = fc.set_index(['origin', 'store_id', 'item_id', 'date']).sort_index().loc[d]
    model = StoreModel()
    target = model.store_targets(dict(cfg=cfg, fc=fcd, today=d), key_order, window)
    items = cfg['order_model']['parameters']['items']
    upc = np.array([items[p]['units_per_case'] for _, p in key_order], float)
    need = target - led.morning_units.to_numpy(float)
    count = np.ceil(np.maximum(need, 0) / upc - 1e-9).astype(int)
    mn = cfg['order_model']['store_min_order_cases']
    count = np.where((count > 0) & (count < mn), mn, count)
    return int((count == plan.cases.to_numpy().astype(int)).sum()), len(key_order)


def build_forecast_scores(test_ledger, test_fc):
    sc = json.loads((ROOT / 'outputs/forecast/scores/model_val_calibrated.json').read_text(encoding='utf-8'))
    base = json.loads((ROOT / 'outputs/forecast/scores/baseline_val.json').read_text(encoding='utf-8'))
    stub = json.loads((ROOT / 'outputs/forecast/scores/stub_val.json').read_text(encoding='utf-8'))

    def seg(rows, name='all'):
        return next(x for x in rows if x['segment'] == name)

    folds = {}
    for f in ('fold1_festive', 'fold2_main'):
        model_rows = sc['folds'][f].get('after') or sc['folds'][f]['before']
        out = {}
        for label, rows in (('model', model_rows), ('naive', base['folds'][f]['scores']),
                            ('stub', stub['folds'][f]['scores'])):
            a = seg(rows)
            out[label] = {k: r(a[k], 4) for k in ('mae', 'pinball_q50', 'pinball_q80', 'pinball_q90', 'pinball_q95',
                                                  'cover_q50', 'cover_q80', 'cover_q90', 'cover_q95', 'n')}
        per = seg(model_rows, 'perishable')
        out['model_perishable'] = {k: r(per[k], 4) for k in ('mae', 'cover_q50', 'cover_q80', 'cover_q90', 'cover_q95')}
        folds[f] = out
    fold_info = {f: dict(start=base['folds'][f].get('val_start'), end=base['folds'][f].get('val_end')) for f in folds}

    holdout = {}
    for key in ('b1', 'b2'):
        h = pd.read_csv(ROOT / PERIODS[key]['forecast_scores']).iloc[0]
        holdout[key] = dict(days=int(h.days), rows=int(h.rows), mae=r(h.q50_mae, 4), cover_q80=r(h.q80_coverage, 3),
                            cover_q90=r(h.q90_coverage, 3), cover_q95=r(h.q95_coverage, 3),
                            source=PERIODS[key]['forecast_scores'])

    # test window: same-day forecast (origin = date) vs the replay's observed demand, 25 Dec left out
    led = test_ledger[['date', 'store_id', 'item_id', 'actual_demand']]
    f0 = test_fc[test_fc.origin == test_fc.date]
    m = f0.merge(led, on=['date', 'store_id', 'item_id'], how='inner')
    m = m[~m.date.map(lambda d: d.strftime('%m-%d') == '12-25')]
    items = pd.read_csv(ROOT / 'data/processed/items.csv').set_index('item_id')
    m['per'] = m.item_id.map(items.is_perishable).astype(bool)

    def score(g):
        return dict(n=int(len(g)), mae=r((g.q50 - g.actual_demand).abs().mean(), 4),
                    **{f'cover_{c}': r((g.actual_demand <= g[c]).mean() * 100, 3) for c in ('q50', 'q80', 'q90', 'q95')})
    test = dict(all=score(m), perishable=score(m[m.per]), long_life=score(m[~m.per]),
                note='computed by export_data.py: same-day forecast (origin = date) vs the replay\'s observed demand, '
                     '1,200 store-items, 25 Dec left out')
    return dict(folds=folds, fold_info=fold_info, holdout=holdout, test=test,
                calibration=dict(shifts=sc.get('shifts'), tolerance_pts=sc.get('tolerance_pts')),
                sources=['outputs/forecast/scores/model_val_calibrated.json', 'outputs/forecast/scores/baseline_val.json',
                         'outputs/forecast/scores/stub_val.json', 'data/processed_holdout_b1/forecast_scores.csv',
                         'data/processed_holdout_b2/forecast_scores.csv'])


def build_whatif():
    w = pd.read_csv(ROOT / 'outputs/replay/whatif_store_ilp_2026-11-02/whatif.csv')
    rows = []
    for x in w.itertuples():
        rows.append(dict(param=x.param, value=r(x.value, 3), fill=r(x.fill_rate, 6), lost=r(x.lost_units, 1),
                         waste=r(x.waste_units, 1), stock=r(x.avg_stock_value, 1), total=r(x.net_cost_plus_penalty, 1),
                         net_cost=r(x.net_cost, 1), penalty=r(x.lost_sales_penalty, 1),
                         shadow=dict(truck=[r(x.shadow_truck_mean, 4), int(x.shadow_truck_binding_store_days)],
                                     space=[r(x.shadow_space_mean, 4), int(x.shadow_space_binding_store_days)],
                                     budget=[r(x.shadow_budget_mean, 4), int(x.shadow_budget_binding_store_days)])))
    return dict(rows=rows, source='outputs/replay/whatif_store_ilp_2026-11-02/whatif.csv',
                note=('Each row changes one setting of the core store ILP on the 8 test weeks. Sensitivity only, '
                      'never used to choose settings. Rows that change the lost-sale penalty price their own '
                      'penalty at that share, so their total is not comparable with the other rows.'))


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ('day', 'series'):
        for f in sorted((OUT / sub).rglob('*.json')) if (OUT / sub).exists() else []:
            f.unlink()
    sizes = {}
    cal = m5_calendar()
    cfgs = {k: load_cfg(p['run_dirs']['store_ilp']) for k, p in PERIODS.items()}
    items = build_items(cfgs)
    item_ids = [x['id'] for x in items['items']]
    stores = sorted(cfgs['test']['order_model']['parameters']['store_averages'].keys())
    sizes['items.json'] = write(OUT / 'items.json', dict(**items, stores=stores))
    summary = build_summary()
    sizes['summary.json'] = write(OUT / 'summary.json', summary)

    calendars, shadows, test_ledger, test_fc = {}, {}, None, None
    print('Exporting days:')
    for key, p in PERIODS.items():
        cal_rows, sh, days, ledger, fc = build_period(key, p, cfgs[key], cal, stores, item_ids, sizes)
        calendars[key] = dict(label=p['label'], range=p['range'], start=days[0].isoformat(), end=days[-1].isoformat(),
                              layers='off (core ILP)' if key == 'test' else 'on (service floor + spoilage cap)',
                              days=cal_rows)
        shadows[key] = sh
        if key == 'test':
            test_ledger, test_fc = ledger, fc
            key_order = [(s, i) for s in stores for i in item_ids]
            checks = [self_check_store_model(cfgs['test'], key_order, d, p['run_dirs']['store_model'], fc)
                      for d in (days[0], days[20], days[40])]
            ok = sum(c[0] for c in checks)
            tot = sum(c[1] for c in checks)
            print(f'  self-check: store_model orders rebuilt from exported targets match {ok}/{tot} store-item-days')
            calendars[key]['self_check'] = dict(matched=ok, total=tot,
                                                 what='store_model orders rebuilt from the exported targets and '
                                                      'morning stock vs its saved order_plan, 3 days')
    sizes['calendar.json'] = write(OUT / 'calendar.json', dict(periods=calendars, period_order=list(PERIODS),
                                                                 stores=stores, method_order=METHODS))
    sizes['shadow.json'] = write(OUT / 'shadow.json', dict(periods=shadows, whatif=build_whatif()))
    sizes['forecast.json'] = write(OUT / 'forecast.json', build_forecast_scores(test_ledger, test_fc))

    total = sum(sizes.values())
    print('Sizes:')
    for k in sorted(sizes):
        print(f'  {k:<16} {sizes[k] / 1e6:7.2f} MB')
    print(f'  {"total":<16} {total / 1e6:7.2f} MB')
    t = summary['periods']
    for key in PERIODS:
        ilp = next(x for x in t[key]['methods'] if x['id'] == 'store_ilp')
        print(f"  {key}: store_ilp fill {ilp['fill'] * 100:.2f}%  waste {ilp['waste']:,.0f}  "
              f"net+penalty {math.floor(ilp['total'] + 0.5):,}  vs best simple {ilp['vs_best_simple']:+.2f}%")
    print(f'Done in {time.perf_counter() - t0:.1f} s')


if __name__ == '__main__':
    main()
