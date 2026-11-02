"""Day-by-day replay of one ordering method on top of her `execute_day`.

`run(policy, sales, start, weeks, cfg, ...)` (build-roadmap.md). Each day d:
  1. the policy decides from the State, the forecast rows with origin d, cfg and
     (simple rules only) a PastSales object holding sales strictly before d;
  2. her `optimize.replay.execute_day` executes the decision against the
     observed sales of d (the only place sales of d are used);
  3. the rules-never-broken checks run on that day's ledger and orders.
Test window only: a start before `test_start`, an end after `test_end`, or any
sales row after `test_end` is refused (the holdout stays sealed).

Store-only (replay.warehouse off, 2026-11-02): every warehouse-item starts with
replay.unlimited_warehouse_units and warehouse pipeline rows are dropped, so a store order
is never cut and no supplier order is placed; her execute_day and audit run unchanged.
The forecast covers replay.forecast_days (2) per origin, and execute_day's waste estimate
uses that horizon. The scorecard leaves the warehouse out (replay.kpis).
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from optimize.replay import execute_day
from optimize.replay_audit import audit_replay

from .baselines import warehouse_on
from .history import PastSales
from .kpis import check_rules, compute_kpis

PROJECT = Path(__file__).resolve().parents[1]


def config_sha256(cfg):
    clean = deepcopy(cfg)
    clean.get('order_model', {}).pop('warm_start_plan', None)
    text = yaml.safe_dump(clean, sort_keys=True)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def state_sha256(state):
    parts = [pd.Timestamp(state.date).date().isoformat()]
    for frame in [state.store_stock, state.dc_stock, state.pipeline]:
        parts.append(frame.to_csv(index=False))
    return hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()


def git_commit():
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=PROJECT, capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return f'uncommitted (HEAD {head})' if dirty else head


def _as_date(series):
    return pd.to_datetime(series).dt.date


def _days(start, weeks, end):
    start = pd.Timestamp(start).date()
    if end is None:
        if not isinstance(weeks, int) or isinstance(weeks, bool) or weeks <= 0:
            raise ValueError('weeks must be a positive whole number')
        end = start + timedelta(days=7 * weeks - 1)
    end = pd.Timestamp(end).date()
    if end < start:
        raise ValueError('Replay end precedes start')
    return start, end


def _scalar_audit(audit):
    out = {}
    for key, value in (audit or {}).items():
        if isinstance(value, dict):
            if value:
                out[f'{key}_min'] = float(min(value.values()))
        elif np.isscalar(value):
            out[key] = value
    return out


def run(policy, sales, start, weeks, cfg, *, state, forecasts, end=None, output_dir=None, method=None,
        run_log=None, keep_decisions=False):
    """Replay `policy` from `start` for `weeks` weeks (or to `end`). Returns (daily DataFrame, kpis dict)."""
    started = perf_counter()
    first, last = _days(start, weeks, end)
    windows = cfg['data']['windows']
    test_start, test_end = (pd.Timestamp(windows[k]).date() for k in ['test_start', 'test_end'])
    if first < test_start or last > test_end:
        raise ValueError(f'Replay {first}..{last} must stay inside the test window {test_start}..{test_end}; '
                         'the holdout is sealed')
    if pd.Timestamp(state.date).date() != first:
        raise ValueError('Start state date differs from the replay start')
    om, rp = cfg['order_model'], cfg['replay']
    method = method or getattr(policy, 'name', None) or type(policy).__name__
    store_only = not warehouse_on(cfg)
    if store_only:
        from optimize import OptimizerPolicy
        if isinstance(policy, OptimizerPolicy):
            raise ValueError('The optimizer plans the warehouse too: run it with replay.warehouse on '
                             'and a 7-day forecast')
        cfg = deepcopy(cfg)
        cfg['order_model']['horizon_days'] = int(rp['forecast_days'])   # execute_day's waste estimate
        om = cfg['order_model']
        state = deepcopy(state)
        state.dc_stock['units'] = float(rp['unlimited_warehouse_units'])
        if len(state.pipeline):
            keep = state.pipeline.location_type.astype(str).str.lower() != 'dc'
            state.pipeline = state.pipeline[keep].reset_index(drop=True)

    sales = sales[['date', 'store_id', 'item_id', 'units']].copy()
    sales['date'] = _as_date(sales.date)
    if len(sales) and sales.date.max() > test_end:
        raise ValueError('Sales rows after the test window were passed in; the holdout is sealed')
    store_ids = list(dict.fromkeys(state.store_stock.store_id))
    items = list(dict.fromkeys(state.store_stock.item_id))
    expected = set(zip(state.store_stock.store_id, state.store_stock.item_id))
    oldest = first - timedelta(days=rp['history_days'])
    sales = sales[sales.store_id.isin(store_ids) & sales.item_id.isin(items) & (sales.date >= oldest)]
    sales = sales[sales.date <= last].sort_values(['date', 'store_id', 'item_id']).reset_index(drop=True)
    if sales.duplicated(['date', 'store_id', 'item_id']).any():
        raise ValueError('Duplicate sales key')
    days = pd.date_range(first, last).date
    observed = {d: g for d, g in sales[sales.date >= first].groupby('date')}
    fc = forecasts.copy()
    fc['origin'], fc['date'] = _as_date(fc.origin), _as_date(fc.date)
    fc = fc[fc.origin.isin(days) & fc.store_id.isin(store_ids) & fc.item_id.isin(items)]
    if fc.duplicated(['origin', 'date', 'store_id', 'item_id']).any():
        raise ValueError('Duplicate forecast key')
    fc_by_day = {d: g for d, g in fc.groupby('origin')}
    horizon = om['horizon_days']
    for d in days:
        if d not in observed or set(zip(observed[d].store_id, observed[d].item_id)) != expected:
            raise ValueError(f'Incomplete observed-sales grid for {d}')
        want = {(s, p, d + timedelta(days=t)) for s, p in expected for t in range(horizon)}
        if d not in fc_by_day or set(zip(fc_by_day[d].store_id, fc_by_day[d].item_id, fc_by_day[d].date)) != want:
            raise ValueError(f'Incomplete forecast horizon for origin {d}')

    start_state = deepcopy(state)
    state = deepcopy(state)
    start_hash = state_sha256(start_state)
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'daily_metrics.csv').exists():
            raise ValueError(f'{out} already holds a replay; choose a fresh run name')

    rows, ledgers, decisions = [], [], []
    tables = dict(store_orders=[], dc_orders=[], deliveries=[])
    rule_totals, max_residual = {}, 0.0
    for d in days:
        t0 = perf_counter()
        if getattr(policy, 'uses_past_sales', False):
            past = PastSales(sales, d, rp['history_days'], rp['missing_history_units'])
            decision = policy.decide(state, fc_by_day[d], cfg, past)
        else:
            decision = policy.decide(state, fc_by_day[d], cfg)
        decide_seconds = perf_counter() - t0
        result = execute_day(state, decision, fc_by_day[d], observed[d], cfg)
        rules = check_rules(result.ledger, decision.store_orders, decision.dc_orders, decision.deliveries,
                            result.metrics['conservation_residual_units'], cfg)
        for key, value in rules.items():
            if key not in ('passed', 'max_conservation_residual'):
                rule_totals[key] = rule_totals.get(key, 0) + value
        max_residual = max(max_residual, rules['max_conservation_residual'])
        row = dict(result.metrics)
        row['trucks'] = float(decision.deliveries.trucks.sum())
        plan = decision.plan
        if plan is not None:
            stats = plan.stats
            row.update(status=plan.status, objective=plan.objective, mip_gap=stats['mip_gap'],
                       solve_seconds=stats['solve_seconds'], build_seconds=stats['build_seconds'],
                       used_start_fallback=stats['used_start_fallback'],
                       planned_future_withholding=stats['future_sales_rationing_units'],
                       n_vars=stats['n_vars'], n_int=stats['n_int'],
                       plan_audit_passed=decision.audit['passed'],
                       physical_sales_checks=decision.audit['physical_sales_checks'])
        else:
            row.update(status='rule', plan_audit_passed=True, **_scalar_audit(decision.audit))
        row.update(decide_seconds=decide_seconds, elapsed_seconds=perf_counter() - t0)
        rows.append(row)
        ledgers.append(result.ledger)
        for name in tables:
            tables[name].append(getattr(decision, name))
        if keep_decisions:
            decisions.append(decision)
        if out is not None:
            header = len(rows) == 1
            pd.DataFrame([row]).to_csv(out / 'daily_metrics.csv', mode='a', header=header, index=False)
            result.ledger.to_csv(out / 'inventory_ledger.csv', mode='a', header=header, index=False)
            for frame, name in [(decision.store_orders, 'order_plan.csv'), (decision.dc_orders, 'dc_orders.csv'),
                                (decision.deliveries, 'deliveries.csv'),
                                (decision.pipeline_adjustments, 'pipeline_adjustments.csv')]:
                frame.to_csv(out / name, mode='a', header=header, index=False)
            if plan is not None and plan.stats.get('solver_log'):
                (out / 'solver_logs').mkdir(exist_ok=True)
                (out / 'solver_logs' / f'{d.isoformat()}.log').write_text(plan.stats['solver_log'])
        state = result.state

    daily = pd.DataFrame(rows)
    ledger = pd.concat(ledgers, ignore_index=True)
    store_orders, dc_orders, deliveries = (pd.concat(tables[n], ignore_index=True)
                                           for n in ['store_orders', 'dc_orders', 'deliveries'])
    kpis = dict(method=method, start=first.isoformat(), end=last.isoformat(), days=len(days))
    kpis.update(compute_kpis(daily, ledger, store_orders, dc_orders, deliveries, start_state, state, cfg))
    kpis['start_state_sha256'] = start_hash
    kpis['rules'] = dict(passed=True, **rule_totals, max_conservation_residual=max_residual)
    kpis['pipeline_adjustments'] = int(daily.pipeline_adjustments.sum())
    if 'budget_scaled' in daily:
        kpis['budget_scaled_days'] = int(daily.budget_scaled.astype(bool).sum())
        kpis['dc_short_lines'] = int(daily.dc_short_lines.sum())
        kpis['store_order_blocked_days'] = int((~daily.store_can_order.astype(bool)).sum())
    if 'n_vars' in daily:
        gaps = daily.mip_gap.dropna()
        kpis['optimizer'] = dict(n_vars_max=int(daily.n_vars.max()), n_int_max=int(daily.n_int.max()),
                                 solve_seconds_median=float(daily.solve_seconds.median()),
                                 solve_seconds_max=float(daily.solve_seconds.max()),
                                 solve_seconds_total=float(daily.solve_seconds.sum()),
                                 fallback_days=int(daily.used_start_fallback.astype(bool).sum()),
                                 status_counts={str(k): int(v) for k, v in daily.status.value_counts().items()},
                                 max_gap=float(gaps.max()) if len(gaps) else None,
                                 median_gap=float(gaps.median()) if len(gaps) else None,
                                 all_plan_audits_passed=bool(daily.plan_audit_passed.all()),
                                 solver_settings=om['solver'])
    if 'ilp_n_vars' in daily:                          # Lump 3 store ILP (method store_ilp)
        kpis['store_ilp'] = dict(n_vars_max=int(daily.ilp_n_vars.max()), n_int_max=int(daily.ilp_n_int.max()),
                                 solve_seconds_total=float(daily.ilp_seconds.sum()),
                                 solve_seconds_max=float(daily.ilp_seconds.max()),
                                 days_solved=int(daily.ilp_n_vars.notna().sum()),
                                 binding_limit_store_days=int(daily.ilp_binding.fillna(0).sum()))
    kpis['decide_seconds_total'] = float(daily.decide_seconds.sum())
    kpis['elapsed_seconds'] = perf_counter() - started
    service = kpis.pop('_service_table')

    if out is not None:
        state.store_stock.to_csv(out / 'final_store_stock.csv', index=False)
        state.dc_stock.to_csv(out / 'final_dc_stock.csv', index=False)
        state.pipeline.to_csv(out / 'final_pipeline.csv', index=False)
        service.to_csv(out / 'service_by_store_item.csv', index=False)
        if rp['independent_audit']:
            audit = audit_replay(start_state, fc, sales[sales.date >= first], cfg, out)
            kpis['independent_audit'] = {k: v for k, v in audit.items() if k != 'scope'}
        (out / 'kpis.json').write_text(json.dumps(kpis, indent=2, allow_nan=False, default=str) + '\n')
        kpis['elapsed_seconds'] = perf_counter() - started
    if run_log is not None:
        record = dict(logged=datetime.now(timezone.utc).isoformat(timespec='seconds'), method=method,
                      start=first.isoformat(), end=last.isoformat(), config_sha256=config_sha256(cfg),
                      git_commit=git_commit(), output_dir=str(out) if out else None)
        Path(run_log).parent.mkdir(parents=True, exist_ok=True)
        with open(run_log, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')
    if keep_decisions:
        kpis['_decisions'] = decisions
    return daily, kpis
