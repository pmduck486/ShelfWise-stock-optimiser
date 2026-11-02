"""Run the replay for several methods from one common start and write the scorecard.

python -m replay.cli --profile small --config configs/validation_fast.yaml \
    --forecast-file data/processed/forecasts.parquet --end-date 2015-11-07 \
    --methods last_week,topup_avg,topup_forecast,optimizer --run-name gate1_small

`--config` supplies data/network/order_model (her compiled parameters, solver budget);
the `replay:` settings always come from `--replay-config` (default: config.yaml).
The start state is loaded once through her common initialization
(`optimize.parameters.load_case`: capacity calibration + safety-net clipping) and
every method gets an identical deep copy of it.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
from time import perf_counter

import pandas as pd
import pulp
import yaml

from optimize.parameters import load_case

from .baselines import make_policy
from .engine import config_sha256, run

ROOT = Path(__file__).resolve().parents[1]

TABLE_COLUMNS = ['fill_rate', 'stockout_days', 'lost_units', 'waste_units', 'avg_stock_value',
                 'avg_store_stock_value', 'avg_warehouse_stock_value', 'days_of_stock', 'turnover',
                 'total_cost', 'net_cost', 'lost_sales_penalty', 'orders', 'trucks', 'supplier_order_lines',
                 'supplier_units', 'service_met_share', 'service_mean_no_loss', 'purchase_cash', 'truck_cash',
                 'order_cash', 'holding_cash', 'disposal_cash', 'budget_scaled_days', 'dc_short_lines',
                 'n_vars', 'n_int', 'solve_seconds_total', 'fallback_days', 'wall_seconds']


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_sales(path, test_end):
    """Past + test-window sales; the parquet filter means holdout rows are never read."""
    sales = pd.read_parquet(_path(path), columns=['date', 'store_id', 'item_id', 'units'],
                            filters=[('date', '<=', pd.Timestamp(test_end))])
    if len(sales) and pd.Timestamp(sales.date.max()) > pd.Timestamp(test_end):
        raise ValueError('Sales after test_end were read; the holdout is sealed')
    return sales


def table_row(method, k):
    opt = k.get('optimizer') or k.get('store_ilp') or {}
    row = {c: k.get(c) for c in TABLE_COLUMNS}
    row.update(k['cost_breakdown'])
    row.update(service_met_share=k['service_target']['share_meeting_target'],
               service_mean_no_loss=k['service_target']['mean_no_loss_day_share'],
               n_vars=opt.get('n_vars_max'), n_int=opt.get('n_int_max'),
               solve_seconds_total=opt.get('solve_seconds_total'), fallback_days=opt.get('fallback_days'),
               wall_seconds=k['elapsed_seconds'])
    return dict(method=method, **row)


SHADOW_COLUMNS = ['date', 'constraint', 'store_id', 'item_id', 'value']   # contract (build-roadmap.md)


def write_shadow_prices(rows, out_dir):
    """shadow_prices.csv (contract columns, plus the limit and how much of it was used)."""
    frame = pd.DataFrame(rows)[SHADOW_COLUMNS + ['limit', 'used']]
    frame.to_csv(Path(out_dir) / 'shadow_prices.csv', index=False)
    return frame


def write_markdown(table, path, meta):
    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return '-'
        if isinstance(v, float):
            return f'{v:.4f}' if abs(v) < 10 else f'{v:,.2f}'
        return str(v)
    t = table.set_index('method').T
    lines = [f"# Replay results: {meta['run_name']}", '',
             f"{meta['profile']} profile, {meta['start']} to {meta['end']}; config `{meta['config']}` "
             f"(solver limit {meta['time_limit_s']} s); forecasts `{meta['forecast_file']}`.", '',
             '| KPI | ' + ' | '.join(t.columns) + ' |', '|---|' + '---|' * len(t.columns)]
    for kpi, values in t.iterrows():
        lines.append(f'| {kpi} | ' + ' | '.join(fmt(v) for v in values) + ' |')
    lines += ['', 'Service measures leave out closed store days (25 Dec); costs keep every day. '
              'Definitions and denominators: kpis.json.']
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', choices=['small', 'full'], default='small')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--replay-config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--input-dir', type=Path, help='Default: replay.input_dir')
    parser.add_argument('--forecast-file', type=Path, default=ROOT / 'data' / 'processed' / 'forecasts.parquet')
    parser.add_argument('--weeks', type=int, help='Default: replay.weeks')
    parser.add_argument('--end-date', help='Overrides --weeks (short runs such as Gate 1)')
    parser.add_argument('--methods', help='Comma list; default: replay.methods')
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args(argv)

    main_cfg = yaml.safe_load(_path(args.replay_config).read_text())
    rp = main_cfg['replay']
    input_dir = _path(args.input_dir or rp['input_dir'])
    case = load_case(input_dir, _path(args.config), profile=args.profile, forecast_path=_path(args.forecast_file))
    cfg = case.parameters.config
    cfg['replay'] = rp
    cfg['analysis'] = main_cfg.get('analysis', {})      # Lump 3 settings (store ILP, what-ifs)
    windows = cfg['data']['windows']
    start = case.state.date
    methods = (args.methods.split(',') if args.methods else rp['methods'])
    run_dir = _path(rp['output_dir']) / args.run_name
    if (run_dir / 'kpis.json').exists():
        raise SystemExit(f'{run_dir} already has results; choose a fresh --run-name')
    run_dir.mkdir(parents=True, exist_ok=True)

    sales = load_sales(rp['history_sales_file'], windows['test_end'])
    forecasts = pd.read_parquet(_path(args.forecast_file))
    weeks = args.weeks or rp['weeks']
    # Her initialization log (clipped units, outside waste and cost): once per run.
    case.initialization_log.to_csv(run_dir / 'initialization_log.csv', index=False)
    (run_dir / 'calibration_manifest.json').write_text(json.dumps(case.parameters.manifest, indent=2) + '\n')
    (run_dir / 'compiled_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = dict(run_name=args.run_name, profile=args.profile, config=str(args.config),
                replay_config=str(args.replay_config), input_dir=str(input_dir), forecast_file=str(args.forecast_file),
                start=str(start), end=args.end_date or str(start + pd.Timedelta(days=7 * weeks - 1)),
                time_limit_s=cfg['order_model']['solver']['time_limit_s'], config_sha256=config_sha256(cfg),
                initial_clipped_units=float(case.initialization_log.clipped_units.sum()),
                stores=sorted(case.state.store_stock.store_id.unique()),
                items=int(case.state.store_stock.item_id.nunique()),
                versions=dict(python=platform.python_version(), pandas=pd.__version__, pulp=pulp.__version__))

    results, rows = {}, []
    for method in methods:
        t0 = perf_counter()
        print(f'--- {method} ---', flush=True)
        policy = make_policy(method)
        daily, k = run(policy, sales, start, weeks, cfg, state=deepcopy(case.state),
                       forecasts=forecasts, end=args.end_date, output_dir=run_dir / method, method=method,
                       run_log=_path(rp['test_runs_log']))
        if getattr(policy, 'shadow_rows', None):
            write_shadow_prices(policy.shadow_rows, run_dir / method)
        k['elapsed_seconds'] = perf_counter() - t0
        results[method] = k
        rows.append(table_row(method, k))
        print(f"{method}: fill {k['fill_rate']:.5f}, lost {k['lost_units']:.2f}, cost {k['total_cost']:.2f}, "
              f"{k['elapsed_seconds']:.1f}s, rules passed {k['rules']['passed']}", flush=True)
    if len({k['start_state_sha256'] for k in results.values()}) != 1:
        raise RuntimeError('Methods did not start from the identical state')
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / 'results_table.csv', index=False)
    write_markdown(table, run_dir / 'results_table.md', meta)
    # kpis.json = {method: {...}} exactly as the roadmap contract; run details go to run.json.
    (run_dir / 'kpis.json').write_text(json.dumps(results, indent=2, allow_nan=False, default=str) + '\n')
    (run_dir / 'run.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')
    print(table.set_index('method').T.to_string())
    print(f'Outputs: {run_dir}')


if __name__ == '__main__':
    main()
