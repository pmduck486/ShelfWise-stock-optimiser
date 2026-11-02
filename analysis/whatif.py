"""Lump 3 what-ifs on the store-only model (2026-11-02).

what_if(param, values, cfg, ...) re-runs the replay once per value with ONE setting changed
(param is a dotted path into the compiled config, e.g. order_model.truck_capacity_cases) and
returns one row per value: the contract columns of whatif.csv (param, value, total_cost,
stockout_days, avg_stock_value) plus fill, lost and waste units, net cost + lost-sale penalty and,
for method store_ilp, the mean shadow price of each limit and on how many store-days it bound.

Run every what-if listed in config.yaml (analysis.whatif.runs) from the retailops folder:
    python -m analysis.whatif --config configs/validation_fast.yaml --run-name whatif_store_ilp_2026-11-02
Writes outputs/replay/<run-name>/whatif.csv and whatif.md; each run's replay files sit in
<run-name>/<param>=<value>/<method>/ and every run is logged to outputs/test_runs.log (R9a).
These are sensitivity runs on the test window: they show how the plan reacts, and are never
used to choose settings.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import yaml

from replay.baselines import make_policy
from replay.cli import ROOT, _path, load_sales, write_shadow_prices
from replay.engine import run

CONTRACT = ['param', 'value', 'total_cost', 'stockout_days', 'avg_stock_value']
EXTRA = ['fill_rate', 'lost_units', 'waste_units', 'net_cost', 'lost_sales_penalty', 'net_cost_plus_penalty']


def get_path(cfg, dotted):
    node = cfg
    for part in dotted.split('.'):
        node = node[part]
    return node


def set_path(cfg, dotted, value):
    parts = dotted.split('.')
    node = cfg
    for part in parts[:-1]:
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f'{dotted} is not a setting in the config')
    node[parts[-1]] = value


def _row(param, value, k, shadow):
    row = dict(param=param, value=value, **{c: k[c] for c in CONTRACT[2:]})
    row.update({c: k[c] for c in EXTRA[:-1]})
    row['net_cost_plus_penalty'] = k['net_cost'] + k['lost_sales_penalty']
    if shadow is not None and len(shadow):
        for name, g in shadow.groupby('constraint'):
            row[f'shadow_{name}_mean'] = float(g.value.mean())
            row[f'shadow_{name}_binding_store_days'] = int((g.value > 0).sum())
    return row


def what_if(param, values, cfg, *, method, sales, start, state, forecasts, weeks=None, end=None,
            out_dir=None, run_log=None, cache=None):
    """One replay per value with `param` set to it. `cache` maps a value to a finished row (the base)."""
    rows = []
    for value in values:
        if cache is not None and value in cache:
            rows.append(dict(cache[value], param=param, value=value))
            continue
        c = deepcopy(cfg)
        set_path(c, param, value)
        policy = make_policy(method)
        sub = None if out_dir is None else Path(out_dir) / f'{param}={value}' / method
        _, k = run(policy, sales, start, weeks, c, state=deepcopy(state), forecasts=forecasts, end=end,
                   output_dir=sub, method=f'{method} [{param}={value}]', run_log=run_log)
        shadow = None
        if getattr(policy, 'shadow_rows', None):
            shadow = pd.DataFrame(policy.shadow_rows)
            if sub is not None:
                write_shadow_prices(policy.shadow_rows, sub)
        rows.append(_row(param, value, k, shadow))
    return pd.DataFrame(rows)


def main(argv=None):
    from optimize.parameters import load_case
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, default=ROOT / 'configs' / 'validation_fast.yaml')
    ap.add_argument('--replay-config', type=Path, default=ROOT / 'config.yaml')
    ap.add_argument('--forecast-file', type=Path, default=ROOT / 'data' / 'processed' / 'forecasts.parquet')
    ap.add_argument('--profile', choices=['small', 'full'], default='full')
    ap.add_argument('--end-date', help='Shorter runs (tests); default: replay.weeks')
    ap.add_argument('--run-name', required=True)
    args = ap.parse_args(argv)

    main_cfg = yaml.safe_load(_path(args.replay_config).read_text())
    rp, an = main_cfg['replay'], main_cfg['analysis']
    case = load_case(_path(rp['input_dir']), _path(args.config), profile=args.profile,
                     forecast_path=_path(args.forecast_file))
    cfg = case.parameters.config
    cfg['replay'], cfg['analysis'] = rp, an
    run_dir = _path(rp['output_dir']) / args.run_name
    if (run_dir / 'whatif.csv').exists():
        raise SystemExit(f'{run_dir} already has results; choose a fresh --run-name')
    run_dir.mkdir(parents=True, exist_ok=True)
    sales = load_sales(rp['history_sales_file'], cfg['data']['windows']['test_end'])
    forecasts = pd.read_parquet(_path(args.forecast_file))
    method = an['whatif']['method']
    common = dict(method=method, sales=sales, start=case.state.date, state=case.state, forecasts=forecasts,
                  weeks=None if args.end_date else rp['weeks'], end=args.end_date, out_dir=run_dir,
                  run_log=_path(rp['test_runs_log']))
    base = None
    tables = []
    for spec in an['whatif']['runs']:
        param, values = spec['param'], list(spec['values'])
        if get_path(cfg, param) != values[0]:
            raise ValueError(f'{param}: the first value must be the current setting {get_path(cfg, param)}')
        cache = None
        if base is not None:
            cache = {values[0]: base}
        t = what_if(param, values, cfg, cache=cache, **common)
        if base is None:
            base = t.iloc[0].to_dict()
        tables.append(t)
        print(t.to_string(index=False), flush=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(run_dir / 'whatif.csv', index=False)
    cols = ['param', 'value', 'fill_rate', 'lost_units', 'waste_units', 'avg_stock_value', 'net_cost_plus_penalty']
    cols += [c for c in table.columns if c.startswith('shadow_') and c.endswith('_mean')]
    lines = [f'# What-ifs: {args.run_name}', '', f'Method {method}, store-only, {case.state.date} '
             f'to {args.end_date or "the end of the test window"}. One setting changed per row; the first '
             'row of each setting is the base.', '',
             '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in table[cols].iterrows():
        lines.append('| ' + ' | '.join(f'{v:,.4f}' if isinstance(v, float) and abs(v) < 10 else
                                         (f'{v:,.0f}' if isinstance(v, float) else str(v)) for v in r) + ' |')
    (run_dir / 'whatif.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (run_dir / 'run.json').write_text(json.dumps(dict(run_name=args.run_name, config=str(args.config),
                                                      forecast_file=str(args.forecast_file), method=method,
                                                      runs=an['whatif']['runs']), indent=2) + '\n')
    print(f'Outputs: {run_dir}')


if __name__ == '__main__':
    main()
