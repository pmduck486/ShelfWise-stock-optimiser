"""Holdout blocks (user, 2026-11-02): the sealed holdout 2015-12-27 .. 2016-05-22 is split into
3 blocks of about 7 weeks. Each block is run once, with the frozen code and settings, and only
that block's sales are ever read; later blocks stay sealed.

  block 1: 2015-12-27 .. 2016-02-13    block 2: 2016-02-14 .. 2016-04-02    block 3: 2016-04-03 .. 2016-05-22

This file only writes copies of the config files with the block's dates and its own folders,
so nothing in config.yaml, configs/validation_fast.yaml or data/processed is touched:
  config_holdout_b<k>_prepare.yaml          data_prep: same windows, holdout_end = block end
  config_holdout_b<k>.yaml                  forecaster + replay: test window = the block,
                                            holdout_start = the day after it (still sealed)
  configs/validation_fast_holdout_b<k>.yaml order model: the same date change
Products, perishables, every model and order setting stay exactly as frozen. Products are still
chosen on sales before 2015-11-01 (data_prep reads the original windows).

Steps, from the retailops folder (laptop):
  python -m holdout.make_block configs --block 1
  python -m data_prep.prepare --config config_holdout_b1_prepare.yaml --include-holdout
  python -m holdout.make_block finish --block 1      # sales_full -> sales.parquet, checks, start stock
  python -m forecast.freeze --config config_holdout_b1.yaml          # trains up to the block's day before
  python -m forecast.write_forecasts --config config_holdout_b1.yaml
  python -m replay.cli --profile full --config configs/validation_fast_holdout_b1.yaml
         --replay-config config_holdout_b1.yaml
         --forecast-file data/processed_holdout_b1/forecasts.parquet --run-name holdout_block1
  python -m holdout.make_block score --block 1       # forecaster accuracy on the block
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = {1: ('2015-12-27', '2016-02-13'), 2: ('2016-02-14', '2016-04-02'), 3: ('2016-04-03', '2016-05-22')}


def _sub(text, pattern, value, count=1):
    """Replace the value of a `key: value` line (keeps indentation and the comment)."""
    rx = re.compile(rf'^(\s*{pattern}:\s*)(["\']?)[^"\'#\s]+(["\']?)', re.M)
    new, n = rx.subn(lambda m: f'{m.group(1)}{m.group(2)}{value}{m.group(3)}', text)
    if n != count:
        raise ValueError(f'{pattern}: expected {count} line(s), found {n}')
    return new


def names(k):
    return dict(processed=f'data/processed_holdout_b{k}', model=f'outputs/forecast/model_holdout_b{k}',
                prepare=ROOT / f'config_holdout_b{k}_prepare.yaml', main=ROOT / f'config_holdout_b{k}.yaml',
                order=ROOT / 'configs' / f'validation_fast_holdout_b{k}.yaml')


def block_dates(k):
    start, end = (pd.Timestamp(d) for d in BLOCKS[k])
    if k == 3:
        raise NotImplementedError('block 3 ends on the last M5 day; its configs need a different seal')
    return start, end, end + pd.Timedelta(days=1)


def write_configs(k):
    start, end, sealed = block_dates(k)
    n = names(k)
    weeks = ((end - start).days + 1) // 7
    main = (ROOT / 'config.yaml').read_text()
    order = (ROOT / 'configs' / 'validation_fast.yaml').read_text()
    head = (f'# HOLDOUT BLOCK {k} COPY made by holdout/make_block.py: only dates and folders differ from the\n'
            f'# frozen file it was copied from. Do not edit by hand.\n')

    prep = _sub(main, 'processed_dir', n['processed'])
    prep = _sub(prep, 'holdout_end', end.date())                        # read sales up to the block's end only
    n['prepare'].write_text(head + prep)

    m = _sub(main, 'processed_dir', n['processed'])
    for key, value in (('test_start', start.date()), ('test_end', end.date()), ('holdout_start', sealed.date()),
                       ('model_dir', n['model']), ('weeks', weeks), ('input_dir', n['processed']),
                       ('history_sales_file', f"{n['processed']}/sales.parquet"),
                       ('output_dir', 'outputs/replay_holdout'), ('test_runs_log', 'outputs/holdout_runs.log')):
        m = _sub(m, key, value)
    n['main'].write_text(head + m)

    o = _sub(order, 'processed_dir', n['processed'])
    for key, value in (('test_start', start.date()), ('test_end', end.date()), ('holdout_start', sealed.date())):
        o = _sub(o, key, value)
    n['order'].write_text(head + o)
    print(f'block {k}: {start.date()} .. {end.date()} ({weeks} weeks); sealed from {sealed.date()}')
    for key in ('prepare', 'main', 'order'):
        print(f'  wrote {n[key].relative_to(ROOT)}')


def finish(k):
    """After data_prep: check the block's sales, rename them, write the start stock."""
    from stubs.make_stubs import start_state, write_state
    from forecast.data import load_config
    start, end, _ = block_dates(k)
    n = names(k)
    out = ROOT / n['processed']
    full = pd.read_parquet(out / 'sales_full.parquet')
    full['date'] = pd.to_datetime(full['date'])
    # reference = the development file for block 1, the previous block's file for later blocks
    ref_dir = 'data/processed' if k == 1 else names(k - 1)['processed']
    dev = pd.read_parquet(ROOT / ref_dir / 'sales.parquet')
    dev['date'] = pd.to_datetime(dev['date'])
    if full['date'].max() != end:
        raise ValueError(f'block sales end on {full["date"].max().date()}, expected {end.date()}')
    # same products, and every row up to the block start identical to the development file
    items_dev = pd.read_csv(ROOT / 'data' / 'processed' / 'items.csv')
    items_new = pd.read_csv(out / 'items.csv')
    if not items_dev.equals(items_new):
        raise ValueError('items.csv differs from the development one: products must stay as chosen')
    before = full[full['date'] < start].sort_values(['date', 'store_id', 'item_id']).reset_index(drop=True)
    dev_part = dev[dev['date'] < start].sort_values(['date', 'store_id', 'item_id']).reset_index(drop=True)
    if dev_part['date'].max() != start - pd.Timedelta(days=1):
        raise ValueError(f'{ref_dir}/sales.parquet ends {dev_part["date"].max().date()}, not the day before the block')
    pd.testing.assert_frame_equal(before, dev_part[before.columns], check_dtype=False)
    (out / 'sales_full.parquet').rename(out / 'sales.parquet')
    cfg = load_config(names(k)['main'])
    sales = pd.read_parquet(out / 'sales.parquet', columns=['date', 'store_id', 'item_id', 'units'])
    sales['date'] = pd.to_datetime(sales['date'])
    written = write_state(out, start_state(sales[sales['date'] < start], cfg), '')
    print(f'block {k}: sales {full["date"].min().date()} .. {end.date()}, {len(full):,} rows; '
          f'rows before {start.date()} identical to {ref_dir}/sales.parquet; items identical')
    for p, rows in written:
        print(f'  {p.name}: {rows:,} rows')


def score_forecasts(k):
    """Day-ahead (origin = date) accuracy of the block's forecasts against the block's sales:
    q50 MAE and the share of days at or below each line; 25 Dec left out (closed)."""
    from forecast.data import load_config
    from forecast.metrics import QUANTILES, closed_mask, coverage, mae
    start, end, _ = block_dates(k)
    n = names(k)
    cfg = load_config(n['main'])
    out = ROOT / n['processed']
    fc = pd.read_parquet(out / 'forecasts.parquet')
    fc = fc[pd.to_datetime(fc['date']) == pd.to_datetime(fc['origin'])]
    s = pd.read_parquet(out / 'sales.parquet', columns=['date', 'store_id', 'item_id', 'units'])
    s['date'] = pd.to_datetime(s['date'])
    fc['date'] = pd.to_datetime(fc['date'])
    d = fc.merge(s, on=['date', 'store_id', 'item_id'], how='inner')
    d = d[(d['date'] >= start) & (d['date'] <= end) & ~closed_mask(d['date'], cfg).to_numpy()]
    row = dict(block=k, days=int(d['date'].nunique()), rows=len(d), q50_mae=mae(d['units'], d['q50']))
    row.update({f'{q}_coverage': coverage(d['units'], d[q]) for q in QUANTILES if q != 'q50'})
    pd.DataFrame([row]).to_csv(out / 'forecast_scores.csv', index=False)
    print(pd.DataFrame([row]).to_string(index=False))


def main(argv=None):
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['configs', 'finish', 'score'])
    ap.add_argument('--block', type=int, required=True, choices=[1, 2])
    a = ap.parse_args(argv)
    dict(configs=write_configs, finish=finish, score=score_forecasts)[a.step](a.block)


if __name__ == '__main__':
    main()
