"""Checks for dashboard/export_data.py output.

Run from RetailOps/dashboard:  ../../.venv/Scripts/python -m pytest test_export.py -q
(Exports first if public/data/summary.json is missing.)
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / 'public' / 'data'


@pytest.fixture(scope='module', autouse=True)
def exported():
    if not (DATA / 'summary.json').exists():
        subprocess.run([sys.executable, str(HERE / 'export_data.py')], check=True, cwd=HERE)


def load(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def method(summary, period, mid):
    return next(m for m in summary['periods'][period]['methods'] if m['id'] == mid)


def test_files_exist_and_parse():
    for name in ('summary.json', 'calendar.json', 'items.json', 'shadow.json', 'forecast.json'):
        assert isinstance(load(name), dict), name
    expected_days = {'test': 56, 'b1': 49, 'b2': 49}
    for period, n in expected_days.items():
        assert len(list((DATA / 'day' / period).glob('*.json'))) == n
        assert len(list((DATA / 'series' / period).glob('*.json'))) == 10


def test_total_size_under_25_mb():
    total = sum(f.stat().st_size for f in DATA.rglob('*.json'))
    assert total < 25e6, total


def test_headline_numbers_match_results_tables():
    s = load('summary.json')
    rt = pd.read_csv(ROOT / 'outputs/replay/store_ilp_2026-11-02/results_table.csv').set_index('method')
    ilp = method(s, 'test', 'store_ilp')
    assert round(ilp['fill'], 4) == 0.9487
    assert ilp['fill'] == pytest.approx(rt.loc['store_ilp', 'fill_rate'], abs=1e-6)
    assert round(ilp['total']) == 934088
    assert ilp['total'] == pytest.approx(rt.loc['store_ilp', 'net_cost'] + rt.loc['store_ilp', 'lost_sales_penalty'], abs=0.1)
    assert round(ilp['waste']) == 8043
    # best simple rule on the test window is top-up to our forecast; README says -7.7%
    assert s['periods']['test']['best_simple'] == 'topup_forecast'
    assert round(ilp['vs_best_simple'], 1) == -7.7


@pytest.mark.parametrize('period,folder,fill,total,vs', [
    ('b1', 'holdout_block1', 94.14, 855095, -5.2),
    ('b2', 'holdout_block2', 95.40, 887087, -7.1),
])
def test_holdout_numbers(period, folder, fill, total, vs):
    s = load('summary.json')
    rt = pd.read_csv(ROOT / f'outputs/replay_holdout/{folder}/results_table.csv').set_index('method')
    ilp = method(s, period, 'store_ilp')
    assert round(ilp['fill'] * 100, 2) == fill
    assert abs(ilp['total'] - total) <= 1
    assert ilp['fill'] == pytest.approx(rt.loc['store_ilp', 'fill_rate'], abs=1e-6)
    assert round(ilp['vs_best_simple'], 1) == vs
    assert all(m['audit_passed'] and m['rules_passed'] for m in s['periods'][period]['methods'])


def test_layer_numbers_are_in_the_decisions_log():
    text = (ROOT / 'docs/decisions.md').read_text(encoding='utf-8')
    sys.path.insert(0, str(HERE))
    import export_data
    for row in export_data.LAYERS:
        for k in ('fill_txt', 'lost_txt', 'waste_txt', 'stock_txt', 'total_txt'):
            assert row[k] in text, (row['id'], k, row[k])
    core = load('summary.json')['layers']['rows'][0]
    ilp = method(load('summary.json'), 'test', 'store_ilp')
    assert core['fill'] == round(ilp['fill'] * 100, 2)
    assert core['total'] == round(ilp['total'])


def test_calendar_adds_up_to_the_headline():
    cal = load('calendar.json')
    s = load('summary.json')
    for period in ('test', 'b1', 'b2'):
        days = cal['periods'][period]['days']
        served = sum(d['served'] for d in days)
        demand = sum(d['demand'] for d in days)
        assert served / demand == pytest.approx(method(s, period, 'store_ilp')['fill'], abs=2e-5)
        assert sum(d['lost'] for d in days) == pytest.approx(method(s, period, 'store_ilp')['lost'], abs=5)
    test_days = cal['periods']['test']['days']
    assert test_days[0]['date'] == '2015-11-01' and test_days[-1]['date'] == '2015-12-26'
    xmas = next(d for d in test_days if d['date'] == '2015-12-25')
    assert xmas['closed'] and xmas['demand'] == 0
    assert any(e['name'] == 'Thanksgiving' for d in test_days for e in d['events'])
    assert not next(d for d in test_days if d['date'] == '2015-12-24')['can_order']


def test_day_files_schema_and_self_checks():
    cal = load('calendar.json')
    chk = cal['periods']['test']['self_check']
    assert chk['matched'] == chk['total'] > 0
    shadow = load('shadow.json')
    for period in ('test', 'b1', 'b2'):
        c = shadow['periods'][period]['ilp_term_check']
        assert c['matched'] / c['total'] > 0.999
    d = load('day/b1/2016-01-08.json')
    assert len(d['rows']) == 1200 and all(len(r) == len(d['cols']) for r in d['rows'])
    assert d['floor_on'] and d['spoil_in_ilp']
    t = load('day/test/2015-11-01.json')
    assert not t['floor_on'] and t['layers'] == []
    ledger_orders = pd.read_csv(ROOT / 'outputs/replay/store_ilp_2026-11-02/store_ilp/order_plan.csv')
    first = ledger_orders[ledger_orders.date == '2015-11-01']
    assert sum(r[12] for r in t['rows']) == int(first.cases.sum())


def test_forecast_scores():
    f = load('forecast.json')
    assert round(f['holdout']['b1']['mae'], 2) == 3.52 and round(f['holdout']['b2']['mae'], 2) == 3.63
    assert round(f['holdout']['b1']['cover_q90'], 1) == 90.6 or round(f['holdout']['b1']['cover_q90'], 1) == 90.7
    assert round(f['folds']['fold1_festive']['model']['mae'], 3) == 3.550
    assert round(f['folds']['fold2_main']['model']['mae'], 3) == 3.817
    for fold in f['folds'].values():
        assert fold['model']['mae'] < fold['stub']['mae'] < fold['naive']['mae']
