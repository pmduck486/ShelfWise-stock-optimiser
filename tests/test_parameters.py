from copy import deepcopy
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from optimize.parameters import compile_parameters, initialize_state, load_small_case


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def calibration_data():
    cfg = yaml.safe_load((ROOT/'config.yaml').read_text())
    rows = []
    for day in pd.date_range('2015-09-28', '2015-10-25'):
        for store, units in [('CA_1', 10), ('TX_1', 100)]:
            rows.append(dict(date=day, store_id=store, item_id='ITEM', units=units, price=2.0))
    items = pd.DataFrame([dict(item_id='ITEM', rank=40, avg_daily_units=10,
                               is_perishable=False, shelf_life_days=float('nan'))])
    return pd.DataFrame(rows), items, cfg


def test_calibration_uses_each_store_average_and_explicit_unit_costs(calibration_data):
    sales, items, cfg = calibration_data
    bundle = compile_parameters(sales, items, cfg, ['CA_1', 'TX_1'], ['ITEM'])
    params = bundle.config['order_model']['parameters']
    assert params['store_averages']['CA_1']['ITEM'] == pytest.approx(10)
    assert params['store_averages']['TX_1']['ITEM'] == pytest.approx(100)
    p = params['items']['ITEM']
    assert p['units_per_case'] == 12
    assert p['buy_cost'] == pytest.approx(1.7)
    assert p['holding_cost_per_unit_day'] == pytest.approx(1.7*0.25/365)
    assert params['dc_budgets']['DC_WEST'] == pytest.approx(1.25*7*10*1.7)
    assert params['dc_budgets']['DC_EAST'] == pytest.approx(1.25*7*100*1.7)
    caps = bundle.store_parameters.set_index('store_id').shelf_capacity_units
    assert caps['TX_1'] == pytest.approx(1000)
    assert caps['CA_1'] == pytest.approx(100)
    assert 'parameters' not in cfg['order_model']


def test_future_poison_cannot_change_frozen_parameters(calibration_data):
    sales, items, cfg = calibration_data
    original = compile_parameters(sales, items, cfg, ['CA_1', 'TX_1'], ['ITEM'])
    future = sales.iloc[:2].copy()
    future['date'] = pd.Timestamp('2015-12-01')
    future['units'] = 10**8
    future['price'] = 10**7
    poisoned = compile_parameters(pd.concat([sales, future]), items, cfg, ['CA_1', 'TX_1'], ['ITEM'])
    assert poisoned.config['order_model']['parameters'] == original.config['order_model']['parameters']
    assert poisoned.manifest['calibration_sha256'] == original.manifest['calibration_sha256']
    assert original.manifest['average_end'] == '2015-10-25'
    assert original.manifest['average_start'] == '2015-09-28'


def test_metadata_service_precedence_stays_in_sync_with_l1(calibration_data):
    sales, base, cfg = calibration_data
    scenarios = [(True, 40, 1, 0.95), (False, 1, 1, 0.95),
                 (False, 31, 4.9, 0.80), (False, 31, 5, 0.90)]
    for perishable, rank, average, expected in scenarios:
        items = base.copy()
        items['avg_daily_units'] = items.avg_daily_units.astype(float)
        items.loc[0, ['rank', 'avg_daily_units', 'is_perishable', 'shelf_life_days']] = [
            rank, average, perishable, 2 if perishable else float('nan')]
        bundle = compile_parameters(sales, items, cfg, ['CA_1', 'TX_1'], ['ITEM'])
        assert bundle.config['order_model']['parameters']['items']['ITEM']['service_level'] == expected


def test_missing_calendar_sales_are_zero_not_removed_from_denominator(calibration_data):
    sales, items, cfg = calibration_data
    sales = sales[~((sales.store_id == 'CA_1') & (sales.date == pd.Timestamp('2015-09-28')))]
    bundle = compile_parameters(sales, items, cfg, ['CA_1', 'TX_1'], ['ITEM'])
    assert bundle.config['order_model']['parameters']['store_averages']['CA_1']['ITEM'] == pytest.approx(270/28)
    assert bundle.manifest['missing_calendar_rows'] == 1


def test_duplicate_pretest_observations_are_rejected(calibration_data):
    sales, items, cfg = calibration_data
    with pytest.raises(ValueError, match='Duplicate'):
        compile_parameters(pd.concat([sales, sales.iloc[:1]]), items, cfg, ['CA_1', 'TX_1'], ['ITEM'])


def test_missing_reference_price_has_no_future_fallback(calibration_data):
    sales, items, cfg = calibration_data
    sales['price'] = float('nan')
    with pytest.raises(ValueError, match='price'):
        compile_parameters(sales, items, cfg, ['CA_1', 'TX_1'], ['ITEM'])


def test_initialization_logs_clipping_without_waste_or_daily_mutation(calibration_data):
    sales, items, cfg = calibration_data
    bundle = compile_parameters(sales, items, cfg, ['CA_1', 'TX_1'], ['ITEM'])
    stores = pd.DataFrame([dict(date='2015-11-01', store_id='CA_1', item_id='ITEM', units=120),
                           dict(date='2015-11-01', store_id='TX_1', item_id='ITEM', units=300)])
    dcs = pd.DataFrame([dict(date='2015-11-01', dc_id=d, item_id='ITEM', units=1000)
                        for d in ['DC_WEST', 'DC_EAST']])
    pipe = pd.DataFrame(columns=['order_date', 'arrival_date', 'location_type', 'location_id', 'item_id', 'units'])
    before = stores.copy(deep=True)
    state, log = initialize_state(stores, dcs, pipe, bundle)
    assert state.date == date(2015, 11, 1)
    assert state.store_stock.set_index('store_id').loc['CA_1', 'units'] == pytest.approx(100)
    assert log.clipped_units.sum() == pytest.approx(20)
    assert set(log.accounting_category) == {'initial_state_adjustment'}
    pd.testing.assert_frame_equal(stores, before)
    pd.testing.assert_frame_equal(state.pipeline, pipe)


def test_supplied_small_case_dimensions_and_global_rank_service_levels():
    case = load_small_case(ROOT/'data'/'checkpoint2', ROOT/'config.yaml')
    assert case.state.store_stock.shape[0] == 20
    assert case.state.dc_stock.shape[0] == 20
    assert case.forecasts.shape[0] == 140
    assert case.forecasts.origin.nunique() == 1
    assert case.state.date == date(2015, 11, 1)
    assert set(case.parameters.item_parameters.service_level) == {0.95}
    assert case.parameters.manifest['average_end'] < '2015-11-01'
