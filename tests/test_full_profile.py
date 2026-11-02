from pathlib import Path

import pytest

from optimize.parameters import load_full_case


ROOT = Path(__file__).resolve().parents[1]


def test_full_profile_loads_all_stores_items_and_one_forecast_origin():
    case = load_full_case(ROOT/'data'/'checkpoint3', ROOT/'config.yaml')
    assert case.state.store_stock.shape[0] == 1200
    assert case.state.dc_stock.shape[0] == 240
    assert case.forecasts.shape[0] == 8400
    assert case.forecasts.origin.nunique() == 1
    assert case.forecasts.date.nunique() == 7
    assert len(case.parameters.manifest['stores']) == 10
    assert len(case.parameters.manifest['items']) == 120
    assert case.parameters.manifest['average_end'] == '2015-10-25'
    assert not case.parameters.item_parameters.price.isna().any()
    assert case.parameters.item_parameters.price.min() > 0


def test_full_profile_keeps_declared_service_level_precedence():
    case = load_full_case(ROOT/'data'/'checkpoint3', ROOT/'config.yaml')
    for row in case.parameters.item_parameters.itertuples(index=False):
        expected = (0.95 if row.is_perishable or row.rank <= 30 else
                    0.80 if row.global_avg_daily_units < 5 else 0.90)
        assert row.service_level == expected


def test_full_profile_rejects_small_state_files(tmp_path):
    small = ROOT/'data'/'checkpoint2'
    for kind in ['store_stock', 'dc_stock', 'pipeline']:
        (tmp_path/f'state_start_{kind}.csv').write_bytes((small/f'state_start_{kind}_small.csv').read_bytes())
    with pytest.raises(ValueError, match='full initial-state dimensions'):
        load_full_case(tmp_path, ROOT/'config.yaml')
