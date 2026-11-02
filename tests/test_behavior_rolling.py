from pathlib import Path

import pytest

from examples.check_behavior import run_synthetic


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def rolling_results():
    cfg = ROOT/'configs'/'checkpoint4_candidate.yaml'
    old_frame,old,_ = run_synthetic(cfg,legacy=True,surprise=False)
    new_frame,new,_ = run_synthetic(cfg,legacy=False,surprise=False)
    return old_frame,old,new_frame,new


def test_constant_demand_replanning_orders_real_supply_and_keeps_dc_cover(rolling_results):
    old_frame,old,new_frame,new = rolling_results
    assert new['days'] >= 2*5
    assert new['lost_units'] == pytest.approx(0)
    assert new['supplier_units'] > 0
    assert new['dc_protection_deficit_unit_days'] == pytest.approx(0)
    assert old['dc_protection_deficit_unit_days'] > new['dc_protection_deficit_unit_days']
    assert new_frame[['store_closing_units','dc_closing_units']].min().min() >= 0
    assert new_frame.planned_future_withholding.sum() == pytest.approx(0)


def test_unforecast_surprise_is_reported_as_real_lost_demand():
    cfg = ROOT/'configs'/'checkpoint4_candidate.yaml'
    frame,summary,_ = run_synthetic(cfg,legacy=False,surprise=True)
    # Warehouse coverage is not a guarantee against an unforecast store spike.
    assert summary['lost_units'] == pytest.approx(2)
    assert (frame.actual_demand-frame.served).sum() == pytest.approx(summary['lost_units'])
    assert frame[['store_closing_units','dc_closing_units']].min().min() >= 0
