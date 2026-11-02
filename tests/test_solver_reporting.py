import pytest

from optimize.model import _solver_gap


def test_rounded_cbc_gap_is_recomputed_from_objective_and_bound():
    log = 'Objective value: 37730.06360002\nLower bound: 37729.793\nGap: 0.00\n'
    report = _solver_gap(log, 'GapSatisfied-Feasible', 37730.06359907496, 0.0001)
    assert report['cbc_reported_gap'] == 0
    assert report['mip_gap'] == pytest.approx(0.000007171975055)
    assert report['mip_gap'] > 0
    assert report['best_bound'] == pytest.approx(37729.793)


def test_proven_optimum_without_printed_bound_has_zero_gap():
    report = _solver_gap('Result - Optimal solution found', 'Optimal', 0.32, 0.0001)
    assert report['mip_gap'] == 0
    assert report['best_bound'] is None


def test_no_incumbent_does_not_invent_a_zero_gap():
    report = _solver_gap('Gap: 0.00\n', 'TimeLimit-NoIncumbent', None, 0.0001)
    assert report['mip_gap'] is None
