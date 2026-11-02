"""The solver's head start (warm start) must actually reach CBC on every OS.

Guards the Windows bug found 2026-10-31: CBC could not open PuLP's absolute
temp .mst path, ignored the start and searched from scratch, with no error.
optimize/cbc.py fixes it; these tests read CBC's own log to prove the start
was loaded, for the single-model path and the per-DC path.
"""

from examples.fixture import load_tiny
from optimize import plan_orders
from tests.test_dc_decomposition import two_dc_case


def _read_start(log):
    return 'mipstart values read' in log.lower()


def test_warm_start_is_read_by_cbc_single_model():
    state, fc, cfg = load_tiny()
    cfg['order_model']['solver'].update(warm_start='verified_plan', start_plan_fallback=True)
    plan = plan_orders(state, fc, cfg)
    assert plan.stats['verified_start_source'] is not None
    assert _read_start(plan.stats['solver_log']), plan.stats['solver_log'][-2000:]


def test_warm_start_is_read_by_cbc_in_every_dc_block():
    state, fc, cfg = two_dc_case()
    cfg['order_model']['solver'].update(decompose_by_dc=True, parallel_time_reserve_s=1,
                                        warm_start='verified_plan', start_plan_fallback=True)
    plan = plan_orders(state, fc, cfg)
    log = plan.stats['solver_log']
    blocks = [b for b in log.split('=== ') if b.strip()]
    assert len(blocks) == 2
    assert all(_read_start(b) for b in blocks), log[-2000:]
