"""Continuous service-priority repair; no extra integer decision variables."""

import math


def service_priority_weights(horizon, base, safety, terminal_store, margin):
    """Sufficient dominance weights, in money per lost sales unit.

    Last-day lost-sale cost exceeds the store terminal reward, and each earlier
    day's cost exceeds the next day's cost by more than one safety-slack charge.
    With fixed shipments, serving available demand cannot then worsen the score.
    """
    if horizon <= 0 or any(not math.isfinite(x) or x < 0 for x in [base,safety,terminal_store,margin]):
        raise ValueError('Invalid service-priority parameters')
    if base <= terminal_store or margin <= 0:
        raise ValueError('Last-day lost-sale penalty must exceed terminal store penalty; margin must be positive')
    return [base+(horizon-1-t)*(safety+margin) for t in range(horizon)]


def work_conserving_path(opening, arrivals, demands, original_disposals):
    """Serve first, then preserve only the original disposal that is still possible.

    For an originally feasible path, closing inventory never increases. Thus
    upper storage constraints stay feasible, and disposal never increases.
    """
    if not len(arrivals) == len(demands) == len(original_disposals):
        raise ValueError('Path arrays must have identical lengths')
    stock = float(opening)
    result = []
    for arrived,demand,disposal in zip(arrivals,demands,original_disposals):
        morning = stock+max(float(arrived),0)
        lost = max(float(demand)-morning,0)
        remaining = max(morning-float(demand),0)
        wasted = min(max(float(disposal),0),remaining)
        stock = remaining-wasted
        result.append(dict(morning=morning,lost_sales=lost,waste=wasted,closing=stock))
    return result
