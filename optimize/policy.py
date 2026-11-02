"""Replay adapter. Existing State/Plan and CSV schemas remain unchanged."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .audit import audit_plan
from .model import Plan, State, plan_orders


PIPELINE_COLUMNS = ['order_date', 'arrival_date', 'location_type', 'location_id', 'item_id', 'units']
ADJUSTMENT_COLUMNS = ['location_type', 'location_id', 'item_id', 'units', 'original_arrival_date', 'arrival_date', 'reason']


def reschedule_pipeline(pipeline, cfg):
    """Postpone committed arrivals to the next open date, never bring them forward.

    This is a replay convention for a pre-existing commitment. New decisions
    retain the model's calendar-day lead and closure constraints.
    """
    if not set(PIPELINE_COLUMNS) <= set(pipeline):
        raise ValueError('Pipeline columns do not match the agreed contract')
    result = pipeline[PIPELINE_COLUMNS].copy(deep=True)
    for column in ['order_date', 'arrival_date']:
        result[column] = pd.to_datetime(result[column]).dt.date
    result['location_type'] = result.location_type.str.lower()
    calendar = cfg['order_model']['calendar']
    adjustments = []
    for idx, row in result.iterrows():
        if row.location_type not in {'store', 'dc'}:
            raise ValueError('Invalid pipeline location type')
        closed = set(calendar['closed_store_month_days' if row.location_type == 'store' else 'closed_dc_month_days'])
        arrival = row.arrival_date
        for _ in range(367):
            if arrival.strftime('%m-%d') not in closed:
                break
            arrival += timedelta(days=1)
        else:
            raise ValueError('Calendar has no reachable open arrival date')
        if arrival != row.arrival_date:
            result.at[idx, 'arrival_date'] = arrival
            adjustments.append(dict(location_type=row.location_type, location_id=row.location_id,
                                    item_id=row.item_id, units=row.units,
                                    original_arrival_date=row.arrival_date, arrival_date=arrival,
                                    reason='committed_arrival_on_closure'))
    return result, pd.DataFrame(adjustments, columns=ADJUSTMENT_COLUMNS)


@dataclass
class Decision:
    """Only today's firm rows may be executed. Full Plan is for inspection/L3.

    L4 must adopt effective_pipeline before processing today's receipts. No
    supplier order is implicit in store_orders; both tables must be consumed.
    """
    date: date
    store_orders: pd.DataFrame
    dc_orders: pd.DataFrame
    deliveries: pd.DataFrame
    effective_pipeline: pd.DataFrame
    pipeline_adjustments: pd.DataFrame
    plan: Plan
    audit: dict


class OptimizerPolicy:
    """One instance per replay. Optional prior plan is a hint, never a commitment."""

    def __init__(self, initial_plan=None):
        self.last_plan = initial_plan

    def decide(self, state, fc, cfg):
        origin = pd.Timestamp(state.date).date()
        pipeline, adjustments = reschedule_pipeline(state.pipeline, cfg)
        effective = State(origin, state.store_stock.copy(deep=True),
                          state.dc_stock.copy(deep=True), pipeline)
        config = deepcopy(cfg)
        if self.last_plan is not None:
            config['order_model']['warm_start_plan'] = {
                'orders': self.last_plan.orders, 'dc_orders': self.last_plan.dc_orders}
        forecasts = fc.loc[pd.to_datetime(fc.origin).dt.date == origin].copy()
        plan = plan_orders(effective, forecasts, config)
        if not plan.stats.get('incumbent_validated', False):
            raise RuntimeError(f'No validated executable plan for {origin}: {plan.status}')
        audit = audit_plan(effective, forecasts, config, plan)

        def firm(frame):
            result = frame.loc[pd.to_datetime(frame.date).dt.date == origin].copy()
            result['date'] = origin
            return result.reset_index(drop=True)

        decision = Decision(origin, firm(plan.orders), firm(plan.dc_orders), firm(plan.deliveries),
                            pipeline, adjustments, plan, audit)
        self.last_plan = plan
        return decision
