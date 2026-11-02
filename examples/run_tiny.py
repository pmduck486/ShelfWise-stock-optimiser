"""Run the hand-checked three-product checkpoint: python -m examples.run_tiny."""

import json
from pathlib import Path

from examples.fixture import load_tiny
from optimize import plan_orders


def main():
    state, forecasts, cfg = load_tiny()
    plan = plan_orders(state, forecasts, cfg)
    output = Path(__file__).resolve().parents[1]/'outputs'/'tiny'
    output.mkdir(parents=True, exist_ok=True)
    stats = dict(plan.stats)
    (output/'cbc.log').write_text(stats.pop('solver_log'))
    (output/'stats.json').write_text(json.dumps(stats, indent=2, allow_nan=False)+'\n')
    if not stats['incumbent_validated']:
        raise RuntimeError(f'No CSV plan exported: {plan.status}')
    plan.orders.to_csv(output/'order_plan.csv', index=False)
    plan.dc_orders.to_csv(output/'dc_orders.csv', index=False)
    plan.deliveries.to_csv(output/'deliveries.csv', index=False)
    print('Firm store decisions:')
    print(plan.orders[plan.orders.date == state.date].to_string(index=False))
    print(f'Objective = {plan.objective:.6f}; expected hand calculation = 0.32')
    print(f'Outputs: {output}')


if __name__ == '__main__':
    main()
