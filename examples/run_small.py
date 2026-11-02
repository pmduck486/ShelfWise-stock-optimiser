"""Run one seven-day small-case plan; this is not the L4 replay engine."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import yaml

from optimize import plan_orders
from optimize.audit import audit_plan
from optimize.parameters import load_case


def main(default_profile='small'):
    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['small', 'full'], default=default_profile)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--config', type=Path, default=root/'config.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--warm-start-dir',type=Path,
                        help='Optional previously exported order plan; used only as a validated CBC start')
    args = parser.parse_args()
    input_dir = args.input_dir or root/'data'/('checkpoint2' if args.profile == 'small' else 'checkpoint3')
    calibration_started = perf_counter()
    case = load_case(input_dir, args.config, profile=args.profile)
    calibration_seconds = perf_counter()-calibration_started
    cfg = case.parameters.config
    if args.warm_start_dir is not None:
        cfg['order_model']['warm_start_plan'] = dict(
            orders=pd.read_csv(args.warm_start_dir/'order_plan.csv'),
            dc_orders=pd.read_csv(args.warm_start_dir/'dc_orders.csv'))
    print(f'Loaded {args.profile} profile: {case.state.store_stock.store_id.nunique()} stores, '
          f'{case.state.store_stock.item_id.nunique()} items, origin {case.state.date}; '
          f'calibration {calibration_seconds:.3f}s', flush=True)
    plan = plan_orders(case.state, case.forecasts, cfg)
    output = args.output_dir or root/'outputs'/args.profile
    output.mkdir(parents=True, exist_ok=True)
    stats = dict(plan.stats)
    (output/'cbc.log').write_text(stats.pop('solver_log'))
    (output/'stats.json').write_text(json.dumps(stats, indent=2, allow_nan=False)+'\n')
    if not stats['incumbent_validated']:
        raise RuntimeError(f'No validated plan to export: {plan.status}; inspect status and CBC log')
    audit_started = perf_counter()
    audit = audit_plan(case.state, case.forecasts, cfg, plan)
    audit_seconds = perf_counter()-audit_started
    plan.orders.to_csv(output/'order_plan.csv', index=False)
    plan.dc_orders.to_csv(output/'dc_orders.csv', index=False)
    plan.deliveries.to_csv(output/'deliveries.csv', index=False)
    case.parameters.item_parameters.to_csv(output/'item_parameters.csv', index=False)
    case.parameters.store_parameters.to_csv(output/'store_parameters.csv', index=False)
    case.initialization_log.to_csv(output/'initialization_log.csv', index=False)
    pd.DataFrame(stats['store_inventory']).to_csv(output/'store_inventory.csv', index=False)
    pd.DataFrame(stats['dc_inventory']).to_csv(output/'dc_inventory.csv', index=False)
    pd.DataFrame(stats['store_flows']).to_csv(output/'store_flows.csv', index=False)
    pd.DataFrame(stats['future_sales_rationing'], columns=['date','store_id','item_id','withheld_units']).to_csv(output/'future_sales_rationing.csv', index=False)
    (output/'calibration_manifest.json').write_text(json.dumps(case.parameters.manifest, indent=2)+'\n')
    serializable_cfg = {**cfg,'order_model':{k:v for k,v in cfg['order_model'].items() if k != 'warm_start_plan'}}
    (output/'compiled_config.yaml').write_text(yaml.safe_dump(serializable_cfg, sort_keys=False))
    (output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    firm = plan.orders[plan.orders.date == case.state.date]
    firm_dc = plan.dc_orders[plan.dc_orders.date == case.state.date]
    firm_trucks = plan.deliveries[plan.deliveries.date == case.state.date]
    summary = dict(profile=args.profile, origin=case.state.date.isoformat(), stores=case.state.store_stock.store_id.nunique(),
                   items=case.state.store_stock.item_id.nunique(), horizon_days=cfg['order_model']['horizon_days'],
                   status=plan.status, objective=plan.objective, n_vars=stats['n_vars'], n_int=stats['n_int'],
                   n_constraints=stats['n_constraints'], build_seconds=stats['build_seconds'],
                   solve_seconds=stats['solve_seconds'], mip_gap=stats['mip_gap'],
                   fulfillment_policy=stats['fulfillment_policy'],
                   warehouse_reserve_policy=stats['warehouse_reserve_policy'],
                   decomposition=stats['decomposition'],
                   used_start_fallback=stats['used_start_fallback'],
                   calibration_seconds=calibration_seconds, audit_seconds=audit_seconds,
                   elapsed_seconds=perf_counter()-started,
                   firm_store_cases=float(firm.cases.sum()), firm_store_units=float(firm.units.sum()),
                   firm_trucks=float(firm_trucks.trucks.sum()), firm_supplier_pallets=float(firm_dc.pallets.sum()),
                   planned_supplier_pallets_all_dates=float(plan.dc_orders.pallets.sum()),
                   initialized_clipped_units=float(case.initialization_log.clipped_units.sum()),
                   terminal_store_shortfall_units=sum(stats['terminal_store_shortfall'].values()),
                   terminal_dc_shortfall_units=sum(stats['terminal_dc_shortfall'].values()),
                   future_sales_rationing_units=stats['future_sales_rationing_units'],
                   planned_waste_proxy_units=stats['waste_units_total'],
                   overflow_unit_mornings=stats['overflow_unit_mornings'],
                   planned_lost_sales_units=stats['lost_sales_total'], independent_audit_passed=audit['passed'],
                   scope='One 7-day plan on stub forecasts; no realized replay or holdout results.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))
    if args.profile == 'small':
        print('Nonzero firm store orders:')
        print(firm[firm.cases > 0].to_string(index=False))
    else:
        print('Firm store totals:')
        print(firm.groupby('store_id')[['cases','units']].sum().to_string())
    print('Objective components (model penalties, not a replay cash-cost KPI):')
    print(json.dumps(stats['objective_breakdown'], indent=2))
    print(f'Outputs: {output}')


if __name__ == '__main__':
    main()
