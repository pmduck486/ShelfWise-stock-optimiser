"""Validate daily L2 decisions on real sales; not the L4 baseline comparison."""

import argparse
import json
from pathlib import Path
import platform

import pandas as pd
import pulp
import yaml

from optimize import OptimizerPolicy, Plan
from optimize.parameters import load_case
from optimize.replay import run_replay
from optimize.replay_audit import audit_replay
from optimize.sensitivity import fixed_integer_lp


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=root/'data'/'validation')
    parser.add_argument('--config',type=Path,default=root/'config.yaml')
    parser.add_argument('--forecast-file',type=Path,help='Override stub with same-column real forecasts')
    parser.add_argument('--profile',choices=['small','full'],default='full')
    parser.add_argument('--end-date',help='Default: configured test_end; holdout prohibited')
    parser.add_argument('--output-dir',type=Path,default=root/'outputs'/'validation')
    parser.add_argument('--warm-start-dir',type=Path)
    args=parser.parse_args()
    case=load_case(args.input_dir,args.config,profile=args.profile,forecast_path=args.forecast_file)
    cfg=case.parameters.config
    fc_path=args.forecast_file or Path(case.source_files['forecasts'])
    forecasts=pd.read_parquet(fc_path)
    sales=pd.read_parquet(case.source_files['sales'],columns=['date','store_id','item_id','units'])
    initial=None
    if args.warm_start_dir:
        initial=Plan(pd.read_csv(args.warm_start_dir/'order_plan.csv'),
                     pd.read_csv(args.warm_start_dir/'dc_orders.csv'),pd.DataFrame(),0,'Hint',None)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    case.initialization_log.to_csv(args.output_dir/'initialization_log.csv',index=False)
    (args.output_dir/'calibration_manifest.json').write_text(json.dumps(case.parameters.manifest,indent=2)+'\n')
    (args.output_dir/'compiled_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (args.output_dir/'runtime_versions.json').write_text(json.dumps(dict(python=platform.python_version(),
         pandas=pd.__version__,pulp=pulp.__version__),indent=2)+'\n')
    policy=OptimizerPolicy(initial)
    summary,_=run_replay(case.state,forecasts,sales,cfg,
             end_date=args.end_date or cfg['data']['windows']['test_end'],
             output_dir=args.output_dir,policy=policy)
    execution_audit=audit_replay(case.state,forecasts,sales,cfg,args.output_dir)
    _,_,sensitivity=fixed_integer_lp(policy.last_plan,cfg)
    (args.output_dir/'fixed_integer_lp.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    summary['independent_execution_audit_passed']=execution_audit['passed']
    summary['fixed_integer_lp_validated']=True
    summary['initial_clipped_units']=float(case.initialization_log.clipped_units.sum())
    summary['forecast_source']=str(fc_path)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
