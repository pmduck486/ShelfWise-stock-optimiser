"""Build the handoff evidence report from completed runs; never rerun a solver."""

import json
from pathlib import Path
import re


def main():
    root=Path(__file__).resolve().parents[1]
    tests=(root/'outputs'/'final_tests.log').read_text()
    match=re.search(r'(\d+) passed',tests)
    if not match:raise ValueError('No passing final test result')
    names=[('Small Gate 1','validation_small'),('Full production sample','replay_production'),
           ('Full test window','replay_full_window')]
    runs=[]
    for label,name in names:
        p=root/'outputs'/name
        summary=json.loads((p/'summary.json').read_text())
        audit=json.loads((p/'execution_audit.json').read_text())
        lp=json.loads((p/'fixed_integer_lp.json').read_text())
        if not (summary['all_plan_audits_passed'] and summary['independent_execution_audit_passed'] and audit['passed']):
            raise ValueError('Cannot publish a failed validation: '+name)
        runs.append((label,name,summary,audit,lp))
    rows=[]
    for label,name,s,a,lp in runs:
        rows.append(f'| {label} | {s["days"]} | {s["solver_settings"]["time_limit_s"]} s | {100*s["unit_fill_rate"]:.3f}% | {s["lost_units"]:,.2f} | {s["cash_cost"]:,.2f} | {100*s["max_gap"]:.3f}% |')
    _,_,s,a,lp=runs[-1]
    import pandas as pd
    daily=pd.read_csv(root/'outputs'/'replay_full_window'/'daily_metrics.csv')
    christmas=daily[daily.date=='2015-12-25'].iloc[0]
    text=f'''# L2 final validation evidence

1 November 2026. **{match[1]} tests pass.** These are saved completed runs on
the supplied M5 sales and **stub** forecasts. Real forecasts have not been
received, and the sealed holdout has not been read.

## Completed runs

| Run | Calendar days | Search budget/day | Observed-unit fill | Lost units | Cash cost | Maximum planning gap |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

Cash is in the documented M5 USD-equivalent reference-price units. It includes
purchases once, truck/order charges, closing-stock holding and incremental
disposal; preference penalties are excluded. Lost-sales penalty is reported
separately in each summary. These runs have different product/day/solve-budget
scopes and are **not comparative policy results**.

The full production sample starts without a prior CSV plan, exercising the new
forecast replenishment hint. The complete test-window run uses the historical
checkpoint-4 full plan only as an optional initial hint, then replans daily.
Only firm decisions execute. Production remains at 60 seconds and a 1% target;
the 15-second integration run is explicitly reduced-budget validation. The target
is a stopping condition, not an optimality guarantee. No result is relabelled
optimal merely because a gap prints as zero.

## Complete-window execution

- Dates: {s['start']} through {s['end']}, all **{s['days']} calendar dates**,
  **{s['open_days']} open service dates**.
- Observed open-day demand proxy: {s['actual_demand']:,.2f} units; served
  {s['served_units']:,.2f}; lost {s['lost_units']:,.2f}.
- Expected-disposal proxy: {s['waste_proxy_units']:,.2f} units. This is not
  measured expiry or an exact cohort model.
- Supplier purchases: {s['supplier_units']:,.2f} units; final store/DC stock
  plus both pipeline stages: {s['ending_stock_and_pipeline_units']:,.2f} units.
- All daily forecast-plan audits pass; {s['physical_sales_checks']:,} physical
  forecast-sales checks; future withholding {s['planned_future_withholding']:g}.
- Separately reconstructed execution: {a['store_balance_checks']:,} store
  balances, {a['dc_balance_checks']:,} DC balances, {a['firm_pack_checks']:,}
  firm pack rows, {a['truck_checks']:,} truck rows, {a['cash_checks']} cash and
  {a['conservation_checks']} chain-conservation checks. Maximum reconstruction
  residual is {a['max_residual']:.3g}; tolerance is 0.0001.
- Status counts: {json.dumps(s['status_counts'],sort_keys=True)}.
  Explicit start-fallback days: {s['fallback_days']}; unavailable gaps:
  {s['gaps_unavailable']}.
- Planning gaps: median {100*s['median_gap']:.3f}%, maximum
  {100*s['max_gap']:.3f}%. These are objective-bound gaps, not forecast errors
  or a claim about a percentage of cash left on the table.
- Daily elapsed time: median {s['median_daily_seconds']:.2f} s, maximum
  {s['max_daily_seconds']:.2f} s; rolling phase {s['elapsed_seconds']:.2f} s.
  File writing/audits add overhead; runs overlapped on this machine.

The independent auditor starts from supplied initial inventories and source
sales/forecasts, reschedules initial commitments, reconstructs every receipt
from firm order dates/leads, and compares all ledger rows, cash and final
pipeline. A regression deliberately corrupts saved cash and confirms rejection.
The fixed-integer LP check fixes all {lp['fixed_integer_count']:,} first-day
integers in a copy and solves the conditional LP. Tests also perturb a shelf
RHS and verify its dual prediction without changing the original MILP.

## Christmas check

25 December remains an executed calendar date. Observed demand used is
{christmas.actual_demand:g}, ignored closed-day raw sales are
{christmas.ignored_closed_day_recorded_units:g}, store receipts are
{christmas.store_receipts:g}, and cash is {christmas.cash_cost:.4f}. Holding and
disposal remain counted; that date is excluded from service means. A separate
committed-delivery regression postpones 25 December arrivals to 26 December and
checks the adjustment log. The calendar disposal-proxy test confirms that a
closure contributes zero selling cover rather than extending shelf life.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m examples.run_replay --profile small --end-date 2015-11-07 --output-dir outputs/new_small
python -m examples.run_replay --end-date 2015-11-03 --output-dir outputs/new_production
python -m examples.run_replay --config configs/validation_fast.yaml --warm-start-dir outputs/full --output-dir outputs/new_full_window
```

These are the commands used for the corresponding saved profiles. A fresh run
can return another feasible incumbent because CBC is time-limited and runtime
depends on the host. Counts, constraints, schemas and audits are the validation
criteria; identical objective values are not promised. The data snapshot contains
100,800 sales rows and all 470,400 stub forecasts with their original schemas.
The frozen calibration digest is
`6a5de7b154b7964dcdc0649b24a00c7ce1d15fc15153650d053ca351e769e72b`.

## Earlier attempts and repairs

Earlier checkpoint-4 cold/monolithic attempts and all original baseline evidence
remain under outputs/experiments and the checkpoint documents. The initial new
replay exposed repeated pipeline scans and poor zero-order starts under a short
budget. Partial runs are retained rather than claimed successful.

The first replenishment hint was rejected where future activation big-M was
less than the minimum order. The corrected hint leaves those orders at zero,
as the existing constraints require, and is separately validated before CBC.
Tests cover that edge case, duplicate commitments, forced no-incumbent fallback,
real supplier commitments and future-sales poisoning. The hint is a deterministic
forecast-only construction with no fitted replay coefficients. No demand,
service-level, cost or business constraint was tuned to test outcomes.

## Delivery boundary

The L2 core and its execution interface are complete for handoff. HANDOFF.md
defines the Decision object L4 must consume. This validation does not establish
improvement over three baselines, replace L1's real forecasts, or finish L5's
dashboard/PPT. Expected disposal, unlimited backroom and uncensored-demand
assumptions remain explicit limitations. Exact age tracking and other optional
layers remain off. The coordinated holdout run remains a team action after freeze.
'''
    (root/'VALIDATION.md').write_text(text)
    print(root/'VALIDATION.md')


if __name__=='__main__':
    main()
