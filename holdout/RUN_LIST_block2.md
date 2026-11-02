# Holdout block 2: run list (written 2026-11-02 ~11:30, before any block 2 data is read)

User asked at 11:23 to run block 2. Same procedure, rules and frozen code as block 1
(`RUN_LIST_block1.md`); only the dates move. Block 3 (3 Apr – 22 May 2016) stays sealed.

## Period
- Block 2: **14 Feb – 2 Apr 2016** (49 days, 7 weeks). Sales are read up to 2 Apr only.

## What is frozen (same as block 1)
- Products, perishables and shelf lives unchanged. Code checks that items.csv matches the development
  one, and that every sales row before 14 Feb matches block 1's sales file exactly.
- Forecaster: same settings and calibration numbers, retrained once on train_start .. 13 Feb
  (cutoff = block start), then frozen. Forecasts for 49 mornings (today + tomorrow), rule R2.
- Order model and replay: config.yaml as frozen (store-only; service floor 0.9 and spoilage cap on,
  stock cap off). Its averages and prices are recalculated at the block start by the same rule.
- Only harness change since block 1: `holdout/make_block.py finish` now compares block k's old rows with
  block k-1's sales file (block 1's check used the development file). No model or setting changed.

## Starting stock
- Same rule as 1 Nov and block 1: 3 days of each store-item's average daily sales over the 28 days before
  14 Feb. The same for every method, with nothing on the way. (The block does not carry on from block 1's
  end stock: every block starts fresh, like the test window.)

## Methods
1. last_week (habit)  2. topup_avg  3. topup_forecast  4. store_model  5. **store_ilp (main model)**

## What is reported, as it comes out
- Per method: fill rate, lost units, waste units, average store stock value, net cost, lost-sale penalty,
  net cost + penalty, rules + audit passed, run time; store_ilp shadow price summary.
- Forecaster on block 2: q50 MAE and q80/q90/q95 coverage.
- Side by side with block 1 and the test window. No method added, dropped or re-tuned afterwards.
