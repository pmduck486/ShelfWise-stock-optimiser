# Holdout block 1: run list (written 2026-11-02, before any holdout data is read)

User's plan (2026-11-02): the sealed holdout (27 Dec 2015 – 22 May 2016, 148 days) is split into
3 blocks of about 7 weeks. Block 1 is run once now; blocks 2 and 3 stay sealed. Code and settings
are frozen: nothing changes after this point unless something breaks (a crash or a failed rule check),
and then the user is told first. Block 2 will be run later with the same frozen model.

## Period
- Block 1: **27 Dec 2015 – 13 Feb 2016** (49 days, 7 weeks). Only these days' sales are read
  (data_prep reads up to 13 Feb; from 14 Feb everything stays sealed).

## What is frozen
- Products, perishables and shelf lives: unchanged (chosen on sales before 1 Nov 2015). The block's
  items.csv must be identical to data/processed/items.csv, and every sales row before 27 Dec must be
  identical to data/processed/sales.parquet (checked in code, run stops otherwise).
- Forecaster: same features, settings and calibration numbers; retrained once on train_start .. 26 Dec
  (cutoff = block start), then frozen. Forecasts for 49 mornings (today + tomorrow), each using only
  sales two or more days old (rule R2).
- Order model and replay: config.yaml as it is now (store-only; service floor 0.9 and spoilage cap on,
  stock cap off; 10 s per daily store solve). The order model's averages and prices are recalculated
  at the block start by the same rule as at 1 Nov ("freeze at test_start").
- Only dates and folders differ, in config copies made by holdout/make_block.py
  (config_holdout_b1.yaml, config_holdout_b1_prepare.yaml, configs/validation_fast_holdout_b1.yaml).

## Starting stock (user's choice, 2026-11-02 10:10)
- Same rule as 1 Nov: each store-item starts with 3 days of its average daily sales over the 28 days
  before 27 Dec (the 28 days include closed 25 Dec). Same start for every method (R11). Nothing on the way.

## Methods (all five, one run each)
1. last_week (habit)  2. topup_avg  3. topup_forecast  4. store_model (rule only)  5. **store_ilp (main model)**

## What is reported, as it comes out (no method added, dropped or re-tuned afterwards)
- Per method: fill rate, lost units, waste units, average store stock value, net cost, lost-sale
  penalty, net cost + penalty, rules + audit passed, run time; store_ilp shadow price summary.
- Forecaster on block 1: MAE of q50 and coverage of q80/q90/q95 (the test-window report's measures).
- Side by side with the 8-week test-window results (store_ilp 94.96% fill, 932,094).
- Output folders: outputs/replay_holdout/holdout_block1, data/processed_holdout_b1, outputs/forecast/model_holdout_b1.
  Log: outputs/holdout_runs.log.
