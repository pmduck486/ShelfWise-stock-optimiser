# RetailOps forecaster: decisions, in plain words

> **What this is:** every important choice made for the ML forecaster (Lump 1) and the project around it. For each one: what we chose, why, what else we considered, and the evidence. Written for viva and presentation prep.
>
> **Sources:** `plan/forecaster-build-plan.md` (decisions table 1–22), `plan/forecast-training-plan.md` (rules R1–R12), `HANDOFF.md` (decisions log), `config.yaml`, the `forecast/` code and `tests/`, plus real-data checks run on 2026-10-30.
>
> **Status when written (2026-10-30):** Phase A (config, loader, baselines, scorer), Phase B (features + fairness tests) and Phase C (q50 model, upper lines, post-processing, review checks, stopped-selling fix #26) and Phase D Tasks 6–7 (tuning, calibration, #28) are built; Phase E Tasks 8–10 (freeze, `forecasts.parquet`, slide scores; #29) were built on 2026-10-31. 163 tests pass on the laptop (Windows `.venv`), and the 19 freeze tests pass 19/19 against the real written file. (Phase C added 2026-10-30.) Decisions about unbuilt parts are marked **(planned)**.

---

## Quick list (one line each)

| # | Decision | Chosen |
|---|---|---|
| 1 | What the forecaster outputs | Four lines per store-item-day: 50%, 80%, 90%, 95% |
| 2 | Model type | scikit-learn `HistGradientBoostingRegressor` with quantile loss |
| 3 | One model or many | One global model per line, all stores and items together |
| 4 | Forecast horizon | 7 days, re-planned every day |
| 5 | Rule R2 (7-day gap) | Kept: store data must be at least 7 days old |
| 6 | Calendar inputs | Allowed for the target day itself (R3) |
| 7 | Time windows | Split by date; sealed holdout run once |
| 8 | Validation | Two folds: Nov–Dec 2014 (festive) and Jul–Oct 2015 |
| 9 | The bar to beat | Two baselines: same day last week, and the teammate's 4-week average |
| 10 | 25 December | Forecast 0, dropped from training, left out of scores, missing in inputs |
| 11 | Long zero-sales runs | Dropped from training only, still forecast |
| 12 | Calibration | One stretch factor per line on the spread (planned) |
| 13 | Inputs added | price vs normal price, momentum, store sales rank, next event name |
| 14 | Inputs dropped | Same week last year, price vs department, promo inputs |
| 15 | Price inflation | No fix needed; already handled |
| 16 | Horizon-aware model | Dropped |
| 17 | Recent price change input | Dropped |
| 18 | Feature cache | Dropped; inputs rebuilt each run, stored as float32 |
| 19 | Slow sellers | Under 5 units/day (the order model's rule), reporting only |
| 20 | Scoring the line actually used | `cover_used_line` per product |
| 21 | Calendar source | `calendar.csv`, not `sales.parquet` |
| 22 | Optional layers | B after the core; C and M only if the user says yes |
| 23 | Project scope | Stock ordering only; libraries, not hand-written algorithms |
| 25 | Overfitting fixes | Tree count from the validation curve, built-in early stopping off, 3-seed check (planned) |
| 24 | Phase C checks | A change stays only if it helps both folds (all tries logged); q80/q90/q95 compared with both baselines; 20 worst store-items + shuffle leak test (built) |
| 27 | Recency weighting + optional pre-Phase-D fixes | Dropped (user's call 2026-10-31) |
| 28 | Phase D: tuning + calibration | Own settings per line (upper lines lr 0.05); calibration k = 1 (already calibrated); no group fallback needed (built) |
| 29 | Phase E: freeze + forecasts file | One model trained through 2015-10-31, frozen with a manifest; 56 origins written to `forecasts.parquet`; both run on the laptop only (built) |
| 26 | Items that stopped selling | Keep long zero runs in training + q50 = 0 after 28 days with no sales (built; reverses #11) |
| 30 | Final order model vs our forecaster | They line up: same forecast file format, same service lines, same shared config; `policy.decide` returns a Decision object; copied into the project with 5 extra tests, 229 tests pass (built 2026-11-01) |
| 31 | Order model on Windows: pass the solver's head start correctly | `optimize/cbc.py` gives CBC a relative scratch folder so it can read the start file; keepFiles tested and rejected (built 2026-11-01) |
| 32 | Solver time per day in the 4-method replay | 15 s per day while building and testing; one final 60 s run for the core results in the slides (planned) |
| 34 | The replay's 3 simple rules and scorecard | Top-up rules with 2-day store cover and 6-day warehouse cover; method 2 adds z·σ·√D safety; net cost corrects for end stock; budget cut by scale-down + largest remainder (built 2026-11-01; Gate 1 passed, full run planned) |
| 35 | First full core replay (test window, 15 s) | Optimizer: fill 89.5%, least stock, lowest cost + lost-sales score, but below both top-up rules on service; result reported as is; to be rerun at 60 s for the slides (built 2026-11-02) |
| 33 | How the holdout is used | Sealed until all layers are done and code is frozen; the replay gives feedback on the test window only; before opening, write down the exact versions (core with all switches off, final with finished layers on, each against the 3 simple rules); report both as they come out (planned) |

---

## A. What the forecaster is

### 1. Output: four lines, not one number
- **Chosen:** for every store, item and day, four forecast lines: q50 (the most likely value), q80, q90 and q95. "q95 = 23" means "on about 95% of days, sales will be 23 or less".
- **Why:** the order model (the ILP) needs to know *how much stock covers most days*, not just the average. A perishable item is stocked to q95, a slow seller to q80, most items to q90. A single average forecast would leave the order model guessing how much safety stock to add.
- **Alternatives:** a single point forecast plus a fixed safety-stock rule; a normal-curve spread (mean + z × std, which is what the stub and baselines do).
- **Evidence:** the stub's normal-curve 95% line covers only 81.7% of days on the day before Thanksgiving 2014 and 85.4% on 23 Dec 2014 (target 95%). Real sales are skewed and spiky, so a learned line is needed.
- **Contract:** `forecasts.parquet` columns `origin, date, store_id, item_id, q50, q80, q90, q95` (`build-roadmap.md`), exactly the same as the stub so Lump 2 can swap it in.

### 2. Model: gradient-boosted trees with quantile loss
- **Chosen:** `HistGradientBoostingRegressor(loss="quantile", quantile=q)` from scikit-learn 1.9.1, one per line.
- **Why:** trees handle missing values (late-launch items, 25 Dec) without extra code, mix numbers and categories, need little tuning, and train in minutes on a laptop. The team is new to ML, so a well-known library model is easier to explain than a custom network.
- **Quantile loss in one example (95% line):** a line too low costs 0.95 per unit missed, a line too high costs only 0.05 per unit over. Line 23, sales 25 → loss 0.95 × 2 = 1.9. Line 23, sales 15 → 0.05 × 8 = 0.4. So the model pushes the line up until only about 5% of days go above it. Code: `forecast/metrics.py::pinball`; test `test_pinball_example`.
- **Alternatives:** a small neural network (moved to `reserved-plan.md`); writing our own gradient descent regression (dropped when the project switched to "use libraries"); linear quantile regression (cannot learn festival spikes or item-by-weekday patterns as easily).
- **Evidence:** a quick test model using only lags and rolling means already gave fold 2 MAE 4.18 against 5.21 for "same day last week". Also tested and dropped after that quick check: more than 300 trees, a log target, averaging several seeds, blending with the baseline.

### 3. One global model per line
- **Chosen:** each of the four models is trained on all 10 stores × 120 items together, with store, item, department and perishable flag as inputs.
- **Why:** one item alone has only about 800 days of history, too few to learn from. Pooling lets a quiet item borrow patterns (weekday shape, festival lift) from similar items.
- **Alternatives:** 1,200 separate models (one per store-item); one model per store.
- **Evidence:** the size argument above; HGB's category limit (at most 255 values per category column) is checked by `test_categories_fit_model`, and 120 items fit.

### 4. A 7-day horizon, re-planned daily
- **Chosen:** each morning ("origin") we forecast the next 7 days, origin to origin+6. The order model plans all 7 days, carries out only day 1, and plans again tomorrow.
- **Why a 7-day forecast is useful when we only act on day 1:**
  - **Supplier lead time is 5 days** (config), so a warehouse order placed today lands on day 5. You must know day 5's demand today.
  - **Batching orders:** full cases, trucks and minimum order sizes make it cheaper to order once for several days than every day; you need several days of demand to size the batch.
  - **Perishables:** stock bought today must sell before its shelf life (2–7 days) runs out, so you need the next few days of demand to avoid waste.
  - **Festival build-up:** stock for a spike (the day before Thanksgiving is the year's biggest day, about 12,400 units vs about 9,000 nearby) must be ordered days before.
  - **Safety stock and budget:** spreading cash and shelf space across the week needs the whole week's picture.
  - In short: **plan 7 days, act on 1.** Re-planning daily means mistakes in days 2–7 are fixed tomorrow, so only day 1 needs whole cases (CLAUDE.md rule 3).
- **Alternatives:** a 1-day forecast (cannot cover the 5-day lead time); a 14-day plan (CLAUDE.md: too big for the CBC solver).

---

## B. Fairness: never peeking at the future

### 5. Rule R2, the 7-day gap: kept
- **Chosen:** every input taken from the store's own records (units, prices, averages, spreads) must be **at least 7 days older than the target day**. Calendar facts are the only exception (decision 6).
- **Why:** we forecast up to 7 days ahead. On the morning of day *o*, the forecast for *o*+6 can only use data up to *o*−1, which is 7 days before *o*+6. If every input is at least 7 days old, **one model serves the whole week** and a forecast for a given date is identical whichever morning asked for it. That is also what makes the future-poison test possible.
- **Alternatives considered:**
  - A **horizon-aware model** (inputs up to yesterday, so tomorrow's forecast uses fresher data). See decision 16.
  - Adding the **target-day price** (prices are set in advance). See decision 17.
- **Evidence (real-data check, 2026-10-30):** horizon-aware lags were about 12% better for tomorrow and about 5% better across the week (fold 2 MAE by days ahead: 3.62, 3.80, 3.97 … 4.12 at day 7, which equals R2). But it would need about 7× the training rows and a rewrite of Tasks 3, 4, 7 and 9. **User decided 2026-10-30: keep R2, no changes.**
- **Code:** `FEATURE_SOURCES` in `forecast/features.py` lists the fewest days back each input reads; `build_features` never places a sales row newer than `target − 7` in its grid. Tests: `test_feature_age_registry`, `test_feature_age_by_perturbation`, `test_future_poison_features`, `test_cross_item_perturbation`.

### 6. Calendar inputs may be for the target day (R3)
- **Chosen:** weekday, month, M5 event name, SNAP day for the store's state, days until / since the nearest event, the next event's name, and the closed-day list may describe the target day itself.
- **Why:** calendars are published in advance. A manager knows next Thursday is Thanksgiving.
- **Alternatives:** applying the 7-day rule to the calendar too (it would hide the festival from the model, for no fairness gain).
- **Guard:** nothing else gets this exception. `test_calendar_features_ignore_sales` poisons all sales and prices and checks the calendar inputs do not move.

### 7. Time windows, split by date
- **Chosen (config `data.windows`):**

  | Window | Dates | Use |
  |---|---|---|
  | Warm-up | 2012-11-26 → 12-31 | history only, never a target |
  | Train | 2013-01-01 → 2015-06-30 | fitting |
  | Validation fold 1 | 2014-11-01 → 12-31 | settings, calibration (festive) |
  | Validation fold 2 | 2015-07-01 → 10-31 | settings, calibration |
  | Test (development replay) | 2015-11-01 → 12-26 | 8-week replay, compare methods and layers |
  | Holdout | 2015-12-27 → 2016-05-22 | **sealed**, run once after code freeze |

- **Why by date:** a random split would put Tuesday in test and Wednesday in train, so the model would "know" the future.
- **Why both a test window and a holdout:** we rerun the test replay for every layer and bug fix, so it slowly stops being "unseen". The holdout stays sealed through all of that and gives one clean final check. If its gain is close to the test gain, the result is real.
- **Guards:** `load_sales` refuses any file with holdout rows (`test_dev_loader_never_returns_holdout_rows`); `check_cutoff` refuses training data on or after a cutoff (`test_cutoff_guard_raises`); window order checked by `test_real_config_windows`.
- **Related fix:** the first data prep ranked the 120 products over the whole history, test weeks included (a leak). Fixed in `prepare.py`: products are chosen on sales before 2015-11-01 (R5), guarded by `test_selection_ignores_test_and_holdout_sales`.

### 8. Two validation folds
- **Chosen:** fold 1 = Nov–Dec 2014 (Thanksgiving + Christmas; its model trains only on 2013-01-01 → 2014-10-31), fold 2 = Jul–Oct 2015. Every gate must pass on both.
- **Why:** fold 2 has no big festivals, but the test window has Thanksgiving and Christmas. Fold 1 lets us check settings and calibration on festive days *before* touching the test window.
- **Alternatives:** one fold only (Jul–Oct); moving the windows (would change `sales.parquet` and the teammate's files).
- **Evidence:** the stub's 95% line covers about 91% of days on both folds overall but only 81.7% on 2014-11-26 and 85.4% on 2014-12-23. Fold 1's dates live only in the `forecast:` section of config, so no shared file changed.
- **Code:** `val_folds` in `forecast/data.py` checks the folds are in order, train_end is the day before val_start, and both end before the test window (`test_val_folds_in_order`).

### 9. The bar to beat: two baselines
- **Chosen:** (a) **naive** = same day last week (also the fallback for items with no history); (b) **stub** = the teammate's rule, average of the same weekday over the last 4 weeks (what the order model uses today). The q50 model must beat **both** on **both** folds.
- **Why:** beating only the weak naive bar would prove little. The stub is the harder, honest bar. It is computed by calling her own function (`stubs.make_stubs.forecast_targets`), read only, so the bar is exactly her file.
- **Evidence (Gate A, saved in `outputs/forecast/scores/`):**

  | MAE (units/day) | Fold 1 | Fold 2 |
  |---|---|---|
  | Naive | 4.88 | 5.21 |
  | Stub | 4.39 | 4.72 |
  | Quick test model (not the final one) | – | 4.18 |

  95% line coverage: naive 88.9% / 89.6%, stub 90.6% / 91.1%. Coverage of the line each product is stocked to: stub 86.6% / 86.9% against a target of 89.3%.

---

## C. Data cleaning choices

### 10. Christmas Day (25 Dec): four rules
Stores are closed (7 to 19 units chain-wide each year). Four choices, all settled 2026-10-30:
1. **Dropped from training** (`drop_dates`), so the model does not learn a fake slump.
2. **Forecast as 0 on all four lines** (`closed_dates`, planned in post-processing). Otherwise the model would forecast a normal day and the order model would stock up for a shut store.
3. **Left out of every score**, so free "perfect" days cannot flatter the results (`test_score_skips_closed_days`).
4. **Missing in the inputs of later days:** 25 Dec's units are set to NaN in the feature grid. Averages use the days present (at least half the window: 4 of 7, 14 of 28); a lag landing on 25 Dec stays missing.
- **Alternatives:** leave 25 Dec as a zero day; fill a lag on 25 Dec from the week before.
- **Evidence:** on 1–28 Jan 2015, MAE 4.141 with 25 Dec as missing vs 4.159 left as zero; March a tie. Filling the lag from the week before tested no better.
- **Safety check:** `load_config` refuses a config where a closed date is not also dropped from training, so the two lists cannot disagree (`test_closed_dates_are_dropped`).
- **Known side effect, accepted:** the two baselines still treat 25 Dec as a real near-zero day, so their early-January forecasts are low. No validation or test day is affected (only the holdout's first 4 weeks). The baselines stay unchanged because they are the teammate's rule; this is noted in the reports.

### 11. Long zero-sales runs
- **Changed 2026-10-31 (decision 26):** the runs are now **kept** in training (`drop_long_zero_runs: false`); the drop below is the original choice, still available as a switch.
- **Chosen:** stretches of more than 90 open days with zero sales are dropped from **training only**; those items are still forecast. The runs are found on training rows only (R4), and 25 Dec neither breaks nor extends a run.
- **Why:** a 90-day zero run usually means "not on the shelf", not "nobody wanted it". Training on it teaches the model to forecast 0 for healthy items.
- **Alternatives:** keep them; drop those items entirely (R10 says never drop products).
- **Evidence:** 184 store-items have such runs before validation (the old figure of 166 was from older data; 185 counted 25 Dec as a zero day).
- **Code:** `zero_run_stats` and `training_rows` in `features.py`; tests `test_long_zero_rule_uses_train_only`, `test_zero_run_skips_closed_day`, `test_training_rows_drop_long_zero_runs`.

### 19. Slow sellers
- **Chosen:** an item selling under 5 units/day on average (from `items.csv`) is "slow", the order model's own rule (24 items). Used for reporting only.
- **Why:** to report honestly where the model is weak, using the same groups the order model uses.
- **Alternative rejected:** under 1 unit/day left only 16 of 1,200 store-items, too few to score.

### 21. Calendar comes from `calendar.csv`
- **Chosen:** every target day's calendar inputs are read from `data/m5/calendar.csv`.
- **Why:** sales stop on 2015-12-26 but the last forecast origins reach 2016-01-01. Reading the calendar from sales would leave those days with no weekday or New Year flag. Calendars are public in advance, and no holdout sales are read.
- **Tests:** `test_calendar_matches_sales`, `test_calendar_covers_forecast_days`, `test_targets_after_sales_end`.

---

## D. Inputs (features)

### 13. Inputs we use
All store-data inputs end at least 7 days before the target.

| Group | Inputs | Why |
|---|---|---|
| Recent sales | sales 7, 14, 21, 28 days back | same weekday in each of the last 4 weeks |
| Averages and spread | 7-day and 28-day mean, 28-day std | level and how jumpy the item is |
| Price | price 7 days back; change vs the 4 weeks before; price vs the item's normal price | catches discounts |
| Trend | momentum = 7-day mean ÷ 28-day mean | is the item speeding up or slowing down |
| Popularity | store_rank_28 = item's sales rank in its store, 0–1 | a top seller behaves differently |
| Calendar | weekday, month, event flag and name, SNAP, days to / since event, next event name | festivals and food-stamp days |
| Who | store, item, department, perishable flag | lets one global model tell items apart |

- **next_event_name, why:** the biggest day, 2015-11-25, is the day *before* Thanksgiving and has no event of its own. Knowing the next event lets the model tell the run-up to Thanksgiving (1.38× the day before) from the dip before Memorial Day (0.85×).
- **Code:** `feature_names`, `build_features` in `features.py`; `test_sale_features_age`, `test_sale_features_no_inf`, `test_price_inputs_exact`.

### 14. Inputs we left out, and why
| Input | Why it was left out | Evidence |
|---|---|---|
| Same week last year (364 days back) | sales start 2012-11-26, so about 40% of training rows (most of 2013) have no year-ago value; fetching older data means rebuilding `sales.parquet`, which the teammate uses | data range |
| Price vs department average | 99.5% of its variation is just which item it is, so it adds nothing over `item_id` | tested 2026-10-30 |
| Promo inputs (discount depth, promo flag and age, chain-wide promo share) | promos are 2.7% of days with an average lift of 1.15×; no consistent gain on either fold | tested 2026-10-30 |

### 15. Price inflation: no fix needed
- **Question:** prices drift up over the years. Does that break the model?
- **Answer:** no. Average prices rise about 1% a year (+2.5% by 2015 Q3 vs Q1 2013); validation and test sit only 1.5–2.5% above the training "normal price", and only 1–2% of rows go beyond any price the item had in training.
- **Why it is already handled:** the recent-sales inputs carry the current level, `price_chg_4w` is a ratio, and `price_rel_item` is a ratio; trees treat a slightly higher price as a known value.
- **Alternative tested:** a 52-week rolling "normal price". MAE (fold 1 / fold 2): plan as-is 3.99 / 4.14; 52-week normal 4.06 / 4.15 (worse in fold 1); no price-level inputs 4.02 / 4.14. All within about 1%, so no gain. No files changed.

### 16. Horizon-aware model: dropped
- **Idea:** let the forecast for tomorrow use yesterday's sales, the day after use the day before, and so on.
- **Evidence:** about 12% better for tomorrow, about 5% better over the week (decision 5).
- **Why dropped:** about 7× the training rows and a rewrite of Tasks 3, 4, 7 and 9, too costly before the 1 Nov deadline; it also breaks the "one model, one forecast per date" property that the fairness tests rely on. **User's call 2026-10-30: dropped entirely, not even optional.**

### 17. Recent price change (target-day price): dropped
- **Idea:** prices are set in advance, so use the price on the target day.
- **Evidence:** M5 prices change only on Saturdays, weekly; about 1% of validation days have a price different from 7 days earlier. MAE 4.040 → 4.049 (fold 1, worse) and 4.120 → 4.114 (fold 2). Rows with a price change were about 10% better, but they are too rare to matter overall.
- **Why dropped:** no overall gain. **User's call 2026-10-30: dropped entirely.**

---

## E. Training and checking (planned for Phases C–E)

### 12. Calibration: stretch the spread, not shift it (built 2026-10-31, Task 7; see #28)
- **Chosen:** if a line's coverage on the validation folds is off by more than 3 points, find one factor *k* per line so that `q50 + k × (line − q50)` hits the target (k found by bisection between 0.5 and 4). q50 itself keeps a plain unit shift. One number per line, not per group, to avoid fitting noise.
- **Why:** a fixed unit shift (e.g. +2 units) over-stocks small sellers and under-stocks big ones. Stretching each product's own gap treats a 2-unit seller and a 48-unit seller fairly.
- **Alternatives:** a fixed unit shift per line (the old plan); one factor per segment.
- **Evidence:** baseline check (fit on one fold, judged on the other): with a unit shift the 95% line gave 96.9% for the smallest quarter of sellers and 93.0% for the biggest. With the scaled correction every size quarter landed between 94.3% and 94.6%.
- **Order:** tuning (Task 6) comes before calibration (Task 7), so the factors are fitted on the model we actually ship. Lines are sorted *after* calibration so they never cross.

### Other planned training rules
- **Tuning:** at most 6 settings (learning rate 0.05/0.1, leaf nodes 31/63, 300 trees), best by average score over both folds, but only among settings that are not worse than the current ones on either fold (decision 24); validation only (R6).
- **Train once:** retrain on train + validation, using data through 2015-10-31 (cutoff 2015-11-01 = the first unseen day), freeze with a `manifest.json` (cutoff, windows, features, seed, sklearn version, config hash). The replay refuses a model whose cutoff is after the replay start (R7). Weekly retraining is optional later. **Built 2026-10-31 (#29).**
- **Test-window scores only after the replay** (R10), and every result is reported, including where the model loses.
- **Reproducible:** seed 1 in config; same config → identical forecasts (R12).

### 24. Phase C checks: both-folds rule, every line vs the baselines, error review (built in Tasks 4–5; Task 6 part planned)
Approved by the user on 2026-10-30 (re-check strategies 3, 4 and 5). Explained in https://claude.ai/artifact/Qp1ix7qnqpzzhcCdy4WuRi.
- **Experiment log + both-folds rule (Tasks 4, 6).** Every change tried on the model (a setting, an input switched off) is one row in `outputs/forecast/scores/experiments.csv`, one change at a time. A change stays only if it lowers q50 pinball on fold 1 **and** fold 2; a tie or a loss on either fold means the simpler version stays. The Task 6 tuning winner must also not be worse than the current settings on either fold.
  - **Why:** the more versions we try, the more likely one looks better by luck. Two different periods (festive and ordinary) make a lucky win much less likely, and the log is a ready "what we tried" table.
  - **Evidence:** the target-day price test: fold 1 MAE 4.040 → 4.049 (worse), fold 2 4.120 → 4.114 (better). A row-weighted average said keep it (4.094 → 4.093), a plain average said drop it (4.080 → 4.082). The both-folds rule gives one clear answer: drop.
- **Every line vs the baselines (Task 5, shown again at the Task 7 gate).** q80, q90 and q95 pinball are compared with the naive and stub baselines on each fold; a line that loses is flagged to the user.
  - **Why:** no product is stocked to q50 (30 items use q95, 66 use q90, 24 use q80), so the lines that drive orders must beat the simple rules too. Coverage alone can't tell a tight line from a needlessly wide one; pinball can.
  - **Bar to beat (pinball, fold 1 / fold 2):** q80 naive 1.971 / 2.077, stub 1.805 / 1.939; q90 naive 1.398 / 1.464, stub 1.284 / 1.374; q95 naive 0.964 / 1.003, stub 0.883 / 0.943.
- **Error review + leak check (Task 5, reports only).** The 20 worst store-items per fold, ranked by error relative to their own average sales, each with the stub's error and its traits (perishable, late launch, long zero runs, odd prices). A shuffle test on fold 2 scrambles one input at a time and measures how much the error rises.
  - **Why:** group averages hide a few very bad store-items, and the Phase B tests check the input code only, not the trained model.
  - **Leak yardstick:** a model allowed yesterday's sales scored fold 2 MAE 3.62 in the R2 check; our 7-day model should score near 4.1. A fold 2 q50 MAE below 3.6 means it is almost surely peeking, so work stops for diagnosis.
- **Alternatives:** judge changes on one combined score; check q50 only; rely on the input tests alone.
- **Code:** `forecast/experiments.py` (`keep_change`, `log_experiment`, `read_log`), `log_try` in `forecast/evaluate.py` (a try counts only on `forecast.keep_seeds` [1, 2, 3], with no suspected leak, against the last kept row of the same method and seeds), `forecast/review.py` (`compare_lines`, `item_traits`, `worst_items`, `shuffle_importance`, `leak_check`); tests in `tests/test_forecast_model.py`.
- **Result (gate C, 2026-10-30):** all six upper lines beat both baselines (q95 pinball 0.820 / 0.788 vs stub 0.883 / 0.943). The worst store-items are items that stopped selling for good (FOODS_3_764 at all 10 stores in fold 2: no sales since April 2015, model MAE 4–6 units/day, stub about 0). The shuffle test shows recent sales matter most (7-day average, lag 7, 28-day average, then the item code); no sign of a leak (fold 2 MAE 4.21, floor 3.6).

### 25. Overfitting fixes: tree count from the validation curve, 3-seed check (built in Task 4; tree-count choice planned, Task 6)
- **Chosen:** scikit-learn's built-in early stopping is off; the validation error is recorded every 50 trees and the tree count is picked on both folds. A change is kept only on the mean of 3 seeds (1 seed while exploring).
- **Why / evidence (2026-10-30 run, planned settings, 3 seeds):** festive-fold q50 pinball 2.029 at 100 trees vs 2.068 at 300 (1.9% worse); main fold flat. Built-in early stopping never fired (its random 10% hold-back shares days with training rows). Seed alone moved the score by up to 0.015 pinball (about 0.7%).
- **Alternatives:** fixed 300 trees; single-seed judging.


### 26. Items that stopped selling: keep long zero runs + dead-item rule (fix C+A, user's call 2026-10-31; built)
- **Chosen:** (C) long zero-sales runs now **stay in training** (`drop_long_zero_runs: false`, which reverses decision 11's first choice), and (A) q50 is set to 0 for a store-item whose 28-day average ending 7 days back is 0 (`dead_item_q50: true`). Only q50: q80/q90/q95 keep the model's value, so an item that starts selling again still gets some stock.
- **Why:** the error review found the worst store-items were items that stopped selling for good (e.g. FOODS_3_764 at all 10 stores, 0 sales since April 2015) while the model kept forecasting their old level. Dropping long zero runs meant the model almost never saw "a month of zeros, then still zero", and it leaned on the item code. The stub handled them because its 4-week average falls to 0.
- **Evidence (3 seeds, all four lines, both folds; fold 1 / fold 2):** q50 MAE 4.117 / 4.187 → **3.931 / 4.150** (−4.5% / −0.9%); q80 1.775 → 1.731 / 1.744 → 1.730; q90 1.250 → 1.228 / 1.216 → 1.206; q95 0.818 → 0.813 / 0.788 → 0.785. Every line better on both folds; seed spread 0.026 / 0.025. The worst store-items are now about as good as the stub on the same rows. Caveat: fold 2 perishables slightly worse (MAE 6.89 vs 6.77).
- **Alternatives:** A alone (q50 better, upper lines unchanged; the log's q50-only rule prefers it by 0.019 on fold 2, but the upper lines drive orders); C alone; A on all four lines (fold 2 q95 worse, and a returning item gets no stock); no change.
- **Code:** `dead_item_rule` in `forecast/postprocess.py`, applied in `predict_lines` (`forecast/model.py`) from the `roll_mean_28` input; config `forecast.drop_long_zero_runs`, `forecast.dead_item_q50`. Tests `test_config_keeps_long_zero_runs_and_dead_item_rule`, `test_dead_item_rule_zeroes_q50_only`, `test_predict_lines_applies_dead_item_rule`; `test_training_rows_drop_long_zero_runs` still checks the switch.


### 27. Recency weighting and the optional pre-Phase-D fixes: dropped (user's call 2026-10-31)
- **Chosen:** no recency weighting (re-check strategy 6), and none of the three small fixes found in the pre-Phase-D check (all-lines keep rule, dead-item rule after calibration, tuning q50 after the dead-item rule).
- **Why:** the model already sees each item's recent level (lag 7, 7- and 28-day averages), so weighting was expected to help little, and it would halve the weight of fold 1's only earlier Christmas (2013). The measured effect of the fixes: pooled coverage is already within tolerance (53.5 / 80.4 / 89.8 / 94.6%), so calibration would not move; the tuning effect is at most 0.35% MAE on fold 1, below seed noise (0.026). The all-lines keep rule was only needed for a four-line change like recency weighting; Task 6 picks each line on its own.
- **Alternatives:** test recency weighting with half-lives 365 / 730 days as an on/off switch.


### 28. Phase D: per-line settings, tuning, calibration, cross-fold and group checks (user's call 2026-10-31; built)
- **Chosen:** each line (q50, q80, q90, q95) has its own settings and tree count (`forecast.line_params`), picked on both folds and confirmed on 3 seeds; calibration as planned (#12); the cross-fold calibration check (re-check strategy 2) is reported, not gated; a group check against the stub (strategy 8) decides whether a fallback is needed.
- **Why:** the lines want different tree counts (q50 is best near 100–300 trees, q95 keeps improving to 450). One setting for all lines would shortchange the lines that drive orders.
- **Result (3 seeds, fold 1 / fold 2):** upper lines improve with learning rate 0.05: q80 pinball 1.727 / 1.721, q90 1.223 / 1.201, q95 0.806 / 0.778 (all lower than before on both folds). q50 keeps lr 0.1 / 300 trees because the tuned q50 was worse on fold 1. Calibration: every factor k = 1 (coverage already within 3 points pooled); the cross-fold check agrees. No group loses to the stub on both folds, so no fallback.
- **Gate D (user's call 2026-10-31): passed with a note.** Fold 1 q50 covers 55.0%, 5 points over 50, only because 8.3% of its rows are exact ties (mostly the dead-item rule forecasting 0 and 0 sold, which count as covered). Counting ties as half gives 50.9% (fold 2 50.0%). Alternatives not taken: count ties as half in the coverage measure; gate q50 only on rows the dead-item rule doesn't touch (51.8%).
- **Code:** `model_params`, `fit_models` (model.py); `forecast/tune.py`; `forecast/calibrate.py`; `group_table`, `group_check` (review.py). Tests: `test_model_params_per_line`, `test_fit_models_uses_line_params`, `test_best_params_needs_both_folds`, `test_tune_reads_no_test_rows`, `test_scale_fixes_coverage_on_synthetic`, `test_cross_fold_check_fits_on_one_judges_other`, `test_group_check_flags_only_both_fold_losses` and others.

### 29. Phase E: one frozen model and the forecasts file, made on the laptop only (user's call 2026-10-31; built)
- **Chose:** `python -m forecast.freeze` retrains all four lines once on 2013-01-01 → 2015-10-31 (cutoff 2015-11-01, the first unseen day) with the final settings (#28) and calibration numbers (all k = 1, q50 shift 0), and saves `outputs/forecast/model/bundle.joblib` + `manifest.json`. `python -m forecast.write_forecasts` replays the 56 test origins (2015-11-01 → 12-26) with that one model, reading only sales before each origin, and writes `data/processed/forecasts.parquet` in the stub's exact columns. Both refuse to run unless `platform.system()` equals `forecast.freeze_platform` (`Windows`).
- **Why laptop only (loophole 6):** laptop and cloud differ in the third decimal (fold-1 q50 MAE 4.108 vs 4.106), so the file the teammate uses has to be reproducible on the machine that hands it over.
- **Guards:** `load_frozen` refuses a model whose cutoff is after the replay start, a changed config (sha256 of the forecast section + data windows as sorted JSON, with model_dir, scores_dir and freeze_platform left out, so line endings and comments don't matter), or a different scikit-learn version. A store-item with no sales before an origin gets the baseline (0 on every line) and is counted (0 in the real run).
- **Evidence (laptop, 2026-10-31):** freeze 126 s on 1,226,019 rows; forecasts 470,400 rows in 36 s; the contract test against the stub passes (same columns, dtypes, rows and keys; lines finite, ≥ 0, never crossing). Mean q50 6.78 vs stub 7.92; q90 14.20 vs 14.08; q95 17.02 vs 15.84 (our spread is wider at the top). Slide scores: MAE vs stub 3.931 vs 4.389 on fold 1 (10.4% lower) and 4.149 vs 4.725 on fold 2 (12.2% lower).
- **Considered:** freezing in the cloud and copying (rejected: not reproducible on the laptop); retraining every origin (rejected: train once, R7; weekly retraining stays optional).

### 20. Score the line each product actually uses
- **Chosen:** besides overall coverage, report `cover_used_line`: coverage of the line each product is stocked to (perishables and top 30 → q95, slow → q80, rest → q90), mirrored from the order model's proposed service levels.
- **Why:** that is the number that decides empty shelves. A great q80 means nothing for a perishable stocked to q95.
- **Code:** `service_line`, `segments`, `score` in `forecast/metrics.py`; `test_service_line_rules`, `test_cover_used_line`.

---

## F. Engineering choices

### 18. Feature cache: dropped; float32 inputs
- **Chosen:** inputs are rebuilt on every run; feature columns are stored as float32.
- **Why:** building inputs for all ~1.08M training rows takes about 3 seconds. A cache would save almost nothing, but each saved table is 17–35 MB inside the OneDrive folder, rewritten after every feature-code edit, and its naming key had already needed two fixes (a stale cache is a silent-bug risk). float32 halves memory (about 70 MB instead of 130 MB) with the same results.
- **Alternative:** the approved cache keyed by cutoff, config, sales hash and code version (strategy 7). **User's call 2026-10-30: dropped.**

### Process decisions
- **Every number in `config.yaml`**, no constants in code; every extra layer is a switch, default off, and the core must pass its tests with all switches off.
- **Build in gated phases (A–E)**, stopping for the user's review at 5 gates (Tasks 2, 3, 4, 7, 9). Each phase starts only on the user's "start".
- **Tests first** for each task (write failing test → implement → pass).
- **Don't break the teammate's inputs:** `sales.parquet`, `items.csv`, the stub files, the `data:`/`network:`/`stubs:` config sections and the forecast contract stay unchanged. The only behaviour change she sees is a 0 forecast on 25 Dec.

---

## G. Optional layers and project scope

### 22. Layers
| Layer | What | Decision |
|---|---|---|
| B | Forecasts for the random product sets used by the robustness replay (R1) | Build after the core, same frozen settings, no new tuning |
| C | 5-day total forecasts for warehouse orders (adding five daily q95s overstates the total) | Optional, built last, **only if the user says yes** and the teammate's warehouse rule would use it |
| M | Manual demand multiplier for what-ifs (e.g. a crisis or festival week) | Optional, built last, **only if the user says yes**. It must stay at 1.0 in the fair replay, so it cannot improve scores |
| P7 hook | Empty-shelf correction | A hook in `training_rows` that does nothing while its switch is off |

### 23. Project scope
- **Chosen:** RetailOps, stock ordering for a 10-store, 2-warehouse chain on real Walmart M5 food sales. Two models: the ML forecaster (Lump 1) feeds the order ILP (Lump 2, built by a teammate), then shadow prices, an 8-week replay and a dashboard.
- **Why only stock ordering:** keeps the core small enough to finish and test. Staff shifts, billing counters, truck routing, the transport problem and from-scratch textbook algorithms were moved to `reserved-plan.md`.
- **Use libraries** (scikit-learn, PuLP/CBC) rather than hand-coding algorithms (reversed the 2026-10-27 decision), so effort goes into a correct, fair model.
- **Products:** top 120 FOODS items by sales before the test window; 20 fast FOODS_3 items flagged perishable, with assumed shelf lives of 2–7 days (M5 items are anonymous, so this is a stated assumption).

---

## H. Connecting the forecaster and the order model

### 30. The final order model lines up with our forecaster (checked and copied in 2026-11-01; built)
- **Copied in (2026-11-01):** her `optimize/`, `examples/`, `configs/`, `data/checkpoint2`, `data/checkpoint3`, `data/validation` and her 10 test files now live in `project/retailops`, unchanged except for the solver fix (#31). Her `order_model:` section was appended to `config.yaml` word for word; the `data:`, `network:`, `stubs:` and `forecast:` sections parse exactly as before and the frozen forecaster's fingerprint is unchanged, so it still loads. Her docs and saved evidence are kept for reference in `handoff/order-model-final/`. `pulp==3.3.0` is in `requirements.txt`. **Full suite: 229 passed** (our 164 + her 58 + 5 spec-gap tests + 2 warm-start tests), and the service-line test now runs instead of skipping. The small real-forecast replay gives exactly the same numbers as before the copy (fill 96.23%).
- **What was checked:** the teammate's final order model (`retailops_final_L2.zip`, SHA-256 `1391e1e5…ffb9`) was unpacked into a scratch folder (`_checks/l2_final/`) and checked against our forecaster on the laptop. Nothing in `project/retailops` was changed for the check.
- **Result, one line per item:**
  - **Forecast file:** our `forecasts.parquet` and her `forecasts_stub.parquet` have the same columns in the same order, the same dtypes, and the same 470,400 (origin, date, store, item) keys. Ours has no NaN, no negative value and no crossing lines, and is 0 on 25 Dec.
  - **Service lines:** her rule and ours put every one of the 120 items on the same line (30 on q95, 66 on q90, 24 on q80).
  - **Shared config:** the `data:`, `network:` and `stubs:` sections are identical key by key, and all 9 shared input files match her manifest's hashes.
  - **Real-forecast replays:** both passed every audit. Small case, 7 days: fill 96.23%. Full size, 3 days (15 s solver budget): fill 96.28%.
  - **Her tests:** 58 pass on the laptop (pulp 3.3.0 was added to the `.venv`).
- **One contract change to adopt:** `policy.decide` now returns a **Decision** object holding today's store orders **and** today's warehouse (supplier) orders, plus deliveries and the pipeline. The roadmap said it returns one DataFrame. The replay (Lump 4) must read both order tables; if it read only store orders, the warehouses would never be refilled.
- **5 extra tests (built: copied in 2026-11-01, all pass):** `tests/test_core_spec_gaps.py` adds the core-spec checks that had no test of their own:
  - the waste cost works: raising it lowers the stock of a perishable;
  - a zero-sales product gets no order and causes no error;
  - the size log is filled in and printed;
  - the integer vs LP "milk" example: the LP orders 1.5 cartons, the integer model orders 2;
  - the forecaster's service lines equal the order model's service levels.
  
  In the scratch copy they ran 62 passed + 1 skipped (the service-line check skips until both sections share one config). In the project, where they do, everything passes (229).
- **Why check before connecting:** a mismatch found now (a column name, a line, a date rule) costs minutes; found after the replay it would invalidate results.
- **Still open, noted for layers P2/P3:** the order model placed 17 one-carton orders in 3 days for store-items our forecast marks as stopped selling (q50 = 0, upper lines still above 0). This is the "important to notice" item in `build-roadmap.md`.

### 31. Order model on Windows: pass the solver's head start correctly (user's call 2026-11-01; built)
- **Chosen (built 2026-11-01):** a small new file `optimize/cbc.py` builds every warm-started CBC solver. On Windows it points PuLP's temporary folder at a **relative** folder `.cbc_scratch` (gitignored), so CBC can open the start file. PuLP still gives each solve a random file name and deletes its files afterwards, so nothing piles up and parallel solves can't clash. The two solver calls in `optimize/model.py` and `optimize/dc_solver.py` now use `cbc_solver(...)` with the same arguments; nothing else in her code changed. (From 2026-11-01 no more work comes from teammates, so the order model's code is ours to maintain.)
- **Rejected after testing: `keepFiles=True`.** It also fixes the bug, but it leaves about 22 MB of solver files per run in the working folder and gives them fixed names, so two runs started from the same folder would overwrite each other's files.
- **Guard test:** `tests/test_warm_start_reaches_cbc.py` reads CBC's own log and fails unless it says "MIPStart values read", for the single-model path and for each warehouse block. Both tests **fail on her unfixed code** on the laptop and **pass** with the fix.
- **Harmless leftover:** PuLP still prints "warmStart requires keepFiles=True" on Windows. It checks this when the solver is created, before the folder is changed; the CBC logs show the start is read.
- **What the "head start" is:** each day the solver gets 15–60 seconds to search for the best orders. Her code first builds a quick plan from the forecasts and hands it to the solver as a starting point, so the solver only has to improve on it.
- **Why:** on Windows the hand-over fails silently. PuLP writes the starting plan to a temporary file with a full path (`C:\…\pulp.mst`), and CBC puts `.\` in front of it, so it can't open the file. CBC then starts from scratch and can't beat the quick plan in time. On all 3 full-size test days, the code fell back to carrying out the quick plan. Her Linux runs never had this. With `keepFiles=True` the file is written under a short name in the current folder, which CBC finds.
- **Alternatives:**
  - the teammate fixes it and sends a new zip (slower, with the 1 Nov deadline);
  - leave it (then every Windows run never really optimises day 1, and the viva would present a quick plan as the optimizer's answer).
- **Evidence (3 full-size days, real forecasts, 15 s budget, before → with keepFiles):**

  | | Before | keepFiles |
  |---|---:|---:|
  | Days that fell back to the quick plan | 3 | 0 |
  | Plan objective, 1 / 2 / 3 Nov | 60,960 / 78,903 / 79,964 | 58,867 / 76,755 / 78,374 (−3.4% / −2.7% / −2.0%) |
  | Solver gap, median | 9.70% | 6.76% |
  | Wall time | 423 s | 234 s |

  - **Logs:** before, the logs have "opening mipstart file .\C:\…" and nothing more. After, they say "MIPStart values read for 34362 variables" and "MIPStart provided solution with cost …" for both warehouses every day.
  - **Firm orders:** the day-1 orders actually carried out were unchanged on those 3 days, so fill (96.28%) and cash cost (46,930.70) are identical; in 15 s the solver improved only the tentative days 2–7.
  - **With the built fix (`cbc.py`, same 3 days, run in the project):** fallback days 0, "MIPStart values read" for both warehouses on all 3 days, median gap 7.06% (max 7.66%), wall time 172 s, every audit passed, no solver files left behind. Day 1's objective equals the keepFiles run (58,867.1); day 2 found a lower one (76,227.5) but with different orders, which changed day 3's stock, so fill came out 95.61% instead of 96.28%. A rerun of the same command gave the same days 1–2 and a different day 3: with a 15 s limit the solver stops at a different point each time, so short-budget runs vary by a little. Lesson for the analysis: a lower plan score on one day does not guarantee better real sales the next day.
  - **Not tested yet:** whether it beats the quick plan over 56 days, or with the 60 s production budget.
- **Where in code:** `optimize/cbc.py` (new), the solver calls in `optimize/model.py` and `optimize/dc_solver.py`; test `tests/test_warm_start_reaches_cbc.py`. Run outputs: `outputs/order_model/small_real_check/`, `outputs/order_model_runs/full_real_3d/`.

### 32. Solver time per day in the 4-method replay: 15 s while building, one 60 s run for the final results (user's call 2026-11-01; planned)
- **Chosen:** while the replay and its 3 baselines are being built and tested, the optimizer gets **15 seconds of solver search per day** (`configs/validation_fast.yaml`). The core results shown in the slides come from **one final run at the production setting** in `config.yaml`: 60 seconds per day, stopping early once the plan is within 1% of the best possible (the "gap target").
- **What the budget means:** each day the solver searches for the cheapest set of orders. It stops when the time runs out or when it can prove its plan is within 1% of the best possible. More time usually means a slightly cheaper plan, but every run takes longer.
- **Why:** the replay covers 56 days, and the optimizer has to solve every day. At 15 s a full run takes roughly an hour per run of the optimizer; at 60 s it takes about twice that or more. Building and testing need many reruns, so 15 s keeps each cycle short enough to finish before the 1 Nov deadline. The final numbers still use the full production budget. The 3 baselines are simple rules, so they don't use the solver and run fast either way.
- **Alternatives:**
  - 15 s for everything: fastest, but the headline numbers would come from a reduced setting.
  - 60 s for everything: the best plans throughout, but every test cycle takes 2 hours or more.
- **Evidence (measured so far):**
  - Laptop, full size, 15 s budget, with the warm-start fix (#31): 3 days in 172 s wall time (about 57 s per day, of which about 17 s is search and the rest is building the model and the audits). Before the fix the same run took 423 s.
  - Her Linux machine, full 56-day window at 15 s: median 23.5 s per day, 22 min of solving in all. Her 3-day production sample at 60 s: gap up to 5.5% (vs 7–10% at 15 s on the laptop).
  - The 60 s budget has not been timed on the laptop at full size yet; the estimate is 1.5–2+ hours per run.
  - Short budgets vary a little from run to run: the same 3-day command gave the same days 1–2 but a different day 3 (objective 84,198 vs 83,378). This is a reason to take the final numbers from one fixed, logged 60 s run and not mix 15 s results into the slides.
- **Where:** `configs/validation_fast.yaml` (15 s) and `config.yaml → order_model.solver` (60 s, `mip_gap: 0.01`).

### 33. The holdout: sealed until the end, versions written down first, both results reported (user's call 2026-11-01; planned)
- **Chosen:**
  - **While building and adding layers,** the replay (Lump 4) runs on the **test window only** (2015-11-01 → 12-26). That is where every layer is switched on, compared with the core, and kept or dropped.
  - **The holdout (2015-12-27 → 2016-05-22, 148 days) stays sealed** until the core and all layers are finished and the code is frozen. Its sales are not even in `sales.parquet`.
  - **At code freeze, before opening it, we write a run list into the docs** (dated, committed with the final config) saying exactly what it will run:
    - (a) the **core**, with every layer switch off;
    - (b) the **final version**, with the finished layers on;
    - each against the **same 3 simple rules**: same as last week; top up to a fixed level; that rule fed by our forecast.
  - **Then the holdout is opened once,** and both results are reported exactly as they come out, good or bad, with no picking afterwards and no changes.
- **Why:** if a layer were kept or dropped because of how it did on the holdout, the holdout would quietly become a second test window, and its result would no longer prove anything. Writing the versions down first (often called "pre-registration") stops us from running several versions and showing only the best one. Reporting both the core and the final version also shows how much the layers add, on data nobody has looked at.
- **What the comparison shows (each step adds one thing):**

  | Method | Forecast | Ordering | What it shows |
  |---|---|---|---|
  | Same as last week | none (last week's sales) | simple rule | the store's habit, the starting point |
  | Top up to a fixed level | none | simple rule | a standard textbook rule |
  | Top up, fed by our forecast | our ML forecaster | simple rule | the **forecaster's** gain (vs the row above) |
  | Our model (core / final) | our ML forecaster | the order ILP | the **optimizer's** gain (vs the row above), and the layers' gain (final vs core) |

  In short: the 2 rules that don't use our forecast show results with **neither** of our models; the rule fed by our forecast shows the **forecaster alone**; the optimizer versions show **forecaster plus optimizer**, then the **layers on top**.
- **Alternatives:** check layers on the holdout as we go (rejected: that peeks at it); report only the final version (rejected: it hides whether the layers helped); report only the better of the two (rejected: that is picking after the fact).
- **To be ready before opening (planned, see also #34 for the replay itself):** build `sales_full.parquet` with the holdout dates; retrain the forecaster once with the same settings on data up to 2015-12-26 and write holdout forecasts; let the order model's replay accept holdout dates (it refuses them on purpose today); decide the starting stock (continue from each method's end-of-test state, or restart all from one common stock); time a 60 s day (the optimizer's holdout run may take around 4–5 h).

### 34. The replay (Lump 4): the 3 simple rules, the scorecard and the budget cut (built 2026-11-01; Gate 1 passed; full run planned)
- **What the documents already fixed** (checked by reading all ~99 project documents, 2026-11-01): the 4 methods and what each comparison shows (core spec Step 7), the same start for every method (core spec, R11), the scorecard list (`build-roadmap.md` kpis.json), "a 3-day hand-worked example matches exactly; no method ever sees future sales" (core spec), the daily rule checks (spec E) and the test-run log (R9a). Details and sources: `plan/l4-replay-plan.md`.
- **Chosen where nothing was written (all numbers in `config.yaml → replay:`):**
  - **Top up** = order (target − stock on hand − stock already on the way), never below 0, rounded **up** to whole cartons (stores) or pallets (warehouses).
  - **Store cover:** from today until the next delivery that can arrive, normally **2 days** (1-day delivery + ordering every day); around Christmas it stretches (23 Dec covers through 25 Dec, 24 Dec orders nothing).
  - **Warehouse cover:** **6 days** (5-day supplier delay + 1), using its stores' demand by the same method's own signal.
  - **Method 1, habit:** target = last week's sales on the same days. **Method 2, fixed level from the 4-week average:** μ·D + z·σ·√D, with μ and σ from the 28 days before today and z from the product's service level (80% → 0.84, 90% → 1.28, 95% → 1.64). **Method 3:** the same rule with our forecast's line (q80/q90/q95) summed over the cover days.
  - **Warehouse short:** whole cartons shared in proportion to what each store asked for.
  - **Budget:** a simple rule's supplier order must fit the same daily warehouse budget the optimizer has. It is scaled down, rounded down to whole pallets, and the money left over is handed back one pallet at a time to the lines that lost most in rounding ("largest remainder"). **Changed during Gate 1:** plain rounding down cut every 1-pallet line to 0 in the small case (the budget is smaller than one pallet of every item), so the top-up rules bought nothing for a week.
  - **Net cost** = cash cost − (value of stock + orders on the way at the end − the same at the start), at buying price. Without it, a method that ends with empty warehouses looks cheap because it simply bought less. Her math review asked L4 to define this.
- **Why:** these are the textbook "order-up-to" rules a store or a course would use, with the same lead times, service levels, cartons, trucks and budget as our model, so the comparison is fair (R11) and each step shows one thing (habit → average → our forecast → our optimizer).
- **Evidence (Gate 1, small case: 2 stores × 10 products × 1–7 Nov 2015, real forecasts, 15 s):** all 4 methods ran through her engine; every daily rule check and her independent audit passed; identical start state for all; the optimizer row equals her own small run exactly (fill 96.23%). Fill: habit 96.6%, average top-up 98.5%, forecast top-up 98.5%, optimizer 96.2%. Waste: 226 / 752 / 791 / 295 units. Net cost: 5,670 / 7,294 / 7,404 / 5,913. One week on 2 stores is too short to judge; the full 56-day run decides.
- **Gate 1 diagnosis (2026-11-01, read-only + one 14-day small run):**
  - **Measurement is fair:** the waste estimate uses our forecast's q50 for every method; 163 of the waste units on 1 Nov come from the shared start stock and are identical for all 4 methods. Net cost = cash − (end − start value of stock + orders on the way, at buying price); it gives full credit to leftover stock, which slightly favours methods that end the window overstocked (habit ended with 8,088 units still on the way).
  - **The optimizer's weak result is a real model behaviour, not a bug in the replay:** in the small case it **never buys from suppliers** (0 units in 14 days), so it rations warehouse stock (e.g. the CA_1 warehouse kept 29 units of FOODS_3_226 while the store sat empty for 4 days) and collapses in week 2 (14-day fill 69% vs habit 95%).
  - **Why:** in a 7-day plan a pallet ordered today arrives on day 6, so its value comes only from the end-of-plan warehouse target (5 days of need, shortfall charged 1.1 × buying price) and the daily warehouse-protection penalty. If one pallet is bigger than about 1.1 × 5 days of need, buying it costs more than the penalties it removes. With one store per warehouse (small case) a pallet is 22–27 days of sales, so all 20 warehouse-items are never bought.
  - **At full size** (4–6 stores per warehouse) 116 of 240 warehouse-items fail that simple test, but her full 56-day run still bought 231 of 240, because the protection penalty adds up as stock falls. So the small case exaggerates the problem; the full run is the real test.
  - **Possible fix if the full run shows shortages (a model change, user's call):** value supplier stock beyond the 7-day plan (longer warehouse target or a separate warehouse horizon), or smaller supplier packs for slow items.
- **Code and tests:** `replay/` (history.py, baselines.py, engine.py, kpis.py, cli.py); 34 tests in `tests/test_replay.py`, including the 3-day hand-worked case per rule, the future-poison test for all 4 methods, Christmas, budget and warehouse-short checks, and hand-checked KPIs. Full suite 263 passed.

### 35. First full core replay: 56 days, 10 stores × 120 products, 15 s (run 2026-11-02; the 60 s run for the slides is planned)
- **Run:** `outputs/replay/core_test_15s_2026-11-01/`, test window 2015-11-01 → 12-26, our real forecasts, `configs/validation_fast.yaml` (15 s per day), all 4 methods from the same start. Logged in `outputs/test_runs.log`. Every daily rule check and her independent audit passed for all 4 methods (80,640 stock and pack checks, 13,440 warehouse-dispatch checks, 560 truck checks, 112 budget checks, 56 conservation checks each; largest residual 3e-11). Optimizer: 55 days stopped on the 15 s limit with a usable plan, 1 reached the 1% target, gap median 6.0% (max 13.2%), 0 fallback days, 58,110 variables (2,650 whole numbers).
- **Results (habit / top-up 4-week avg / top-up our forecast / optimizer):**
  - fill rate 84.0% / 95.1% / 97.3% / 89.5%; stockout days 10,746 / 3,154 / 1,621 / 5,941; lost units 81,747 / 25,129 / 13,784 / 53,843;
  - waste units 15,543 / 56,867 / 90,586 / 29,779; average stock value 95,720 / 110,302 / 164,253 / **71,969**; days of stock 6.5 / 7.5 / 11.6 / **4.8**; turnover 49 / 48 / 33 / **70** per year;
  - cash cost 802,799 / 984,336 / 1,106,297 / 835,699; net cost 786,939 / 951,135 / 1,009,175 / 861,343; lost-sales penalty (1.5 × price per lost unit) 225,007 / 61,185 / 32,537 / 137,957; **net cost + lost-sales penalty** 1,011,946 / 1,012,320 / 1,041,712 / **999,300** (lowest, by about 1.2%).
- **What it shows:** the forecaster's value is clear (top-up with our forecast vs the 4-week average: fill 95.1% → 97.3%, stockout days halved), though that rule also holds 49% more stock and wastes more. The optimizer holds the least stock and has the best combined cost-plus-lost-sales score, but it serves fewer customers than both top-up rules and its fill drifts down from 94.5% in week 1 to about 86–89% later.
- **Why its service is lower (from the run):** 88% of its lost units (47,444 of 53,843) happened while the home warehouse still had stock, so the main cause is how much it sends to stores, not the warehouse buying. The model only plans for the median (q50) demand and charges a soft 0.5 × price for falling below each product's 80–95% line, so with these cost settings running a little short is often "cheaper" than holding more. The warehouse-buying weakness (#34) is real but smaller: 6,399 lost units with an empty warehouse (vs about 1,000 for the other methods), 21 of 240 warehouse-items never bought and 10 bought once; overall it bought 72% as much from suppliers as the forecast top-up rule.
- **Fairness rule for any fix:** its cost settings were fixed before the test window. Any change (e.g. a stronger shortfall penalty, valuing warehouse stock beyond 7 days) must be chosen on data before 1 Nov 2015 (a validation-window replay), not by looking at these test numbers, and is added as a switch, with both versions reported.

### 36. The final project lives in its own folder, and the frozen model's fingerprint no longer depends on the folder (user's call 2026-11-02; built)
- **What:** the final model (2-day forecaster + `store_ilp` with the service floor and spoilage cap, plus `store_model` and the simple rules as baselines) was copied into a clean folder `CS2105/RetailOps/` for GitHub, with two notebooks that call the project code. Old checks, teammate packages, early plans and superseded runs moved to `CS2105/_archive/` (nothing deleted except caches).
- **Problem found by the move:** the frozen forecaster's config fingerprint included the full path to `calendar.csv`, so in any other folder (or a GitHub clone) the model refused to load ("config changed, freeze again") although nothing had changed.
- **Fix:** path settings now count by file name only in `forecast/freeze.py config_sha256`; the three frozen manifests (test model, holdout blocks 1 and 2) were re-stamped, keeping the old value under `config_sha256_before_move`. New test `test_config_hash_same_in_any_folder`.
- **Why this is safe:** no model, feature, setting or result changed. Proof: forecasts regenerated from the frozen models match the stored files exactly (test window 134,400 rows; holdout blocks 1 and 2, 117,600 rows each), and a real config change is still refused. Full suite 299 passed, 1 skipped.
- **Alternative rejected:** leave the code alone and only run from the old folder; that would break the project for anyone cloning it.

---

## Likely viva questions, short answers

- **"How do you know the model doesn't cheat?"** Every store input is at least 7 days old, checked by a registry (`FEATURE_SOURCES`) and by tests that poison all future data with ×1000 values and check the inputs don't change. Windows are split by date, the holdout is sealed, and all statistics come from training rows only.
- **"Why not just forecast tomorrow?"** The supplier takes 5 days, orders come in cases and trucks, perishables expire, and festivals need build-up. We plan 7 days and act on 1.
- **"Why is your model better?"** It must beat the teammate's 4-week-average rule on both folds, including a festive fold, or we stop and diagnose (Task 4 gate).
- **"Why quantiles?"** Different products are stocked to different safety levels (80/90/95%), so the order model needs those lines directly.
- **"What did you try that didn't work?"** Promo inputs, price vs department, a 52-week normal price, target-day price, more trees, log target, seed averaging, baseline blending. All gave no consistent gain.
- **"What would you do with more time?"** The horizon-aware model (about 5% better over the week, 12% for tomorrow) and the optional 5-day total forecasts.
