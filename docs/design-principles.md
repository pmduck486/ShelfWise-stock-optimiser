# RetailOps forecaster: design and architecture principles

> **What this is:** the principles the ML forecaster (Lump 1) is built on, and for each one *how it actually works in the code*: which file, which function, which test. Written for viva and presentation prep. The companion file `decisions.md` lists each individual choice with its evidence.
>
> **Status when written (2026-10-30, updated 2026-10-31):** Phases A–E are built (`forecast/data.py`, `baseline.py`, `metrics.py`, `evaluate.py`, `features.py`, `model.py`, `experiments.py`, `postprocess.py`, `review.py`, `tune.py`, `calibrate.py`, `freeze.py`, `predict.py`, `write_forecasts.py`, `report.py`); 163 tests pass on the laptop. Parts still unbuilt (optional layers) are marked **(planned)** and describe `plan/forecaster-build-plan.md`.

---

## The big picture

```
 sales.parquet ─┐                                     ┌─► forecasts.parquet ─► Order ILP (Lump 2)
 items.csv ─────┼─► load safely ─► features ─► 4 models ─► post-process ─┤                      ─► Replay (Lump 4)
 calendar.csv ──┘   (data.py)    (features.py) (model.py)  (postprocess.py) └─► scores ─► slides / dashboard
                                      ▲                         ▲
                     baselines + scorer (baseline.py, metrics.py, evaluate.py): the bar to beat
                          config.yaml (every number, every switch)
```

**In one sentence:** a pipeline of small modules, each with one job, where every input is computed backwards from the target day using only data a store manager would have had, every number lives in config, and every rule is enforced by a test.

---

## The principles at a glance

| # | Principle | One-line meaning |
|---|---|---|
| P1 | Fairness first: no peeking | The model only uses what a manager knew on the morning of ordering |
| P2 | Target-day-backwards inputs | One model serves all 7 days; a date gets the same forecast from any origin |
| P3 | Statistics from the past only | Codes, normal prices, zero-run rules come from training rows only |
| P4 | Rules enforced by tests | Each fairness rule has a test that fails the build if broken |
| P5 | One module, one job | load → baseline → features → model → post-process → calibrate → freeze → write |
| P6 | Contracts between lumps | Teams connect only through fixed-column files |
| P7 | Config holds every number | No constants in code; change behaviour by editing `config.yaml` |
| P8 | Layers are switches, default off | The core always runs alone; extras can be turned on one at a time |
| P9 | Beat an honest bar | The model must beat the teammate's rule on both folds, or we stop; a change stays only if it helps both folds |
| P10 | Honest uncertainty | Four quantile lines, checked for coverage and calibrated |
| P11 | Missing means missing | No row or a closed day is NaN, never a fake 0 |
| P12 | Fail loudly, not quietly | Bad input raises an error instead of being trimmed or guessed |
| P13 | Reproducible and frozen | Fixed seed, saved model + manifest, replay refuses a leaky model |
| P14 | Simple and explainable | Library models, light tuning, every input has a reason |
| P15 | Report everything | Bad results and weak groups are shown, never hidden |
| P16 | Don't break teammates | Shared files and the stub contract stay unchanged |
| P17 | The two models share one contract and one config | Same forecast file, same service lines, one config, a solver head start that works on the laptop (built 2026-11-01; replay reading both order tables planned) |

---

## P1. Fairness first: the model never peeks at the future

**Idea.** If the forecaster uses even one day it could not have known, every replay result is fake. So the whole design starts from one question: *what did the store manager know on the morning they placed the order?*

**The rules** (`plan/forecast-training-plan.md`, R1–R12), the main ones:
- **R1** the test window and holdout are locked; no training step may read them.
- **R2** store data must be at least 7 days older than the target day.
- **R3** calendar facts may be same-day (published in advance).
- **R4** statistics come from training rows only.
- **R6** tuning uses validation only. **R7** freeze before each replay. **R12** reproducible.

**How it works in code.**
- `forecast/data.py::load_sales` raises if `sales.parquet` has any row in the sealed holdout, instead of quietly trimming it.
- `forecast/data.py::check_cutoff(sales, cutoff)` raises if a frame has any row on or after a model's cutoff. Every fit (`fit_models`) calls it first.
- `forecast/data.py::val_folds` checks both folds are in order, train_end is the day before val_start, and every fold ends before the test window.
- Time windows come only from `config.yaml → data.windows`; nothing in code hard-codes a date.

**Tests:** `test_cutoff_guard_raises`, `test_cutoff_guard_allows_day_before`, `test_dev_loader_never_returns_holdout_rows`, `test_val_folds_in_order`.

---

## P2. Inputs are computed backwards from the target day

**Idea.** Every input for "sales on date *d*" is read from dates at least 7 days before *d* (store data) or from *d* itself (calendar). It never depends on which morning (origin) asked.

**Why it matters.** Three things follow from this one property:
1. **One model serves the whole 7-day horizon.** No separate "1 day ahead" and "7 days ahead" models.
2. **The future-poison test is possible:** if we replace every value after the origin with garbage, the forecasts must not change.
3. **Validation is scored once per target day**, not once per origin, which keeps scoring simple.

**How it works in code** (`forecast/features.py::build_features`):
- It builds a **dense grid**: one row per store-item, one column per calendar day. "7 days back" is then a column offset, so it means 7 *calendar* days, not 7 *rows* (a missing day can't shift the window).
- The grid's newest column is `g1 = newest target − min_lag_days`. **Sales rows newer than that are never placed in the grid**, so the code physically cannot read them.
- `FEATURE_SOURCES` is a registry: for each store-data input, the fewest days back it reads (`lag_7: 7`, `roll_mean_28: 7`, …). `feature_names` refuses to run if config asks for an input that is not in the registry.
- Calendar inputs come from `_calendar_inputs`, which reads only `calendar.csv`.

**Tests:** `test_future_poison_features` (units ×1000 and prices NaN after the origin, three origins including Thanksgiving week, features identical), `test_feature_age_registry`, `test_feature_age_by_perturbation` (change target−6 → nothing moves; change target−7 → `lag_7` moves), `test_cross_item_perturbation` (other items' recent data can't leak through the store rank), `test_same_target_alone_or_in_batch` (a day's inputs don't depend on which other days are built with it).

---

## P3. Statistics come from the past only (R4)

**Idea.** Any number computed across many rows (category codes, each item's normal price, which zero runs are "long", calibration factors) is computed on the training rows of the model being fitted, then applied unchanged to later dates. Otherwise future data sneaks in through an average.

**How it works in code.**
- `fit_encoders(sales, *, cutoff)` and `zero_run_stats(sales, cfg, *, cutoff)` take a **required** `cutoff` keyword: you cannot call them without saying which rows they may use. They filter to `date < cutoff` first.
- A value never seen before the cutoff (e.g. a new item) gets no code (NaN), so the model treats it as unknown.
- `training_rows` refuses zero-run stats whose cutoff is missing or not after the training end.
- Calibration factors are fitted on the validation folds only.

**Tests:** `test_encoders_ignore_post_train_rows`, `test_long_zero_rule_uses_train_only`, `test_statistics_need_a_cutoff`.

---

## P4. Every rule is enforced by a test

**Idea.** A rule written only in a document will eventually be broken by accident. So each fairness rule has a pytest that fails the build if broken, and nothing is trained until the fairness tests pass (gate B).

**How it works.**
- Tests are written **first** for each task (failing test → implement → pass).
- `tests/test_forecast_fairness.py` holds the fairness tests; `tests/test_forecast.py` holds behaviour and edge-case tests (late launches, zero prices, 25 Dec).
- After Phase B an independent review deliberately broke the code 25 ways (mutations); all 25 were caught by the tests.

| Rule | Enforced by |
|---|---|
| R1 locked windows | `test_cutoff_guard_raises`, `test_dev_loader_never_returns_holdout_rows` |
| R2 7-day rule | `test_future_poison_features`, `test_feature_age_registry`, `test_feature_age_by_perturbation` |
| R3 calendar only | `test_calendar_features_ignore_sales` |
| R4 train-only stats | `test_encoders_ignore_post_train_rows`, `test_long_zero_rule_uses_train_only` |
| R5 product choice | `test_selection_ignores_test_and_holdout_sales` (data prep) |
| R6 tuning on validation | `test_tune_reads_no_test_rows` |
| R7 frozen model | `test_frozen_guard_refuses_late_cutoff` |
| R12 reproducible | `test_reproducible_same_seed` |
| Lines never cross | `test_postprocess_uncrosses`, `test_sanity_full_output` |

---

## P5. One module, one job (a pipeline)

**Idea.** Small modules with clear inputs and outputs are easier to test, explain and hand to different people.

| Module | Job | Status |
|---|---|---|
| `forecast/data.py` | load config, sales, items, calendar safely; cutoff guard; windows and folds | built |
| `forecast/baseline.py` | the two baselines (naive, stub) | built |
| `forecast/metrics.py` | pinball loss, coverage, MAE; product groups; `score` | built |
| `forecast/evaluate.py` | CLI: score a method on both folds, save JSON | built |
| `forecast/features.py` | encoders, zero-run stats, training rows, `build_features` | built |
| `forecast/model.py` | fit the four quantile models; `predict_lines` | built |
| `forecast/postprocess.py` | sort lines, clip at 0, zero closed days | built |
| `forecast/tune.py` | ≤ 6-setting grid on both folds | built |
| `forecast/calibrate.py` | scaled correction per line | built |
| `forecast/freeze.py` | freeze + manifest + guards (`load_frozen`, `check_platform`, `config_sha256`) | built |
| `forecast/predict.py` | `predict(sales, origin)`: contract columns, only sales before origin, baseline fallback for no-history store-items | built |
| `forecast/write_forecasts.py` | 56 origins → `data/processed/forecasts.parquet`, fast path `forecasts_for`; laptop only | built |
| `forecast/report.py` | `summary.md` + charts for the slides | built |

**How it shows in code:** each file starts with a plain-language docstring saying what it does and which rules it follows; functions pass plain pandas DataFrames, not hidden global state.

---

## P6. Lumps connect only through fixed contracts

**Idea.** Five teams work in parallel. They never wait for each other because they agree file formats in the first hour, and each team first ships a stub that writes fake numbers in the right format.

**How it works.**
- `forecasts.parquet` columns are fixed: `origin, date, store_id, item_id, q50, q80, q90, q95`, dates `datetime64[ns]`, ids `str`, lines `float64` (`plan/build-roadmap.md`).
- The order-model teammate builds against `forecasts_stub.parquet` today; the real file will have **exactly the same keys and dtypes**, so she swaps it in with no code change.
- Interfaces: `forecast.predict(sales, origin, horizon=7, quantiles=...)` and the replay's fast path `forecast.forecasts_for(origin, cfg)` (reads the written file by origin).
- `sales.parquet` stores dates as `datetime64[us]`, so the writer casts to `[ns]` to match the stub.

**Tests:** `test_contract_matches_stub` (columns, dtypes, 470,400 rows = 56 origins × 7 days × 1,200 store-items, same keys as the stub).

Built: forecasts.parquet passes test_contract_matches_stub (same columns, dtypes, 470,400 rows, keys as the stub).

---

## P7. Config holds every number

**Idea.** No magic numbers in code. Anyone can see and change an assumption in one place, and the frozen model records a hash of the config it used.

**How it works.**
- `config.yaml → forecast:` holds quantiles, horizon, the 7-day rule, lags, window lengths, closed and dropped dates, the 90-day zero-run rule, the folds, seed, model settings, tuning grid, service lines and layer switches.
- `feature_names(cfg)` builds the input list from config (`lags: [7, 14, 21, 28]` → `lag_7 … lag_28`).
- `load_config` makes paths absolute relative to config.yaml and checks consistency (every closed date must also be a dropped date).
- Assumptions are labelled `ASSUMPTION:` in config comments (e.g. perishable shelf lives, warehouses).

**Test:** `test_forecast_config_has_all_keys`, `test_other_sections_untouched` (the forecaster's section never alters `data:`, `network:` or `stubs:`), `test_closed_dates_are_dropped`.

---

## P8. Layers are switches, default off

**Idea** (CLAUDE.md rule 2). The core must always run and pass its tests alone. Each extra feature is a switch in config, added one at a time with its own test, so a half-built layer can never break the core, and the replay can show results with each layer on and off.

**How it works.**
- `forecast.layers`: `p7_empty_shelf`, `random_product_sets`, `multi_day_totals`, `demand_multiplier`, all `off`.
- `features.py::empty_shelf_filter` is the P7 hook: while the switch is off it returns rows unchanged; switched on before P7 exists it raises `NotImplementedError` instead of silently doing nothing.
- Optional layers C and M are built only if the user says yes.

**Tests:** `test_layers_default_off`, `test_p7_hook_off_is_identity`.

---

## P9. Beat an honest bar, on more than one period

**Idea.** A model is only "good" compared with something. We compare against the rule the order model already uses (the teammate's 4-week average), not just a weak strawman, and on two validation periods, one of them festive.

**How it works.**
- `baseline.py` calls the teammate's own `stubs.make_stubs.forecast_targets` (read only) for the stub baseline, so the bar is exactly her file, not our copy of it. The naive baseline passes the same function a copied config with one lag.
- `evaluate.py` scores both baselines on both folds plus fold 1's peak days and writes `outputs/forecast/scores/baseline_val.json` and `stub_val.json`.
- Gate C (passed 2026-10-30): q50 must beat both baselines on both folds, or work stops for diagnosis. Model q50 MAE 4.12 / 4.21 (fold 1 / fold 2) vs stub 4.39 / 4.72 and naive 4.88 / 5.21.
- Both-folds rule (built in Tasks 4 and 6): every change tried is logged in `outputs/forecast/scores/experiments.csv` and kept only if it lowers q50 pinball on both folds (`keep_change` in `forecast/experiments.py`); the tuning winner must not be worse than the current settings on either fold.
- Every line vs the baselines (built, Task 5; all six upper lines win): q80/q90/q95 pinball compared with both baselines per fold (`compare_lines` in `forecast/review.py`); a losing line is flagged at the Task 7 gate. The lines that drive orders are the upper ones, not q50.

**Current bar (MAE units/day, fold 1 / fold 2):** naive 4.88 / 5.21, stub 4.39 / 4.72.

**Model now (3-seed mean, after fix C+A, decision 26):** q50 MAE 3.93 / 4.15; every upper line beats both baselines. One fixed rule sits after the model: q50 = 0 when the 28-day average ending 7 days back is 0 (`dead_item_rule`), the stub's own answer for items that stopped selling; the upper lines stay with the model.

**Tests:** `test_stub_baseline_matches_stub_file`, `test_baseline_hand_calc`, `test_stub_baseline_hand_calc`.

---

## P10. Honest uncertainty: four lines, checked and calibrated

**Built 2026-10-31 (Task 7, `forecast/calibrate.py`):** `fit_shifts` pools both folds' validation rows (25 Dec left out) and fits one factor per upper line only if its coverage is more than 3 points off; today every factor is 1 because the untuned lines already land within 3 points. `cross_fold_check` fits on one fold and judges the other, so the gate is not marked on the days it was fitted on. `apply_shifts` scales around the original q50, then post-processes, so lines never cross.

**Idea.** The order model stocks each product to a safety level, so the forecaster must say "on 95% of days sales will be at most X", and that claim must be *true* on real data.

**How it works.**
- Four models, one per quantile, trained with **pinball (quantile) loss** (`metrics.py::pinball`).
- **Coverage** = share of days at or under the line (`metrics.py::coverage`); a good 90% line covers about 90%.
- `score` reports coverage for all / perishable / slow and `cover_used_line`: coverage of the line each product is actually stocked to (`service_line`: perishable and top 30 → q95, slow → q80, else q90).
- Post-processing sorts each row's four lines so they never cross and clips at 0; calibration stretches each product's gap above q50 by one factor *k* per line if coverage is off by more than 3 points. Sorting runs **after** calibration.

**Tests:** `test_pinball_example`, `test_coverage_basic`, `test_cover_used_line`, `test_service_line_rules`, `test_scale_fixes_coverage_on_synthetic`, `test_shifted_lines_still_ordered`.

---

## P11. Missing means missing, never a fake zero

**Idea.** A day with no row (item not on sale yet) or a closed day (25 Dec) is *unknown*, not "sold 0". Treating it as 0 would teach the model fake slumps.

**How it works** (`features.py`):
- The dense grid starts as all NaN; only real rows are filled in. Late-launch items simply have NaN early lags (gradient-boosted trees handle NaN natively).
- Closed days: `units[:, closed_mask(...)] = NaN`; their prices are kept (they are normal).
- Window rule: averages, the 28-day spread and the store rank use the days present and are NaN only if fewer than half the window is present (4 of 7, 14 of 28). `_window_sums` counts present days, sums and squares with cumulative sums, so this is fast and exact.
- Divisions are guarded: `price_chg_4w = 0` and `price_rel_item = 1` when a price is missing or 0; `momentum = 1` when the 28-day mean is 0.
- Prices are summed in **whole cents** so averages are exact, and a day's input never depends on which batch it was built in.
- `closed_mask` in `metrics.py` is the single place every part uses to find closed days, so training, features and scores can't disagree.

**Tests:** `test_late_launch_features_nan_not_error`, `test_zero_price_change_no_inf`, `test_sale_features_no_inf`, `test_closed_day_is_missing_in_features`, `test_window_half_rule_std_and_momentum`, `test_christmas_dropped_from_training_rows`, `test_score_skips_closed_days`.

---

## P12. Fail loudly, not quietly

**Idea.** A silent fix hides a bug; an error forces someone to look.

**How it works in code** (all raise `ValueError` with a plain message):
- `load_sales`: file reaches into the holdout → "rebuild it without --include-holdout".
- `check_cutoff`: rows on/after a cutoff → says how many and the latest date.
- `load_config`: a closed date that is not dropped from training.
- `val_folds`: folds out of order or overlapping the test window.
- `feature_names`: an input missing from `FEATURE_SOURCES`.
- `build_features`: a target day missing from `calendar.csv`.
- `training_rows`: zero-run stats from the wrong cutoff.
- `segments`: an item missing from `items.csv`.
- `load_frozen` (built, #29): model cutoff after the replay start, or config / sklearn version changed. `predict` with an origin inside training data is refused.

---

## P13. Reproducible and frozen (built 2026-10-31, Tasks 8–9)

**Idea.** Same config → identical forecasts, and the replay can prove which model it used.

**How it works.**
- `seed: 1` in config for every model. A rerun on the same machine is identical; the laptop and the cloud differ in the third decimal (q50 MAE 4.108 vs 4.106 on fold 1), so the final frozen model is made on one machine.
- `freeze` retrains on train + validation through 2015-10-31 (cutoff 2015-11-01, the first unseen day) with the tuned settings and calibration, then saves `bundle.joblib` and `manifest.json` {cutoff, windows, features, quantiles, params, shifts, seed, sklearn_version, platform, train_rows, config_sha256, created}.
- Laptop only: `freeze` and `write_forecasts` raise unless platform.system() equals forecast.freeze_platform (Windows); the real freeze took 126 s on 1,226,019 rows.
- Config fingerprint: `config_sha256` hashes the `forecast:` section + `data.windows` as sorted JSON, with the paths (model_dir, scores_dir) and freeze_platform left out, so comments and line endings don't change it.
- `load_frozen` refuses a manifest whose cutoff is after the replay start, or whose config hash or sklearn version differs.
- Train once for the test replay; the holdout gets one more retrain with the same settings, run once after code freeze (R9b).
- **Order model runs (planned, decision #32):** the solver stops on a time limit, so two runs of the same command can differ slightly (a 3-day check differed on day 3). Builds and tests use a 15 s budget per day; the core results in the slides come from **one** logged run at the production 60 s budget with a 1% gap target, and 15 s numbers are never mixed into them.

**Tests:** `test_manifest_has_all_fields`, `test_frozen_guard_refuses_late_cutoff`, `test_frozen_guard_refuses_changed_config`, `test_frozen_guard_refuses_other_sklearn`, `test_freeze_refuses_other_platform`, `test_reproducible_same_seed`, `test_frozen_bundle_carries_shifts` (`tests/test_forecast_freeze.py`).

---

## P14. Simple and explainable

**Idea.** The team is new to ML and must explain everything in the viva. Prefer a well-known library model, light tuning and inputs that each have a one-line reason.

**How it shows.**
- scikit-learn `HistGradientBoostingRegressor`, no custom ML code; at most 6 tuning combinations.
- Every input has a reason (see `decisions.md` §13) and inputs that tested no better were removed (promo inputs, price vs department, target-day price, 52-week normal price).
- Ideas that would cost a lot for a small gain were dropped (horizon-aware model, feature cache).
- Every module and config line has a plain-language comment.

---

## P15. Report everything, including the bad parts (R10)

**Idea.** Results are only believable if weak spots are shown too.

**How it works.**
- `score` always reports all / perishable / slow, and `evaluate.py` adds fold 1's peak days (2014-11-26, 12-23, 12-24) separately.
- Products are never dropped after seeing results.
- Test-window scores are produced only after the replay; every test run is logged; the holdout runs exactly once and is reported as it comes out (R9a, R9b).
- **The first full replay is reported as it came out (decision #35):** the optimizer has the lowest stock and the lowest cost-plus-lost-sales score but lower service than both top-up rules; both facts go on the slides. Fixes to its settings must be chosen on data before the test window, not on these numbers.
- **Write down the holdout runs first (planned, decision #33):** before the holdout is opened, a dated file lists exactly what it will run: the core with every switch off and the final version with the finished layers on, each against the same 3 simple rules. Both results are reported as they come out; no version is added, dropped or re-tuned afterwards. Until then, layers are judged on the test window only.
- Known side effects are written down (e.g. the baselines' low early-January forecasts after 25 Dec).
- Error review (built, Task 5; the worst store-items are items that stopped selling for good): the 20 worst store-items per fold, ranked by error relative to their own sales, with the stub's error and their traits (`worst_items` in `forecast/review.py`).
- Leak check on the trained model (built, Task 5; passed at gate C): a shuffle test on fold 2 shows which inputs the model relies on (`shuffle_importance`); a fold 2 q50 MAE below 3.6 is treated as a suspected leak and stops the work.

---

## P16. Don't break teammates' inputs

**Idea.** The order model is built on another laptop. Anything she depends on must stay exactly as it is.

**How it works.**
- `sales.parquet`, `items.csv`, the stub files, `data.windows` and the `data:` / `network:` / `stubs:` config sections are read, never written, by the forecaster. Fold 1 lives only in the `forecast:` section.
- The stub baseline calls her function instead of copying it.
- The only change she will see: real forecasts are 0 on 25 Dec, which her spec already handles.

**Test:** `test_other_sections_untouched`.

---

## P17. The two models share one contract and one config (built 2026-11-01; the replay part planned)

**Idea.** The forecaster and the order model were built on different machines. They only work together if they agree on three things: the file the forecaster writes, which line each product is stocked to, and how the solver runs on the machine that produces the final results. Each agreement is checked by a test or a measured run, not by trust. Details and numbers: `decisions.md` #30 and #31.

**How it works.**
- **One forecast file:** `forecasts.parquet` has exactly the stub's columns, dtypes and 470,400 keys, so the order model reads the real file with no code change (`test_contract_matches_stub`; checked against her final zip on 2026-11-01).
- **One rule for service lines:** `forecast.service_lines` (the line the forecaster is scored on) must equal `order_model.service_levels` (the line the order model stocks to): perishables and the top 30 on q95, under 5 units/day on q80, the rest on q90. They agree for all 120 items, and since both sections sit in the same `config.yaml`, `test_forecaster_service_lines_match_order_model_service_levels` (in `tests/test_core_spec_gaps.py`) fails the build if anyone changes one side only (built).
- **One config:** `config.yaml` now holds `data:`, `network:`, `stubs:`, `forecast:` and the order model's `order_model:` section, which was appended word for word. Adding it changed none of the other sections and not the frozen forecaster's fingerprint (built).
- **The solver's head start must work on the freeze platform (Windows):** the order model hands CBC a quick starting plan. On Windows that hand-over failed silently, because CBC couldn't open PuLP's absolute temp path. `optimize/cbc.py` points PuLP at a relative scratch folder (`.cbc_scratch`, gitignored); PuLP uses random file names and deletes the files, so runs can't clash and nothing piles up (built; keepFiles was tested and rejected, see `decisions.md` #31). This matters because the final results are produced on the laptop, the same rule as the forecaster's laptop-only freeze (P13).
- **Read both order tables:** `policy.decide` returns a Decision with store orders **and** warehouse orders; the replay carries out both (built 2026-11-01, Lump 4).
- **One engine, one set of physics for every method (built, decision #34):** the 3 simple rules and the optimizer all go through her `execute_day`, so deliveries, Christmas, whole packs, the minimum order, warehouse stock, trucks, the budget, sales and costs are identical. The simple rules see sales only through a `PastSales` object that refuses any day on or after today, so peeking is impossible by construction (future-poison test for all 4 methods).

**Tests (all built and passing, 229 in the full suite):** `test_contract_matches_stub`; `test_forecaster_service_lines_match_order_model_service_levels` and the other 4 tests in `test_core_spec_gaps.py`; `test_warm_start_is_read_by_cbc_single_model` and `test_warm_start_is_read_by_cbc_in_every_dc_block` (`test_warm_start_reaches_cbc.py`, which fail on the unfixed code).

---

## How a forecast is made, start to finish (for a slide)

1. **Load safely** (`data.py`): config, sales (holdout refused), items, calendar.
2. **Fit statistics on training rows only** (`fit_encoders`, `zero_run_stats`, with a cutoff).
3. **Pick training rows** (`training_rows`): drop 25 Dec (long zero runs are kept since decision 26; the drop is a switch).
4. **Build inputs** (`build_features`): dense grid, 25 Dec as missing, lags / averages / prices ending 7 days back, calendar for the target day, who-codes; float32.
5. **Train four quantile models** (`model.py`), one per line, all items together.
6. **Post-process** (`postprocess.py`): sort lines, clip at 0, set 25 Dec to 0.
7. **Tune and calibrate** on both validation folds (`tune.py`, `calibrate.py`).
8. **Freeze** with a manifest (`freeze.py`, laptop only).
9. **Write** `forecasts.parquet` for 56 origins × 7 days × 1,200 store-items = 470,400 rows (`write_forecasts.py`, built 2026-10-31).
10. **Score** against both baselines; test-window scores only after the replay.
