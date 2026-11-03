# RetailOps dashboard

An animated, static website that shows the RetailOps results: the method comparison, the
8-week replay as a calendar, one store-item decision step by step ("decision theatre"), the
forecaster's ranges, shadow prices and what-ifs, the layer tests, and a plain-language pipeline.

Every number comes from files already saved in `RetailOps/` (replay outputs, forecasts, items,
the M5 calendar, forecaster scores). Nothing is re-run or re-solved. The one exception is the layer
table, whose runs were made in a cloud copy and are not saved here: its numbers are copied from
`docs/decisions.md` #40, and `test_export.py` checks that each one still appears in that file.

Stack: Vite + React 18 + TypeScript + Tailwind 4 + `motion` (animations) + Apache ECharts (charts).
Fonts (Fraunces, IBM Plex Sans/Mono) are bundled from npm; the built site makes no network calls
except to load its own JSON.

## 1. Regenerate the data

From `RetailOps/dashboard` with the project venv (about 50 s):

```bash
../../.venv/Scripts/python export_data.py          # Windows (Git Bash)
../../.venv/bin/python export_data.py              # macOS / Linux
../../.venv/Scripts/python -m pytest test_export.py -q
```

It writes `public/data/` (about 21 MB in 189 files; the day and series files load only when
opened). The output is deterministic: the same inputs give byte-identical files.

Two self-checks run during the export:
- **Targets:** each store-item's target line and spoilage cap are recomputed with the project's own
  `replay.store_model.StoreModel.lines_and_caps` on the same forecast file and compiled config the
  replay used. As a check, the rule-only `store_model` orders are rebuilt from those targets and the
  morning stock: 3,600 of 3,600 store-item-days match its saved `order_plan.csv`.
- **ILP item term:** for every store-item-day, the cartons the ILP chose are compared with the minimum
  of that item's own term of the ILP objective (the orange curve on the Day decision page):
  66,000 / 66,000 (test), 58,798 / 58,800 (holdout 1), 58,796 / 58,800 (holdout 2).

## 2. Run and build

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # type-checks (tsc) then builds into dist/
npm run preview    # serves dist/ on http://localhost:4173
```

## 3. Deploy to GitHub Pages

`vite.config.ts` sets `base: './'` and the app uses hash routes (`#/overview`, `#/workflow`, ...),
so `dist/` works from any sub-path.

- **Simplest:** build, then copy the contents of `dist/` to a `docs/` folder or a `gh-pages` branch
  and choose it under *Settings → Pages*.
- **GitHub Actions:** a workflow that runs `npm ci && npm run build` in `dashboard/` and publishes
  `dashboard/dist` with `actions/upload-pages-artifact` + `actions/deploy-pages`. `public/data/` must
  be committed (or the export run in the workflow, which needs the Python venv and the outputs).

## Pages

| Page | What it shows | Main data |
|---|---|---|
| Overview | Store ILP KPIs (tickers), 5-method bars, daily fill vs the best simple rule; period toggle | summary.json, calendar.json |
| Workflow calendar | Each replay day coloured by fill rate, event / closed / no-order / service-floor markers, hover tooltip, play button that steps the rolling replay; click a day to open its decision (morphing dialog) | calendar.json |
| Day decision | 5 steps for one store-item: forecast ladder, target line and why, carton options priced, store limits and shadow prices, outcome vs other methods | day/<period>/<date>.json, items.json |
| Forecast explorer | Fan chart q50 to q95 vs actual with a zoom slider; coverage by period; MAE and pinball vs two baselines | series/<period>/<store>.json, forecast.json |
| Shadow prices & what-ifs | Truck / space / budget use and duals; service-floor heatmap; what-if dot plots and table | shadow.json |
| Layers | Layer on/off results with keep/drop verdicts | summary.json (from docs/decisions.md #40) |
| How it works | Pipeline Data → Forecaster → Targets → ILP → LP duals → Replay, with a worked perishable example | items.json, summary.json |

Periods: **test** = 1 Nov to 26 Dec 2015 (56 days; the saved `store_ilp` run is the *core* ILP with
the layers off), **b1** = holdout block 1, 27 Dec 2015 to 13 Feb 2016, **b2** = holdout block 2,
14 Feb to 2 Apr 2016 (49 days each; `store_ilp` is the main model with the service floor and
spoilage cap on).

Money is in M5 shelf-price units: Walmart prices in US dollars (`config.yaml` calls them
"USD-equivalent model price units"; the project notes write them as Rs). The site shows them with `$`.

## JSON schema (`public/data/`)

All files are UTF-8 JSON with sorted keys. Dates are ISO `YYYY-MM-DD`. `null` = not applicable.
Store order: `CA_1..CA_4, TX_1..TX_3, WI_1..WI_3`. Item order: `items.json → items` (sorted by id).

### `summary.json`
```
periods: { test|b1|b2: {
  label, range, weeks, note, best_simple: <method id>, store_ilp_layers,
  methods: [ { id, label, fill (share 0-1), lost, waste, stock (avg store stock value),
               net_cost, penalty (lost-sale penalty), total (= net_cost + penalty),
               vs_best_simple (% vs the cheapest of the 3 simple rules on total),
               stockout_days, orders, trucks, purchase, turnover, days_of_stock,
               n_vars, n_int, solve_seconds, audit_passed, rules_passed, service_met_share, source } ] } }
period_order: ["test","b1","b2"]
layers: { source, test (text of the pre-set test), rows: [ { id, label, floor, spoil, stockcap,
          fill (%), lost, waste, stock, total, verdict } ] }
money: { unit, note }, totals_definition
```
Method ids: `last_week` (habit), `topup_avg`, `topup_forecast`, `store_model`, `store_ilp`.
Sources: `outputs/replay/store_ilp_2026-11-02/results_table.csv` (store_model, store_ilp, test),
`outputs/replay/store_only_2day_2026-11-02/results_table.csv` (simple rules, test),
`outputs/replay_holdout/holdout_block{1,2}/results_table.csv`, each run's `kpis.json` (audit flags).

### `calendar.json`
```
periods: { <p>: { label, range, start, end, layers, self_check?, days: [ {
  date, dow (0 = Monday, Python), closed, can_order, cover_days,
  demand, served, lost, fill, waste, cartons, units, trucks,
  floor_n, floor_mean, floor_max, spoil_n, spoil_mean,     // service floor / spoilage cap duals that day
  binding, n_vars, n_int, ilp_seconds,                     // store ILP size and time (null if no solve)
  events: [ {name, type} ], snap: {CA, TX, WI},            // from data/m5/calendar.csv
  methods: { <method id>: [fill, lost, waste] },           // each method's own daily_metrics.csv
  limits: { truck|space|budget: [mean use share, max use share, binding store count] } } ] } }
stores, method_order, period_order
```
Sources: `<run>/store_ilp/{daily_metrics,inventory_ledger,order_plan,deliveries,shadow_prices}.csv`.

### `day/<period>/<date>.json` (one per replay day, ~100 KB)
```
date, period, can_order, closed, cover: [dates the order must cover], fc_dates: [today, tomorrow],
floor_on, floor_share, spoil_in_ilp,
stores: { <store>: { truck|space|budget: [limit, used, shadow price], trucks, cartons } },
layers: [ [row, kind (0 floor, 1 spoil), shadow price, limit (required units / cap room), used units] ],
cols: ["q50a","q80a","q90a","q95a","q50b","q80b","q90b","q95b","act","pos","line","cap","cases",
       "lost","waste","alt_last_week","alt_topup_avg","alt_topup_forecast","alt_store_model"],
rows: 1,200 arrays in `cols` order, row = store_index * 120 + item_index
```
`q..a` / `q..b` = forecast lines for today / tomorrow from the forecast with origin = that date (the
one the replay used). `act` = units wanted that day, `pos` = stock on hand + arriving today when the
order was placed (ledger `morning_units`), `line` = rule-1 target summed over the cover days, `cap`
= spoilage cap (null for long-life items, and both null on days no order was possible), `cases` =
store ILP cartons ordered, `lost` / `waste` from the ledger, `alt_*` = cartons each other method
ordered for that store-item that day in its own replay.

### `series/<period>/<store>.json`
`{ store, period, dates: [...], items: { <item>: { q50, q80, q90, q95, act, stock, order } } }`:
the same-day forecast lines, actual demand, morning stock and cartons ordered, one value per day.

### `items.json`
```
items: [ { id, dept, rank, avg_daily (units per store per day before the test window),
           perishable, shelf_life, upc (units per carton) } ]
params: { <period>: { items: [ { price, buy, hold (per unit-day), service_level,
          level (Cu/(Cu+Co), the newsvendor level) } ],      // same order as items
          lost_sale_penalty_share, waste_disposal_share, order_line_cost, truck_trip_cost,
          truck_capacity_cases, store_min_order_cases, shelf_days, shelf_days_perishable,
          shelf_min_cases, space_factor, budget_factor, time_limit_s,
          service_floor: {enabled, share, gap_penalty_factor}, spoilage_cap_in_ilp,
          stock_value_cap, closed_days } }
stores, source
```
From `data/processed/items.csv` and each run's `compiled_config.yaml` (prices are calibrated per
period, so they can differ slightly between periods).

### `shadow.json`
```
periods: { <p>: { limits: { truck|space|budget: { store_days, binding, value_max, use_mean, use_max } },
                  layers: { floor|spoil: { rows, mean, max, median, store_items } },
                  floor_matrix: { stores, dates, cells: [store][day] = [count, mean price] },
                  top_floor_items: [ {store, item, days} ], ilp_term_check: {matched, total, what}, source } }
whatif: { rows: [ { param, value, fill, lost, waste, stock, total, net_cost, penalty,
                    shadow: { truck|space|budget: [mean shadow price, binding store-days] } } ], source, note }
```
Source: `<run>/store_ilp/shadow_prices.csv`, `outputs/replay/whatif_store_ilp_2026-11-02/whatif.csv`.

### `forecast.json`
```
folds: { fold1_festive|fold2_main: { model|naive|stub: { mae, pinball_q50..q95, cover_q50..q95, n },
                                     model_perishable: {...} } }, fold_info,
holdout: { b1|b2: { days, rows, mae, cover_q80, cover_q90, cover_q95, source } },
test: { all|perishable|long_life: { n, mae, cover_q50..q95 }, note },   // computed by the export
calibration: { shifts, tolerance_pts }, sources
```
Sources: `outputs/forecast/scores/{model_val_calibrated,baseline_val,stub_val}.json`,
`data/processed_holdout_b{1,2}/forecast_scores.csv`. The test-window scores are computed by the
export (same-day forecast vs the replay's observed demand, 25 Dec left out).

## Notes and limits
- The cost curve on the Day decision page has two lines. The orange one is exact: this item's own
  term of the ILP objective. The blue one is an illustration: an expected cost over a demand curve
  drawn through the four forecast lines (linear between them, extrapolated past q95). The real ILP
  solves all items of a store together with shared limits; none of those limits bound in these runs.
- "Other methods ordered" on the Outcome step comes from each method's own replay, where its stock
  that morning was different.
- `screenshots/` holds captures made while testing (headless Chrome for full pages, the in-app
  browser for the dialog steps).
- Motion respects `prefers-reduced-motion`; the theme toggle is stored in `localStorage` under
  `retailops-theme` (wrapped in try/catch; without storage it follows the system setting).
