# stubs: placeholder inputs for the order model

**These are dummy files, not the real forecaster.** They let Lump 2 (order ILP) build and test the model before Lump 1's quantile gradient-boosting forecaster is ready. The real `forecasts.parquet` will have exactly the same columns, so swapping it in needs no code change.

## How to run
From the `retailops/` folder (needs `data/processed/sales.parquet` and `items.csv` from data_prep):
```
python -m stubs.make_stubs --config config.yaml
pytest -q tests/test_stubs.py
```
All numbers come from the `stubs:` and `network:` sections of `config.yaml`.

## The stub forecast rule
For target day D (every input is at least 7 days older than D, rule R2):
- `q50` = mean of units on D-7, D-14, D-21 and D-28 (the same weekday over the last 4 weeks);
- `sd` = standard deviation of daily units over D-34 .. D-7 (28 days);
- `q80`, `q90`, `q95` = q50 + z * sd, with z = 0.84, 1.28, 1.645; clipped at 0, lines never cross.

Days with no row (the item was not on sale yet) are skipped, not counted as zeros. The forecast for a day depends only on that day, so the same date gets the same numbers from every origin. Because every input is 7+ days old, a forecast made on origin o never sees o or later.

## Outputs (in `data/processed/`)
| File | Contents |
|---|---|
| `forecasts_stub.parquet` | origin, date, store_id, item_id, q50, q80, q90, q95. One origin per test day (2015-11-01 to 2015-12-26, 56 origins), each covering origin .. origin+6, all 10 stores x 120 items |
| `forecasts_stub_small.parquet` | Gate 1 case: CA_1 and TX_1 x the top 10 items, origins = the first 7 test days |
| `state_start_store_stock.csv` | date, store_id, item_id, units = round(3 x mean daily units over the 28 days before test_start) |
| `state_start_dc_stock.csv` | date, dc_id, item_id, units = round(7 x sum of the home stores' 28-day means) |
| `state_start_pipeline.csv` | order_date, arrival_date, location_type (store or dc), location_id, item_id, units. Empty at the start |
| `*_small.csv` | the same three state files for the small case |

These map onto the `State` dataclass (`date`, `store_stock`, `dc_stock`, `pipeline`) in plan/build-roadmap.md.

## Assumptions (all in config.yaml, Lump 2 to confirm)
- Two warehouses: DC_WEST serves CA_1 to CA_4; DC_EAST serves TX_1 to TX_3 and WI_1 to WI_3.
- Stores start with 3 days of stock, DCs with 7 days; nothing is on the way.

## Things to know
- The last origins (2015-12-20 to 12-26) forecast up to 2016-01-01, past the test window. That's fine: they only read sales up to 2015-12-25.
- Christmas Day (2015-12-25, stores closed, 10 units in total) is a lag for 2016-01-01, so that day's forecast is pulled down. The day before Thanksgiving (2015-11-25, about 12,400 units vs about 9,000 on a normal Wednesday) likewise pushes up the stub for 2015-12-02, 12-09, 12-16 and 12-23, but it isn't anticipated on 11-25 itself. The stub knows no calendar; the real forecaster will.
