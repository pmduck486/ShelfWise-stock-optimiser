# data_prep: M5 sales data, cleaned for RetailOps

Turns Walmart's public M5 sales data into **one clean table with one row per store, product and day**. The forecast and replay steps read this table.

## What it does
0. Checks the time windows in `config.yaml` (`data.windows`): in order, no overlap, inside the M5 range (2011-01-29 to 2016-05-22), and `history_start` not after `warmup_start`. A bad window stops the run with a clear error.
1. Reads `calendar.csv` and the wide `sales_train_evaluation.csv` (one column per day), keeping days from `history_start` (2012-11-26) to the day before `holdout_start`. **The holdout (2015-12-27 to 2016-05-22) is sealed:** it is not read at all unless you pass `--include-holdout` (see below).
2. Keeps FOODS items in the configured stores and picks the product set. **Rule R5:** the choice uses only sales from `history_start` to the day before `test_start` (2015-10-31), so it can't favour items that sold well in the test or holdout weeks.
   - `main`: all 10 stores, the top 120 FOODS items by total units before `test_start`.
   - `random`: the seed picks 3 stores and 40 items, sampled only from items that sell at least `min_avg_daily_units` per store per day. Layer R1 uses these sets.
3. Flags perishables and gives each one a shelf life (see Assumptions).
4. Melts the table to long form, then adds the weekly price (from `sell_prices.csv`), the calendar event and the SNAP flag for the store's state.
5. Drops days before an item-store's first price (not on sale yet) and forward-fills any later price gap.

## How to run
From the `retailops/` folder:
```
pip install -r requirements.txt
python -m data_prep.prepare --config config.yaml                              # main set
python -m data_prep.prepare --config config.yaml --product-set random --seed 3   # sales_random_seed3.parquet
pytest -q tests/test_data_prep.py
```
**Final holdout run only** (after code freeze, once; forecast-training-plan.md R9b):
```
python -m data_prep.prepare --config config.yaml --include-holdout   # writes sales_full.parquet
```
This writes the full series to a separate `sales_full.parquet`; `sales.parquet` is left untouched. The product list and perishable flags are identical in both files, because they only use pre-test sales.
All settings are in the `data:` section of `config.yaml`. The M5 CSVs go in `data/m5/`.

## Outputs (in `data/processed/`, not committed to git)
| File | Contents |
|---|---|
| `sales.parquet` | one row per store, item and day (main set), 2012-11-26 to 2015-12-26 (no holdout rows) |
| `sales_full.parquet` | the same, plus the holdout to 2016-05-22. Only made with `--include-holdout` |
| `items.csv` | one row per item (main set) |
| `sales_random_seed<k>.parquet`, `items_random_seed<k>.csv` | the same files for random set k (`sales_full_random_seed<k>.parquet` with `--include-holdout`) |

**sales columns:** `date`, `store_id`, `item_id`, `dept_id`, `units` (int, ≥ 0), `price` (that week's price), `is_perishable`, `shelf_life_days` (empty if not perishable), `wday` (1 = Saturday, as in M5), `month`, `event_name` (empty if none), `is_event`, `snap` (a food-benefit day in the store's state).

**items columns:** `item_id`, `dept_id`, `rank` (1 = biggest seller), `avg_daily_units` (units per store per day from `history_start` to the day before `test_start`; the same pre-test numbers used to choose the items), `is_perishable`, `shelf_life_days`.

From Python:
```python
from data_prep.prepare import load_config, build_sales_table
sales, items = build_sales_table(load_config("config.yaml"))
```

## Assumptions (all set in config.yaml)
- **Perishable stand-in.** M5 products are anonymous, with no names or "fresh" label. We flag the 20 fastest-selling selected items of department FOODS_3 as perishable (7 in a random set). This stands in for fresh, fast-moving goods such as milk or bread.
- **Shelf lives.** Perishables get shelf lives in rank order, cycling through 2, 3, 4, 5 and 7 days. These are invented numbers. Non-perishables have none.
- **Average daily sales** count every day of the pre-test window, including days before a product launched. A product that launched late therefore looks a little slower than it is.
- **Zero sales are kept as real zeros.** Many zero days are probably empty shelves rather than no demand. The forecast step (layer P7) deals with this.
