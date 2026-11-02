# RetailOps: stock ordering for a 10-store chain on real Walmart sales

CS2105 Optimization Techniques course project, IIT Patna.

A grocery chain has to decide every morning how many cartons of each product to send to each
store. If it sends too few, shelves go empty and sales are lost. If it sends too much, money
sits in stock and perishables spoil. RetailOps makes that decision for 10 stores × 120 food
products using real daily sales from the Walmart M5 dataset (Kaggle), and tests it day by day
against what really sold.

## The two models
1. **ML forecaster** (`forecast/`): one quantile gradient-boosting model per line
   (q50, q80, q90, q95) predicts today's and tomorrow's sales for each store and product. Every
   store input is at least 2 days old, so the model never sees the day it predicts.
2. **Order model** (`analysis/store_ilp.py`, `replay/store_model.py`), the `store_ilp` method:
   - newsvendor targets: long-life items are stocked to q95, perishables to the cost-based
     level Cu/(Cu+Co) ≈ 0.63, with a spoilage cap;
   - one integer linear program (ILP) per store per day turns them into whole cartons under
     truck, shelf-space and budget limits, with a soft 90% service floor and the spoilage cap
     as constraints;
   - the integers are fixed and the problem re-solved as an LP to read **shadow prices** (what
     one more unit of each limit is worth).

   `store_model` (targets only, no ILP) and three simple rules (repeat last week, top up to a
   4-week average, top up to our forecast) are kept as baselines. The warehouse level is in the
   code but switched off (`replay.warehouse: off`).

## Results (net cost + lost-sale penalty; lower is better)
| Period | store_ilp vs best simple rule | Fill rate | Waste (units) |
|---|---|---|---|
| Test window, 1 Nov to 26 Dec 2015 (core ILP) | −7.7% | 94.87% | 8,043 |
| Holdout block 1, 27 Dec 2015 to 13 Feb 2016 | −5.2% | 94.14% | 4,469 |
| Holdout block 2, 14 Feb to 2 Apr 2016 | −7.1% | 95.40% | 5,465 |

The holdout blocks were run once, with the code and settings frozen beforehand
(`holdout/RUN_LIST_block*.md`). Block 3 (3 Apr to 22 May 2016) is still sealed.

## Notebooks (start here)
- `notebooks/01_forecaster.ipynb`: data, the 2-day rule and leakage checks, validation
  against the baselines, then writes `outputs/notebooks/forecasts.parquet` with the frozen model
  and checks it is identical to `data/processed/forecasts.parquet`.
- `notebooks/02_order_model.ipynb`: loads that file, explains the newsvendor targets, the ILP
  and shadow prices, solves one store-day, runs a short 7-day replay, and charts the saved
  results (test window, holdout blocks 1 and 2, what-ifs).

The notebooks call the code in the packages below; they hold no copy of it.

## Folder layout
| Folder | What |
|---|---|
| `forecast/` | forecaster: features, model, calibration, freeze, predict |
| `analysis/` | store ILP, shadow prices, what-ifs |
| `replay/` | day-by-day replay, baselines, scorecard (KPIs), `store_model` |
| `optimize/` | day runner and independent audit used by the replay (plus the original warehouse order model) |
| `data_prep/`, `stubs/` | builds `data/processed/` from the raw M5 files |
| `holdout/` | holdout block runner and the run lists written before each run |
| `config.yaml`, `configs/` | every setting and assumed number |
| `data/` | processed sales, products, forecasts, start stock, test fixtures |
| `outputs/` | frozen forecaster, scores and charts, replay results |
| `docs/` | decisions log, design principles, study guides |
| `tests/` | 299 tests |

## Running it
Python 3.14 with the packages in `requirements.txt` (plus `jupyter` for the notebooks).

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m replay.cli --profile full --config configs/validation_fast.yaml --methods store_model,store_ilp --end-date 2015-11-07 --run-name my_short_run
```

The last command is a 7-day replay with the 15-second solver limit. Leave out `--end-date` for
the full 8 weeks, which takes roughly 6 to 12 minutes.

The processed data is included, so the notebooks and tests run without the raw M5 files. To
rebuild `data/processed/` from scratch, download the M5 CSVs into `data/m5/` (see its README) and
run `python -m data_prep.prepare --config config.yaml`.
