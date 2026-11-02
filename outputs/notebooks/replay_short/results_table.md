# Replay results: replay_short

full profile, 2015-11-01 to 2015-11-07; config `configs\validation_fast.yaml` (solver limit 15 s); forecasts `outputs\notebooks\forecasts.parquet`.

| KPI | store_model | store_ilp |
|---|---|---|
| fill_rate | 0.9634 | 0.9688 |
| stockout_days | 269 | 221 |
| lost_units | 2,447.20 | 2,088.33 |
| waste_units | 1,673.11 | 1,824.58 |
| avg_stock_value | 34,539.29 | 34,975.64 |
| avg_store_stock_value | 34,539.29 | 34,975.64 |
| avg_warehouse_stock_value | - | - |
| days_of_stock | 2.5155 | 2.5364 |
| turnover | 163.67 | 162.69 |
| total_cost | 119,312.82 | 120,946.09 |
| net_cost | 116,422.35 | 117,440.58 |
| lost_sales_penalty | 6,526.74 | 5,272.60 |
| orders | 4398 | 4385 |
| trucks | 70 | 70 |
| supplier_order_lines | 0 | 0 |
| supplier_units | 0.0000 | 0.0000 |
| service_met_share | 0.8600 | 0.8842 |
| service_mean_no_loss | 0.9680 | 0.9737 |
| purchase_cash | 114,012.95 | 115,635.77 |
| truck_cash | 2,800.00 | 2,800.00 |
| order_cash | 2,199.00 | 2,192.50 |
| holding_cash | 165.60 | 167.69 |
| disposal_cash | 135.28 | 150.13 |
| budget_scaled_days | 0 | 0 |
| dc_short_lines | 0 | 0 |
| n_vars | - | 6,010.00 |
| n_int | - | 2,410.00 |
| solve_seconds_total | - | 24.70 |
| fallback_days | - | - |
| wall_seconds | 8.4767 | 34.24 |

Service measures leave out closed store days (25 Dec); costs keep every day. Definitions and denominators: kpis.json.
