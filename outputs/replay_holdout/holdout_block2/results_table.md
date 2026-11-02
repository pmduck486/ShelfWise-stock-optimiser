# Replay results: holdout_block2

full profile, 2016-02-14 to 2016-04-02; config `configs\validation_fast_holdout_b2.yaml` (solver limit 15 s); forecasts `data\processed_holdout_b2\forecasts.parquet`.

| KPI | last_week | topup_avg | topup_forecast | store_model | store_ilp |
|---|---|---|---|---|---|
| fill_rate | 0.8556 | 0.9570 | 0.9816 | 0.9448 | 0.9540 |
| stockout_days | 10293 | 3188 | 1425 | 3252 | 2737 |
| lost_units | 75,139.19 | 22,373.25 | 9,584.14 | 28,718.84 | 23,949.58 |
| waste_units | 15,560.12 | 43,958.42 | 57,489.34 | 6,170.64 | 5,464.98 |
| avg_stock_value | 14,921.29 | 25,878.68 | 32,429.11 | 33,864.67 | 34,285.50 |
| avg_store_stock_value | 14,921.29 | 25,878.68 | 32,429.11 | 33,864.67 | 34,285.50 |
| avg_warehouse_stock_value | - | - | - | - | - |
| days_of_stock | 1.0058 | 1.6990 | 2.1203 | 2.2850 | 2.2989 |
| turnover | 351.33 | 225.40 | 183.89 | 168.78 | 168.57 |
| total_cost | 753,814.82 | 897,949.04 | 942,786.58 | 821,399.27 | 830,292.66 |
| net_cost | 765,184.74 | 905,191.67 | 933,125.96 | 813,789.11 | 821,940.76 |
| lost_sales_penalty | 192,391.50 | 52,418.30 | 21,582.51 | 80,264.56 | 65,146.49 |
| orders | 26253 | 32223 | 32380 | 30383 | 30247 |
| trucks | 490 | 490 | 490 | 490 | 490 |
| supplier_order_lines | 0 | 0 | 0 | 0 | 0 |
| supplier_units | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| service_met_share | 0.2292 | 0.8058 | 0.9575 | 0.8275 | 0.8275 |
| service_mean_no_loss | 0.8249 | 0.9458 | 0.9758 | 0.9447 | 0.9535 |
| purchase_cash | 719,244.94 | 857,296.13 | 901,351.41 | 784,968.39 | 793,932.86 |
| truck_cash | 19,600.00 | 19,600.00 | 19,600.00 | 19,600.00 | 19,600.00 |
| order_cash | 13,126.50 | 16,111.50 | 16,190.00 | 15,191.50 | 15,123.50 |
| holding_cash | 500.78 | 868.53 | 1,088.37 | 1,136.55 | 1,150.68 |
| disposal_cash | 1,342.59 | 4,072.88 | 4,556.80 | 502.83 | 485.62 |
| budget_scaled_days | 0 | 0 | 0 | 0 | 0 |
| dc_short_lines | 0 | 0 | 0 | 0 | 0 |
| n_vars | - | - | - | - | 6,010.00 |
| n_int | - | - | - | - | 2,410.00 |
| solve_seconds_total | - | - | - | - | 249.57 |
| fallback_days | - | - | - | - | - |
| wall_seconds | 123.34 | 129.62 | 214.75 | 219.65 | 482.32 |

Service measures leave out closed store days (25 Dec); costs keep every day. Definitions and denominators: kpis.json.
