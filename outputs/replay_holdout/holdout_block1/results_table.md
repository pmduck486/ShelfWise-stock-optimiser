# Replay results: holdout_block1

full profile, 2015-12-27 to 2016-02-13; config `configs\validation_fast_holdout_b1.yaml` (solver limit 15 s); forecasts `data\processed_holdout_b1\forecasts.parquet`.

| KPI | last_week | topup_avg | topup_forecast | store_model | store_ilp |
|---|---|---|---|---|---|
| fill_rate | 0.8244 | 0.9411 | 0.9780 | 0.9337 | 0.9414 |
| stockout_days | 10859 | 3568 | 1541 | 3221 | 2777 |
| lost_units | 84,345.22 | 28,279.30 | 10,571.64 | 31,826.83 | 28,140.35 |
| waste_units | 13,671.52 | 35,578.63 | 70,809.05 | 5,352.38 | 4,468.86 |
| avg_stock_value | 14,597.26 | 25,416.81 | 31,374.33 | 32,478.31 | 32,916.32 |
| avg_store_stock_value | 14,597.26 | 25,416.81 | 31,374.33 | 32,478.31 | 32,916.32 |
| avg_warehouse_stock_value | - | - | - | - | - |
| days_of_stock | 1.0403 | 1.7365 | 2.2084 | 2.3857 | 2.3992 |
| turnover | 337.05 | 218.72 | 182.57 | 168.55 | 167.95 |
| total_cost | 705,454.96 | 835,693.31 | 918,867.90 | 790,985.64 | 798,561.51 |
| net_cost | 713,039.96 | 838,808.76 | 902,939.35 | 777,767.94 | 784,531.17 |
| lost_sales_penalty | 214,667.46 | 63,232.85 | 23,202.17 | 83,358.50 | 70,563.34 |
| orders | 24039 | 29869 | 30746 | 28481 | 28374 |
| trucks | 490 | 490 | 490 | 490 | 490 |
| supplier_order_lines | 0 | 0 | 0 | 0 | 0 |
| supplier_units | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| service_met_share | 0.2350 | 0.7892 | 0.9467 | 0.8292 | 0.8325 |
| service_mean_no_loss | 0.8153 | 0.9393 | 0.9738 | 0.9452 | 0.9528 |
| purchase_cash | 672,372.17 | 797,585.84 | 878,178.74 | 755,677.05 | 763,313.07 |
| truck_cash | 19,600.00 | 19,600.00 | 19,600.00 | 19,600.00 | 19,600.00 |
| order_cash | 12,019.50 | 14,934.50 | 15,373.00 | 14,240.50 | 14,187.00 |
| holding_cash | 489.91 | 853.03 | 1,052.97 | 1,090.03 | 1,104.73 |
| disposal_cash | 973.38 | 2,719.94 | 4,663.18 | 378.07 | 356.71 |
| budget_scaled_days | 0 | 0 | 0 | 0 | 0 |
| dc_short_lines | 0 | 0 | 0 | 0 | 0 |
| n_vars | - | - | - | - | 6,010.00 |
| n_int | - | - | - | - | 2,410.00 |
| solve_seconds_total | - | - | - | - | 175.61 |
| fallback_days | - | - | - | - | - |
| wall_seconds | 91.05 | 89.82 | 94.65 | 93.47 | 336.50 |

Service measures leave out closed store days (25 Dec); costs keep every day. Definitions and denominators: kpis.json.
