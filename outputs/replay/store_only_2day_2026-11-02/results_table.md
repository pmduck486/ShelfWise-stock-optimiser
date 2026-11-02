# Replay results: store_only_2day_2026-11-02

full profile, 2015-11-01 to 2015-12-26; config `configs\validation_fast.yaml` (solver limit 15 s); forecasts `data\processed\forecasts.parquet`.

| KPI | last_week | topup_avg | topup_forecast | store_model |
|---|---|---|---|---|
| fill_rate | 0.8406 | 0.9524 | 0.9792 | 0.9407 |
| stockout_days | 10690 | 3070 | 1401 | 2895 |
| lost_units | 81,425.77 | 24,329.65 | 10,608.25 | 30,282.23 |
| waste_units | 19,907.77 | 67,996.72 | 73,861.33 | 6,841.59 |
| avg_stock_value | 18,062.34 | 28,322.80 | 33,844.75 | 35,240.10 |
| avg_store_stock_value | 18,062.34 | 28,322.80 | 33,844.75 | 35,240.10 |
| avg_warehouse_stock_value | - | - | - | - |
| days_of_stock | 1.2848 | 1.9728 | 2.4315 | 2.6191 |
| turnover | 260.75 | 187.84 | 160.77 | 148.02 |
| total_cost | 773,484.06 | 957,737.57 | 989,845.58 | 852,869.69 |
| net_cost | 791,541.49 | 965,598.51 | 986,549.21 | 851,080.92 |
| lost_sales_penalty | 223,864.89 | 58,573.10 | 25,819.17 | 86,751.22 |
| orders | 25905 | 32359 | 32798 | 30040 |
| trucks | 550 | 550 | 550 | 550 |
| supplier_order_lines | 0 | 0 | 0 | 0 |
| supplier_units | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| service_met_share | 0.3342 | 0.8725 | 0.9708 | 0.8617 |
| service_mean_no_loss | 0.8380 | 0.9535 | 0.9788 | 0.9561 |
| purchase_cash | 736,252.93 | 913,230.12 | 944,813.56 | 813,906.71 |
| truck_cash | 22,000.00 | 22,000.00 | 22,000.00 | 22,000.00 |
| order_cash | 12,952.50 | 16,179.50 | 16,399.00 | 15,020.00 |
| holding_cash | 692.80 | 1,086.35 | 1,298.15 | 1,351.68 |
| disposal_cash | 1,585.82 | 5,241.59 | 5,334.87 | 591.31 |
| budget_scaled_days | 0 | 0 | 0 | 0 |
| dc_short_lines | 0 | 0 | 0 | 0 |
| n_vars | - | - | - | - |
| n_int | - | - | - | - |
| solve_seconds_total | - | - | - | - |
| fallback_days | - | - | - | - |
| wall_seconds | 103.28 | 99.90 | 104.78 | 109.70 |

Service measures leave out closed store days (25 Dec); costs keep every day. Definitions and denominators: kpis.json.
