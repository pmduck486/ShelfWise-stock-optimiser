# Replay results: store_ilp_2026-11-02

full profile, 2015-11-01 to 2015-12-26; config `configs\validation_fast.yaml` (solver limit 15 s); forecasts `data\processed\forecasts.parquet`.

| KPI | store_model | store_ilp |
|---|---|---|
| fill_rate | 0.9407 | 0.9487 |
| stockout_days | 2895 | 2520 |
| lost_units | 30,282.23 | 26,208.44 |
| waste_units | 6,841.59 | 8,042.92 |
| avg_stock_value | 35,240.10 | 35,279.24 |
| avg_store_stock_value | 35,240.10 | 35,279.24 |
| avg_warehouse_stock_value | - | - |
| days_of_stock | 2.6191 | 2.6093 |
| turnover | 148.02 | 149.21 |
| total_cost | 852,869.69 | 862,342.40 |
| net_cost | 851,080.92 | 860,281.63 |
| lost_sales_penalty | 86,751.22 | 73,806.09 |
| orders | 30040 | 29971 |
| trucks | 550 | 550 |
| supplier_order_lines | 0 | 0 |
| supplier_units | 0.0000 | 0.0000 |
| service_met_share | 0.8617 | 0.8608 |
| service_mean_no_loss | 0.9561 | 0.9618 |
| purchase_cash | 813,906.71 | 823,322.02 |
| truck_cash | 22,000.00 | 22,000.00 |
| order_cash | 15,020.00 | 14,985.50 |
| holding_cash | 1,351.68 | 1,353.18 |
| disposal_cash | 591.31 | 681.70 |
| budget_scaled_days | 0 | 0 |
| dc_short_lines | 0 | 0 |
| n_vars | - | 4,810.00 |
| n_int | - | 2,410.00 |
| solve_seconds_total | - | 136.13 |
| fallback_days | - | - |
| wall_seconds | 114.94 | 250.90 |

Service measures leave out closed store days (25 Dec); costs keep every day. Definitions and denominators: kpis.json.
