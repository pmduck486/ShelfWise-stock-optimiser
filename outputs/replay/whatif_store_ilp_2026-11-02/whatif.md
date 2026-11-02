# What-ifs: whatif_store_ilp_2026-11-02

Method store_ilp, store-only, 2015-11-01 to the end of the test window. One setting changed per row; the first row of each setting is the base.

| param | value | fill_rate | lost_units | waste_units | avg_stock_value | net_cost_plus_penalty | shadow_budget_mean | shadow_space_mean | shadow_truck_mean |
|---|---|---|---|---|---|---|---|---|---|
| order_model.truck_capacity_cases | 400 | 0.9487 | 26,208 | 8,043 | 35,279 | 934,088 | 0.0000 | 0.0000 | 0.0000 |
| order_model.truck_capacity_cases | 100 | 0.9482 | 26,463 | 8,038 | 35,161 | 937,121 | 0.0000 | 0.0000 | 0.0000 |
| order_model.truck_capacity_cases | 60 | 0.9475 | 26,830 | 7,971 | 34,938 | 947,292 | 0.0000 | 0.0000 | 0.0065 |
| analysis.store_ilp.space_factor | 1.0000 | 0.9487 | 26,208 | 8,043 | 35,279 | 934,088 | 0.0000 | 0.0000 | 0.0000 |
| analysis.store_ilp.space_factor | 0.5000 | 0.9477 | 26,706 | 8,042 | 35,221 | 934,200 | 0.0000 | 0.0011 | 0.0000 |
| analysis.store_ilp.space_factor | 0.3000 | 0.8605 | 71,222 | 5,916 | 30,309 | 947,310 | 0.0000 | 0.9501 | 0.0000 |
| analysis.store_ilp.budget_factor | 1.2500 | 0.9487 | 26,208 | 8,043 | 35,279 | 934,088 | 0.0000 | 0.0000 | 0.0000 |
| analysis.store_ilp.budget_factor | 0.6000 | 0.9472 | 26,967 | 8,036 | 34,982 | 934,916 | 0.0385 | 0.0000 | 0.0000 |
| analysis.store_ilp.budget_factor | 0.4500 | 0.9303 | 35,595 | 6,881 | 32,000 | 942,990 | 0.5005 | 0.0000 | 0.0000 |
| order_model.lost_sale_penalty_share | 1.5000 | 0.9487 | 26,208 | 8,043 | 35,279 | 934,088 | 0.0000 | 0.0000 | 0.0000 |
| order_model.lost_sale_penalty_share | 1.0000 | 0.9234 | 39,099 | 5,498 | 34,418 | 908,493 | 0.0000 | 0.0000 | 0.0000 |
| order_model.lost_sale_penalty_share | 2.0000 | 0.9599 | 20,458 | 10,112 | 35,966 | 949,810 | 0.0000 | 0.0000 | 0.0000 |
