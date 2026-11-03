"""Historical Holiday Calibration Script (Phase 1).
Calibrates item-specific surge quantiles (tau*) and empirical post-holiday slump ratios (D_post)
strictly on 2013 and 2014 Walmart M5 sales history (zero leak into the 2015 test window).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def calibrate():
    sales_path = ROOT / "data" / "processed" / "sales.parquet"
    items_path = ROOT / "data" / "processed" / "items.csv"
    out_path = ROOT / "outputs" / "holiday_policy.json"

    sales = pd.read_parquet(sales_path)
    items_df = pd.read_csv(items_path)
    items_dict = items_df.set_index("item_id").to_dict(orient="index")

    # Filter strictly to training history (2013-01-01 to 2014-12-31)
    train_sales = sales[(sales["date"] >= "2013-01-01") & (sales["date"] <= "2014-12-31")].copy()

    # Define holiday windows for Thanksgiving (2013-11-28 and 2014-11-27)
    # Peak surge days (Eve-1 and Eve): 2 days preceding Thanksgiving
    surge_dates = pd.to_datetime(["2013-11-26", "2013-11-27", "2014-11-25", "2014-11-26"])
    # Slump days (Thanksgiving and Black Friday weekend)
    slump_dates = pd.to_datetime(["2013-11-28", "2013-11-29", "2014-11-27", "2014-11-28"])
    # Baseline comparison: non-holiday November days (Nov 5 to Nov 20 in 2013 and 2014)
    normal_dates = pd.to_datetime([f"2013-11-{d:02d}" for d in range(5, 21)] + [f"2014-11-{d:02d}" for d in range(5, 21)])

    # Compute mean sales per item across stores for each window
    surge_sales = train_sales[train_sales["date"].isin(surge_dates)].groupby("item_id")["units"].mean()
    slump_sales = train_sales[train_sales["date"].isin(slump_dates)].groupby("item_id")["units"].mean()
    normal_sales = train_sales[train_sales["date"].isin(normal_dates)].groupby("item_id")["units"].mean()

    policy = {}
    sensitive_count = 0

    for item_id in items_df["item_id"]:
        norm = float(normal_sales.get(item_id, 1.0))
        surge = float(surge_sales.get(item_id, norm))
        slump = float(slump_sales.get(item_id, norm))

        norm = max(norm, 0.5)
        lift = surge / norm
        d_post = slump / norm

        # Determine optimal quantile line tau* and M_surge
        if lift >= 1.60:
            tau = "q95"
            m_surge = round(min(lift, 2.75), 2)
            sensitive = True
        elif lift >= 1.30:
            tau = "q90"
            m_surge = round(min(lift, 1.85), 2)
            sensitive = True
        elif lift >= 1.15:
            tau = "q80"
            m_surge = round(min(lift, 1.35), 2)
            sensitive = True
        else:
            tau = "q50"
            m_surge = 1.0
            sensitive = False

        if sensitive:
            sensitive_count += 1

        meta = items_dict.get(item_id, {})
        is_perish = bool(meta.get("is_perishable", False))

        # D_post dampener logic:
        # Perishables undergo strict spoilage throttle (0.20x to 0.50x) with spoilage warning to flush surge pipeline
        # Non-perishables have D_post completely removed (1.00x) as dry goods do not spoil
        if is_perish:
            d_post_bounded = round(float(np.clip(d_post, 0.20, 0.50)), 2)
            spoilage_warning = True
            throttle_type = "spoilage_prevention"
        else:
            d_post_bounded = 1.00
            spoilage_warning = False
            throttle_type = "unthrottled"

        policy[item_id] = {
            "lift": round(lift, 2),
            "tau_star": tau,
            "m_surge": m_surge,
            "d_post": d_post_bounded,
            "is_perishable": is_perish,
            "shelf_life_days": meta.get("shelf_life_days", None),
            "sensitive": sensitive,
            "spoilage_warning": spoilage_warning,
            "throttle_type": throttle_type,
        }

    p_dposts = [p["d_post"] for p in policy.values() if p["is_perishable"]]
    np_dposts = [p["d_post"] for p in policy.values() if not p["is_perishable"]]

    out_data = {
        "calibrated_from": "2013-2014 Thanksgiving & November history",
        "sensitive_item_count": sensitive_count,
        "total_items": len(items_df),
        "mean_surge_multiplier": round(float(np.mean([p["m_surge"] for p in policy.values() if p["sensitive"]])), 2),
        "mean_slump_dampener": round(float(np.mean([p["d_post"] for p in policy.values()])), 2),
        "mean_slump_dampener_perishables": round(float(np.mean(p_dposts)), 2),
        "mean_slump_dampener_non_perishables": round(float(np.mean(np_dposts)), 2),
        "items": policy,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"Holiday calibration completed! Written to {out_path}")
    print(f"Sensitive items identified: {sensitive_count}/{len(items_df)} (Mean surge: {out_data['mean_surge_multiplier']}x, Mean slump dampener: {out_data['mean_slump_dampener']}x)")

if __name__ == "__main__":
    calibrate()
