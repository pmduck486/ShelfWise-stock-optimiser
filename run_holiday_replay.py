"""Phase 3 Replay: Evaluate store_ilp with Holiday Policy Layer active.
Runs the 56-day test window (10 stores, 120 items) on CBC solver.
"""

import json
import time
from copy import deepcopy
from pathlib import Path
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent

from optimize.parameters import load_case
from replay.cli import ROOT, _path, load_sales
from replay.baselines import make_policy
from replay.engine import run

def main():
    t0 = time.time()
    main_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    rp, an = main_cfg["replay"], main_cfg["analysis"]

    case = load_case(
        _path(rp["input_dir"]),
        _path(ROOT / "configs" / "validation_fast.yaml"),
        profile="full",
        forecast_path=_path(ROOT / "data" / "processed" / "forecasts.parquet"),
    )
    cfg = deepcopy(case.parameters.config)
    cfg["replay"] = rp
    cfg["analysis"] = an
    # Enable holiday policy layer
    cfg["analysis"]["store_ilp"]["layers"]["holiday_policy"]["enabled"] = True

    out_dir = _path(rp["output_dir"]) / "store_ilp_holiday"
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sales = load_sales(rp["history_sales_file"], cfg["data"]["windows"]["test_end"])
    forecasts = pd.read_parquet(_path(ROOT / "data" / "processed" / "forecasts.parquet"))
    policy = make_policy("store_ilp")

    print("Starting 56-day Replay with Holiday Policy Layer ON...", flush=True)
    res, k = run(
        policy,
        sales,
        case.state.date,
        rp["weeks"],
        cfg,
        state=deepcopy(case.state),
        forecasts=forecasts,
        output_dir=out_dir / "store_ilp",
        method="store_ilp [holiday_policy]",
        run_log=_path(rp["test_runs_log"]),
    )

    elapsed = time.time() - t0
    total_cost = k["net_cost"] + k["lost_sales_penalty"]
    print(f"\nReplay finished in {elapsed/60:.2f} min!", flush=True)
    print(f"Fill Rate: {k['fill_rate']*100:.2f}%")
    print(f"Lost Units: {k['lost_units']:,.0f}")
    print(f"Waste Units: {k['waste_units']:,.0f}")
    print(f"Store Stock Value: ${k['avg_stock_value']:,.0f}")
    print(f"Net Cost + Penalty: ${total_cost:,.0f}")

    # Inspect Thanksgiving week specifically (Nov 23 to Nov 28)
    daily_csv = out_dir / "store_ilp" / "daily_metrics.csv"
    if daily_csv.exists():
        df_d = pd.read_csv(daily_csv)
        df_tg = df_d[(df_d["date"] >= "2015-11-23") & (df_d["date"] <= "2015-11-28")]
        tg_sales = df_tg["served_units"].sum()
        tg_lost = df_tg["lost_units"].sum()
        tg_fill = tg_sales / (tg_sales + tg_lost) if (tg_sales + tg_lost) > 0 else 1.0
        tg_waste = df_tg["waste_proxy_units"].sum()
        print(f"\n--- Thanksgiving Week (Nov 23-28) ---", flush=True)
        print(f"Thanksgiving Week Fill Rate: {tg_fill*100:.2f}% (Served: {tg_sales:,.0f}, Lost: {tg_lost:,.0f})")
        print(f"Thanksgiving Week Waste Units: {tg_waste:,.0f}")

    kpi_file = out_dir / "kpi_summary.json"
    kpi_data = {
        "method": "store_ilp + holiday_policy",
        "fill": round(k["fill_rate"] * 100, 2),
        "lost": int(round(k["lost_units"])),
        "waste": int(round(k["waste_units"])),
        "stock": int(round(k["avg_stock_value"])),
        "total": int(round(total_cost)),
        "elapsed_s": round(elapsed, 1),
    }
    kpi_file.write_text(json.dumps(kpi_data, indent=2), encoding="utf-8")
    print(f"Saved KPIs to {kpi_file}")

if __name__ == "__main__":
    main()
