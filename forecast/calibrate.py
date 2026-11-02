"""Calibration: the honesty check and the scaled correction (Task 7).

  - fit_shifts: on validation rows (both folds pooled, closed days left out), one number per
    line. Upper lines (q80, q90, q95) get a factor k: if a line's coverage is off its target
    by more than forecast.calibration_tolerance_pts, k is found by bisection in [0.5, 4] so
    that q50 + k * (line - q50) covers the target share; otherwise k = 1. The correction grows
    with each product's own spread, so small and big sellers both end near the target.
    q50 has no spread to scale, so it gets a unit shift (the median of actual - q50), again
    only if its coverage is off by more than the tolerance; otherwise 0.
  - apply_shifts: scales the upper lines around the original q50, then shifts q50, then
    post-processes (sort, clip at 0, 25 Dec = 0), so a stretched line never crosses another.
  - cross_fold_check (re-check strategy 2, approved 2026-10-31): numbers fitted on one fold,
    judged on the other, so the gate's coverage is not marked on the days it was fitted on.

Run it from the retailops folder (after Task 6 wrote the tuned settings):
    python -m forecast.calibrate --config config.yaml
It fits each fold's model on its own training rows, pools both folds' validation rows, fits
the numbers and writes outputs/forecast/scores/model_val_calibrated.json: coverage before and
after per fold (all / perishable / slow, by size quarter, peak days), the cross-fold check, and
the model-vs-stub group check (re-check strategy 8).
"""

import numpy as np
import pandas as pd

from forecast.metrics import QUANTILES, closed_mask
from forecast.postprocess import postprocess

KEY = ["date", "store_id", "item_id"]
UPPER = ["q80", "q90", "q95"]
K_RANGE = (0.5, 4.0)


def _open_rows(lines: pd.DataFrame, actual: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = lines.merge(actual[KEY + ["units"]], on=KEY, how="inner", validate="one_to_one")
    return d[~closed_mask(d["date"], cfg).to_numpy()]


def _coverage(y, line) -> float:
    return float(np.mean(np.asarray(y) <= np.asarray(line)) * 100)


def _fit_k(y, q50, line, target: float) -> float:
    """k so that coverage of q50 + k * (line - q50) reaches target % (coverage only grows with k)."""
    lo, hi = K_RANGE
    for _ in range(50):
        k = (lo + hi) / 2
        if _coverage(y, q50 + k * (line - q50)) < target:
            lo = k
        else:
            hi = k
    return float(hi)


def fit_shifts(val_lines: pd.DataFrame, val_actual: pd.DataFrame, cfg: dict) -> dict:
    """{"q50": unit shift, "q80": k, "q90": k, "q95": k} from the validation rows given."""
    tol = cfg["forecast"]["calibration_tolerance_pts"]
    d = _open_rows(val_lines, val_actual, cfg)
    y, q50 = d["units"].to_numpy(float), d["q50"].to_numpy(float)
    shifts = {"q50": 0.0}
    if abs(_coverage(y, q50) - 50) > tol:
        shifts["q50"] = float(np.quantile(y - q50, 0.5))
    for name in UPPER:
        target = QUANTILES[name] * 100
        line = d[name].to_numpy(float)
        shifts[name] = 1.0 if abs(_coverage(y, line) - target) <= tol else _fit_k(y, q50, line, target)
    return shifts


def apply_shifts(lines: pd.DataFrame, shifts: dict, cfg: dict) -> pd.DataFrame:
    """Scale the upper lines around the original q50, shift q50, then post-process."""
    out = lines.copy()
    q50 = lines["q50"].to_numpy(float)
    for name in UPPER:
        out[name] = q50 + shifts[name] * (lines[name].to_numpy(float) - q50)
    out["q50"] = q50 + shifts["q50"]
    return postprocess(out, cfg)


def coverage_table(lines: pd.DataFrame, actual: pd.DataFrame, cfg: dict) -> dict:
    """Coverage % of each line on open days."""
    d = _open_rows(lines, actual, cfg)
    return {name: _coverage(d["units"], d[name]) for name in QUANTILES}


def cross_fold_check(folds: dict, cfg: dict) -> pd.DataFrame:
    """folds: name -> (lines, actual). For each ordered pair, fit on one and judge on the
    other: one row per line with the fitted number (k; for q50 its unit shift) and the
    coverage before / after."""
    rows = []
    for fit_on, (fl, fa) in folds.items():
        shifts = fit_shifts(fl, fa, cfg)
        for judged_on, (jl, ja) in folds.items():
            if judged_on == fit_on:
                continue
            before = coverage_table(jl, ja, cfg)
            after = coverage_table(apply_shifts(jl, shifts, cfg), ja, cfg)
            for name in QUANTILES:
                rows.append({"fit_on": fit_on, "judged_on": judged_on, "line": name,
                             "k": shifts[name], "target": QUANTILES[name] * 100,
                             "cover_before": before[name], "cover_after": after[name]})
    return pd.DataFrame(rows)


def _size_quarters(items: pd.DataFrame) -> pd.DataFrame:
    q = pd.qcut(items["avg_daily_units"], 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 biggest"])
    return items[["item_id"]].assign(size_quarter=q.astype(str).to_numpy())


def main(argv=None):
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from forecast.baseline import baseline_lines
    from forecast.data import load_config, load_items, load_sales, val_folds
    from forecast.evaluate import fit_and_score
    from forecast.metrics import score, segments
    from forecast.review import group_check, group_table

    ap = argparse.ArgumentParser(description="Task 7: calibration on both validation folds")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = load_config(args.config)
    sales, items = load_sales(cfg), load_items(cfg)
    seg = segments(sales, items, cfg)
    actual = sales[KEY + ["units"]]
    sizes = _size_quarters(items)
    tol = cfg["forecast"]["calibration_tolerance_pts"]

    folds = {}
    for f in val_folds(cfg):
        fit = fit_and_score(sales, f, seg, cfg, list(QUANTILES.values()), cfg["forecast"]["seed"])
        folds[f["name"]] = (postprocess(fit["lines"], cfg), actual, f, fit["targets"])
    pooled = pd.concat([v[0] for v in folds.values()], ignore_index=True)
    shifts = fit_shifts(pooled, actual, cfg)

    report = {"shifts": shifts, "tolerance_pts": tol, "folds": {}, "gate": {}}
    group_rows = []
    for name, (lines, act, f, targets) in folds.items():
        after = apply_shifts(lines, shifts, cfg)
        block = {"before": score(lines, act, seg, cfg).to_dict(orient="records"),
                 "after": score(after, act, seg, cfg).to_dict(orient="records"),
                 "size_quarters": {}, "peak_days": {}}
        for label, frame in (("before", lines), ("after", after)):
            d = _open_rows(frame, act, cfg).merge(sizes, on="item_id", how="left")
            block["size_quarters"][label] = {
                qn: {line: _coverage(g["units"], g[line]) for line in QUANTILES}
                for qn, g in d.groupby("size_quarter")}
            block["peak_days"][label] = {
                str(day.date()): coverage_table(frame[frame["date"] == day], act, cfg)
                for day in f["peak_days"]}
        all_after = next(r for r in block["after"] if r["segment"] == "all")
        report["gate"][name] = {line: {"coverage": all_after[f"cover_{line}"],
                                       "target": QUANTILES[line] * 100,
                                       "within": abs(all_after[f"cover_{line}"] - QUANTILES[line] * 100) <= tol}
                                for line in QUANTILES}
        report["folds"][name] = block
        stub = baseline_lines(sales, targets, cfg, method="stub")
        d = _open_rows(after, act, cfg).merge(
            stub.rename(columns={c: f"stub_{c}" for c in QUANTILES}), on=KEY, how="left")
        d = d.merge(seg[["store_id", "item_id", "is_perishable", "is_slow"]], on=["store_id", "item_id"],
                    how="left").merge(items[["item_id", "dept_id"]], on="item_id", how="left")
        d = d.merge(sizes, on="item_id", how="left").assign(fold=name)
        group_rows.append(d)
    table = group_table(pd.concat(group_rows, ignore_index=True),
                        ["is_perishable", "is_slow", "store_id", "dept_id", "size_quarter"])
    flagged = group_check(table)
    report["group_check"] = {"table": table.to_dict(orient="records"),
                             "loses_on_both_folds": flagged.to_dict(orient="records")}
    cross = cross_fold_check({n: (v[0], v[1]) for n, v in folds.items()}, cfg)
    report["cross_fold_check"] = cross.to_dict(orient="records")
    report["gate_passed"] = all(v["within"] for g in report["gate"].values() for v in g.values())

    out = Path(cfg["forecast"]["scores_dir"]) / "model_val_calibrated.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 160, "display.precision", 3):
        print("shifts:", shifts)
        for name, g in report["gate"].items():
            print(name, {k: round(v["coverage"], 2) for k, v in g.items()})
        print(cross.to_string(index=False))
        print("groups losing to the stub on both folds:", flagged.to_dict(orient="records") or "none")
    print(f"gate {'passed' if report['gate_passed'] else 'NOT passed'}; wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
