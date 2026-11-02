"""Score a forecasting method on both validation folds and save the scores.

For each fold in forecast.val_folds: forecast every store-item day in the fold, score it
(all / perishable / slow, closed days left out), and score the fold's peak days on their
own. Writes <scores_dir>/<method>_val.json, one block per fold. Test-window scores are
never computed here (rule R10: only after the replay).

The model methods train on each fold's training rows (cutoff = that fold's val_start):
  model_q50  the median line only (Task 4); writes model_q50_val.json with the q50 scores,
             the validation curve (score after every 50 trees) and the fit time.
  model      all four lines, post-processed (Task 5); writes model_val_untuned.json (scores,
             a curve per line, leak check), lines_vs_baselines.json (upper lines vs both
             baselines; needs baseline_val.json and stub_val.json) and error_review.json
             (worst store-items per fold, shuffle test on forecast.review.leak_fold).
--seeds trains once per seed; the JSON's scores are the first seed's, the seed runs are
listed too. --log "what changed" adds a row to experiments.csv with each fold's mean score
over the seeds, kept only if it beats the last kept row on every fold (keep_change).

Run it from the retailops folder:
    python -m forecast.evaluate --config config.yaml --method baseline
    python -m forecast.evaluate --config config.yaml --method stub
    python -m forecast.evaluate --config config.yaml --method model_q50 --seeds 1 2 3 --log base
    python -m forecast.evaluate --config config.yaml --method model
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast.baseline import baseline_lines, targets_for  # noqa: E402
from forecast.data import load_config, load_items, load_sales, val_folds, windows  # noqa: E402
from forecast.experiments import fold_score, keep_change, log_experiment, read_log  # noqa: E402
from forecast.features import build_features  # noqa: E402
from forecast.metrics import closed_mask, score, segments  # noqa: E402
from forecast.model import LINE_NAME, fit_models, model_params, predict_lines, val_curve  # noqa: E402
from forecast.postprocess import postprocess  # noqa: E402
from forecast.review import (  # noqa: E402
    compare_lines, item_traits, leak_check, shuffle_importance, worst_items,
)

# CLI name -> baseline_lines method ("baseline" is the plan's name for the naive one).
METHODS = {"baseline": "naive", "stub": "stub"}
# CLI name -> (quantiles to train, output file).
MODEL_METHODS = {"model_q50": ([0.5], "model_q50_val.json"),
                 "model": ([0.5, 0.8, 0.9, 0.95], "model_val_untuned.json")}
Q50_COLS = ["segment", "n", "mae", "pinball_q50", "cover_q50"]


def settings_by_line(cfg: dict) -> dict:
    """Each line's model settings, by line name (for the JSON files and the log)."""
    return {LINE_NAME[q]: p for q, p in model_params(cfg).items()}


def evaluate_fold(sales, fold, seg, cfg, method: str) -> dict:
    """Scores for one fold: the whole fold, then each peak day on its own."""
    targets = targets_for(sales, fold["val_start"], fold["val_end"])
    lines = baseline_lines(sales, targets, cfg, method=method)
    actual = sales[["date", "store_id", "item_id", "units"]]
    block = {
        "val_start": str(fold["val_start"].date()),
        "val_end": str(fold["val_end"].date()),
        "scores": score(lines, actual, seg, cfg).to_dict(orient="records"),
        "peak_days": {},
    }
    for day in fold["peak_days"]:
        one = lines[lines["date"] == day]
        block["peak_days"][str(day.date())] = score(one, actual, seg, cfg).to_dict(orient="records")
    return block


def score_q50(lines, actual, seg, cfg) -> list:
    """q50 scores only (the upper lines are copies of q50 just to satisfy score())."""
    full = lines.assign(q80=lines["q50"], q90=lines["q50"], q95=lines["q50"])
    return score(full, actual, seg, cfg)[Q50_COLS].to_dict(orient="records")


def score_lines(lines, actual, seg, cfg) -> list:
    """q50 only for a median-only model; all four lines (post-processed) otherwise."""
    if "q80" not in lines:
        return score_q50(lines, actual, seg, cfg)
    return score(postprocess(lines, cfg), actual, seg, cfg).to_dict(orient="records")


def fit_and_score(sales, fold, seg, cfg, quantiles, seed) -> dict:
    """One fit on the fold's training rows (cutoff = val_start), scored on the fold. Returns the
    run summary plus the bundle, lines and inputs the review needs."""
    past = sales[sales["date"] < fold["val_start"]]
    targets = targets_for(sales, fold["val_start"], fold["val_end"])
    actual = sales[["date", "store_id", "item_id", "units"]]
    y = targets.merge(actual, on=["date", "store_id", "item_id"], how="left")["units"].to_numpy()
    open_day = ~closed_mask(targets["date"], cfg).to_numpy()      # curves skip 25 Dec too
    t0 = time.time()
    bundle = fit_models(past, fold["val_start"], windows(cfg)["train_start"], cfg, quantiles,
                        model_params(cfg), seed)
    fit_s = time.time() - t0
    lines = predict_lines(bundle, sales, targets, cfg)
    scores = score_lines(lines, actual, seg, cfg)
    X = build_features(sales, targets, cfg, bundle.encoders)[open_day]
    y_open = y[open_day]
    run = {"seed": int(seed), "fit_seconds": round(fit_s, 1), "train_rows": bundle.train_rows,
           "mae": scores[0]["mae"],
           **{k: v for k, v in scores[0].items() if k.startswith("pinball_")}}
    for q, model in bundle.models.items():
        run[f"val_curve_{LINE_NAME[q]}"] = [[t, p] for t, p in val_curve(model, X, y_open, q, step=50)]
    return {"run": run, "scores": scores, "bundle": bundle, "lines": lines, "targets": targets,
            "X": X, "y": y_open}


def evaluate_model_fold(sales, fold, seg, cfg, quantiles, seeds, keep=None) -> dict:
    """Train once per seed and score the fold. The block's scores are the first seed's; if
    `keep` is a dict, the first seed's fit (bundle, lines, inputs) is stored in it."""
    block = {"val_start": str(fold["val_start"].date()), "val_end": str(fold["val_end"].date()),
             "params": settings_by_line(cfg), "seed_runs": []}
    actual = sales[["date", "store_id", "item_id", "units"]]
    for k, seed in enumerate(seeds):
        fit = fit_and_score(sales, fold, seg, cfg, quantiles, seed)
        block["seed_runs"].append(fit["run"])
        if k == 0:
            block["scores"] = fit["scores"]
            block["peak_days"] = {str(d.date()): score_lines(fit["lines"][fit["lines"]["date"] == d],
                                                             actual, seg, cfg)
                                  for d in fold["peak_days"]}
            if keep is not None:
                keep.update(fit)
    runs = block["seed_runs"]
    keys = ["mae"] + [k for k in runs[0] if k.startswith("pinball_")]
    block["mean_over_seeds"] = {k: sum(r[k] for r in runs) / len(runs) for k in keys}
    return block


def review_fold(sales, items, fold, fit, cfg, shuffle: bool) -> dict:
    """Error review for one fold (Task 5): the worst store-items and, on the leak fold, the
    shuffle test. Reports only."""
    r = cfg["forecast"]["review"]
    actual = sales[["date", "store_id", "item_id", "units"]]
    stub = baseline_lines(sales, fit["targets"], cfg, method="stub")
    traits = item_traits(sales, items, cfg, cutoff=fold["val_start"])
    worst = worst_items(fit["lines"], actual, stub, traits, cfg, n=r["worst_n"])
    worst["first_day"] = worst["first_day"].dt.strftime("%Y-%m-%d")
    out = {"worst_items": worst.to_dict(orient="records")}
    if shuffle:
        imp = shuffle_importance(fit["bundle"].models[0.5], fit["X"], fit["y"], 0.5,
                                 n_repeats=r["shuffle_repeats"], seed=cfg["forecast"]["seed"])
        out["shuffle_q50"] = imp.to_dict(orient="records")
    return out


def evaluate_model(cfg: dict, method_name: str, seeds) -> dict:
    """Model scores per fold. For the full model (all four lines) also the Task 5 reviews:
    upper lines vs both baselines, worst store-items, shuffle test and leak check."""
    quantiles, _ = MODEL_METHODS[method_name]
    sales = load_sales(cfg)
    items = load_items(cfg)
    seg = segments(sales, items, cfg)
    result = {"method": method_name, "seeds": [int(s) for s in seeds], "folds": {}}
    fits = {}
    for f in val_folds(cfg):
        fits[f["name"]] = {}
        result["folds"][f["name"]] = evaluate_model_fold(sales, f, seg, cfg, quantiles, seeds,
                                                         keep=fits[f["name"]])
    result["leak_check"] = leak_check({k: b["scores"][0]["mae"] for k, b in result["folds"].items()}, cfg)
    if len(quantiles) == 1:
        return result
    scores_dir = Path(cfg["forecast"]["scores_dir"])
    base = {m: json.loads((scores_dir / f"{m}_val.json").read_text(encoding="utf-8"))["folds"]
            for m in ("baseline", "stub")}
    model_scores = {k: b["scores"] for k, b in result["folds"].items()}
    result["lines_vs_baselines"] = compare_lines(
        model_scores, {k: b["scores"] for k, b in base["baseline"].items()},
        {k: b["scores"] for k, b in base["stub"].items()}).to_dict(orient="records")
    leak_fold = cfg["forecast"]["review"]["leak_fold"]
    result["review"] = {f["name"]: review_fold(sales, items, f, fits[f["name"]], cfg,
                                               shuffle=f["name"] == leak_fold)
                        for f in val_folds(cfg)}
    return result


def log_try(cfg: dict, result: dict, change: str) -> dict:
    """Add one row to experiments.csv. A try can be kept only if it ran on forecast.keep_seeds,
    shows no suspected leak, and beats the last kept row of the same method and seeds on every
    fold (the first such row is the base). Other tries are logged as exploring, never kept."""
    path = Path(cfg["forecast"]["scores_dir"]) / "experiments.csv"
    seeds = " ".join(map(str, result["seeds"]))
    after = {f: b["mean_over_seeds"]["pinball_q50"] for f, b in result["folds"].items()}
    row = {"change": change, "method": result["method"], "seeds": seeds,
           "params": json.dumps(settings_by_line(cfg), sort_keys=True)}
    for f, b in result["folds"].items():
        row[f"{f}_pinball_q50"] = b["mean_over_seeds"]["pinball_q50"]
        row[f"{f}_mae"] = b["mean_over_seeds"]["mae"]
    keep_seeds = " ".join(map(str, cfg["forecast"]["keep_seeds"]))
    log = read_log(path)
    same = log[log["kept"] & (log["method"] == result["method"]) & (log["seeds"] == seeds)] \
        if len(log) else log
    if result.get("leak_check"):
        row["kept"], row["why"] = False, "suspected leak: " + result["leak_check"]
    elif seeds != keep_seeds:
        row["kept"], row["why"] = False, f"exploring on seed(s) {seeds}; keep needs {keep_seeds}"
    elif len(same):
        last = same.iloc[-1]
        before = {f: fold_score(last[f"{f}_pinball_q50"]) for f in after}
        row["kept"] = keep_change(before, after)
        row["why"] = ("lower on every fold" if row["kept"] else "not lower on every fold") + \
            f" than '{last['change']}'"
    else:
        row["kept"], row["why"] = True, "base row"
    log_experiment(path, row)
    return row


def evaluate(cfg: dict, method_name: str) -> dict:
    method = METHODS[method_name]
    sales = load_sales(cfg)
    seg = segments(sales, load_items(cfg), cfg)
    return {"method": method_name,
            "folds": {f["name"]: evaluate_fold(sales, f, seg, cfg, method) for f in val_folds(cfg)}}


def summary_table(result: dict) -> pd.DataFrame:
    """One row per fold and group: MAE and coverage of each line (for printing)."""
    rows = []
    for fold, block in result["folds"].items():
        for r in block["scores"]:
            rows.append({"fold": fold, **r})
    cols = ["fold", "segment", "n", "mae", "cover_q50", "cover_q80", "cover_q90", "cover_q95",
            "cover_used_line", "used_line_target"]
    return pd.DataFrame(rows)[cols]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--method", choices=sorted(METHODS) + sorted(MODEL_METHODS), default="baseline")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="model methods: one fit per seed (default: forecast.seed)")
    ap.add_argument("--log", default=None, metavar="CHANGE",
                    help="model methods: add a row to experiments.csv describing this try")
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config)
    out_dir = Path(cfg["forecast"]["scores_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.method in MODEL_METHODS:
        seeds = args.seeds or [cfg["forecast"]["seed"]]
        result = evaluate_model(cfg, args.method, seeds)
        path = out_dir / MODEL_METHODS[args.method][1]
        extra = {"lines_vs_baselines.json": result.pop("lines_vs_baselines", None),
                 "error_review.json": result.pop("review", None)}
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        for name, part in extra.items():
            if part is not None:
                (out_dir / name).write_text(json.dumps(part, indent=2), encoding="utf-8")
                print(f"wrote {out_dir / name}")
        if extra["lines_vs_baselines.json"]:
            with pd.option_context("display.width", 160, "display.precision", 4):
                print(pd.DataFrame(extra["lines_vs_baselines.json"]).to_string(index=False))
                print(summary_table(result).to_string(index=False))
        if result["leak_check"]:
            print("WARNING:", result["leak_check"])
        for fold, b in result["folds"].items():
            for r in b["seed_runs"]:
                print(f"{fold} seed {r['seed']}: mae {r['mae']:.3f} pinball_q50 {r['pinball_q50']:.4f} "
                      f"fit {r['fit_seconds']}s on {r['train_rows']} rows")
            print(f"{fold} mean over seeds: mae {b['mean_over_seeds']['mae']:.3f}")
        if args.log:
            row = log_try(cfg, result, args.log)
            print(f"experiments.csv: '{args.log}' kept={row['kept']} ({row['why']})")
        if result["leak_check"]:
            print(f"wrote {path}")
            sys.exit(f"stopped: {result['leak_check']}; diagnose before going on")
    else:
        result = evaluate(cfg, args.method)
        path = out_dir / f"{args.method}_val.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        with pd.option_context("display.width", 160, "display.precision", 2):
            print(summary_table(result).to_string(index=False))
    print(f"wrote {path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
