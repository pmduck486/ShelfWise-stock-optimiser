"""Light tuning on validation only (Task 6).

  - grid: the settings in forecast.tuning_grid (at most 6). Each builds max_iter trees (600).
  - tune: for every setting and fold, fit the four lines on the fold's training rows and read
    each line's validation pinball after every 50 trees (val_curve, closed days left out). One
    model up to 600 trees gives the score of every shorter version too, because the trees are
    added one after another (built-in early stopping is off, fix 1). Sales from test_start on
    are cut first, so tuning never sees the test window (R10).
  - best_params: one setting and tree count per line: the lowest pinball averaged over both
    folds, among settings not worse than the current settings on either fold (both-folds rule).
    If none qualifies, the current settings stay.

Run it from the retailops folder:
    python -m forecast.tune --config config.yaml
writes outputs/forecast/scores/tuning.csv and tuning_best.json and prints each line's winner.
The winners go into config.yaml (forecast.line_params) by hand and are confirmed on 3 seeds
with `python -m forecast.evaluate --method model --seeds 1 2 3 --log ...`.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast.baseline import targets_for  # noqa: E402
from forecast.data import load_config, load_sales, val_folds, windows  # noqa: E402
from forecast.features import build_features  # noqa: E402
from forecast.metrics import QUANTILES, closed_mask  # noqa: E402
from forecast.model import LINE_NAME, fit_models, val_curve  # noqa: E402

MAX_COMBOS = 6
STEP = 50
SETTING = ["learning_rate", "max_leaf_nodes"]


def grid(cfg: dict) -> list:
    """Every combination in forecast.tuning_grid, merged into forecast.params."""
    g = cfg["forecast"]["tuning_grid"]
    names = sorted(g)
    combos = [dict(zip(names, values)) for values in itertools.product(*(g[n] for n in names))]
    if len(combos) > MAX_COMBOS:
        raise ValueError(f"tuning_grid gives {len(combos)} settings; keep it to {MAX_COMBOS} or fewer")
    return [{**cfg["forecast"]["params"], **c} for c in combos]


def tune(sales: pd.DataFrame, cfg: dict, seed=None) -> pd.DataFrame:
    """One row per setting x tree count x line x fold: that fold's validation pinball."""
    sales = sales[sales["date"] < windows(cfg)["test_start"]]
    quantiles = list(QUANTILES.values())
    rows = []
    for params in grid(cfg):
        for fold in val_folds(cfg):
            past = sales[sales["date"] < fold["val_start"]]
            bundle = fit_models(past, fold["val_start"], windows(cfg)["train_start"], cfg,
                                quantiles, params, seed)
            targets = targets_for(sales, fold["val_start"], fold["val_end"])
            open_day = ~closed_mask(targets["date"], cfg).to_numpy()
            y = targets.merge(sales[["date", "store_id", "item_id", "units"]],
                              on=["date", "store_id", "item_id"], how="left")["units"].to_numpy()
            X = build_features(sales, targets, cfg, bundle.encoders)[open_day]
            for q, model in bundle.models.items():
                for trees, p in val_curve(model, X, y[open_day], q, step=STEP):
                    rows.append({**{k: params[k] for k in SETTING}, "trees": trees,
                                 "line": LINE_NAME[q], "fold": fold["name"], "pinball": p})
    return pd.DataFrame(rows)


def best_params(results: pd.DataFrame, current: dict) -> dict:
    """{line: settings} with max_iter = the chosen tree count (see the module text)."""
    key = SETTING + ["trees"]
    best = {}
    for line, r in results.groupby("line", sort=False):
        wide = r.pivot_table(index=key, columns="fold", values="pinball")
        now = tuple(current[k] for k in SETTING) + (current["max_iter"],)
        if now not in wide.index:
            raise ValueError(f"{line}: the current settings {now} are not in the tuning table")
        ok = wide[(wide <= wide.loc[now]).all(axis=1)]
        pick = ok.mean(axis=1).sort_values(kind="stable").index[0]
        best[line] = {**current, **dict(zip(SETTING, pick[:2])), "max_iter": int(pick[2])}
        for k in SETTING:
            best[line][k] = type(current[k])(best[line][k])
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = load_config(args.config)
    out_dir = Path(cfg["forecast"]["scores_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results = tune(load_sales(cfg), cfg)
    results.to_csv(out_dir / "tuning.csv", index=False)
    best = best_params(results, cfg["forecast"]["params"])
    (out_dir / "tuning_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    for line, p in best.items():
        r = results[(results["line"] == line) & (results["learning_rate"] == p["learning_rate"])
                    & (results["max_leaf_nodes"] == p["max_leaf_nodes"]) & (results["trees"] == p["max_iter"])]
        print(line, p, dict(zip(r["fold"], r["pinball"].round(4))))
    print(f"wrote {out_dir / 'tuning.csv'} and tuning_best.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
