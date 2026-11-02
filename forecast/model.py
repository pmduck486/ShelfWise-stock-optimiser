"""The trained forecaster: one global quantile model per line (Tasks 4 and 5).

What this file does:
  - fit_quantile: one HistGradientBoostingRegressor with quantile loss. scikit-learn's own
    early stopping is off (fix 1, 2026-10-30): it holds back a random tenth of the training
    rows, which share days with the rows it trains on, so it never stopped and only wasted
    rows. The tree count is chosen from the validation curve instead (val_curve, Task 6).
  - model_params: each line's settings (forecast.params + forecast.line_params).
  - fit_models: everything a model needs, fitted on rows before its cutoff only (R1, R4):
    category codes, the long zero-run rule, the training rows, their inputs, one model per
    line. Refuses sales that reach the cutoff instead of trimming them.
  - predict_lines: q50..q95 for target rows, in the targets' order, with the dead-item rule
    on q50 (decision 26). With calibration numbers in the bundle (Task 7) they are applied and
    the lines post-processed; without, the lines are raw (evaluate post-processes them).
  - val_curve: validation pinball after every `step` trees, from one fitted model.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forecast.data import check_cutoff
from forecast.features import (
    KEY, build_features, categorical_mask, feature_names, fit_encoders, training_rows,
    zero_run_stats,
)
from forecast.metrics import QUANTILES, pinball
from forecast.calibrate import apply_shifts
from forecast.postprocess import dead_item_rule

LINE_NAME = {q: name for name, q in QUANTILES.items()}      # 0.5 -> "q50", ...


@dataclass
class Bundle:
    """A fitted forecaster: one model per quantile plus what it was fitted with."""
    models: dict            # quantile -> fitted estimator
    encoders: dict          # category codes and normal prices (fit_encoders)
    zero_stats: dict        # long zero-run rule (zero_run_stats)
    features: list          # input columns, in model order
    cutoff: pd.Timestamp    # first day the model may not have seen
    params: dict            # model settings per line: "q50" -> settings, ...
    seed: int
    train_rows: int
    shifts: dict = field(default_factory=dict)      # calibration numbers (Task 7): line -> k or shift


def fit_quantile(X, y, q: float, params: dict, seed: int, cat_mask) -> HistGradientBoostingRegressor:
    """One quantile model. Every tree in params["max_iter"] is built (early stopping off)."""
    model = HistGradientBoostingRegressor(loss="quantile", quantile=q, early_stopping=False,
                                          random_state=seed, categorical_features=cat_mask,
                                          **params)
    return model.fit(X, y)


def model_params(cfg: dict) -> dict:
    """Settings for each line: forecast.params, overridden per line by forecast.line_params
    (Task 6 picks each line's settings and tree count on its own)."""
    f = cfg["forecast"]
    extra = f.get("line_params") or {}
    unknown = sorted(set(extra) - set(QUANTILES))
    if unknown:
        raise ValueError(f"forecast.line_params has unknown lines {unknown}; use {list(QUANTILES)}")
    return {q: {**f["params"], **extra.get(name, {})} for name, q in QUANTILES.items()}


def fit_models(sales: pd.DataFrame, cutoff, train_start, cfg: dict, quantiles, params: dict,
               seed=None) -> Bundle:
    """Fit one model per quantile on training rows train_start .. cutoff - 1 day. `params` is
    one settings dict for every line, or {quantile: settings} (model_params).

    Raises ValueError if `sales` holds any row on or after `cutoff` (R1).
    """
    cutoff = pd.Timestamp(cutoff)
    check_cutoff(sales, cutoff)
    seed = cfg["forecast"]["seed"] if seed is None else seed
    encoders = fit_encoders(sales, cutoff=cutoff)
    stats = zero_run_stats(sales, cfg, cutoff=cutoff)
    rows = training_rows(sales, pd.Timestamp(train_start), cutoff - pd.Timedelta(days=1), cfg, stats)
    X = build_features(sales, rows[KEY], cfg, encoders)
    y = rows["units"].to_numpy(dtype=float)
    mask = categorical_mask(cfg)
    per_line = {float(q): dict(params[float(q)] if float(q) in params else params) for q in quantiles}
    models = {q: fit_quantile(X, y, q, per_line[q], seed, mask) for q in per_line}
    return Bundle(models=models, encoders=encoders, zero_stats=stats, features=list(X.columns),
                  cutoff=cutoff, params={LINE_NAME[q]: p for q, p in per_line.items()},
                  seed=int(seed), train_rows=int(len(rows)))


def predict_lines(bundle: Bundle, sales: pd.DataFrame, targets: pd.DataFrame,
                  cfg: dict) -> pd.DataFrame:
    """Raw lines for each target row: date, store_id, item_id, then one column per quantile
    in the bundle (q50, q80, ...), in the targets' order."""
    if feature_names(cfg) != bundle.features:
        raise ValueError("config inputs differ from the ones this model was trained on")
    X = build_features(sales, targets[KEY], cfg, bundle.encoders)
    out = targets[KEY].reset_index(drop=True).copy()
    for q in sorted(bundle.models):
        out[LINE_NAME[q]] = bundle.models[q].predict(X)
    if "q50" in out:
        out = dead_item_rule(out, X["roll_mean_28"].to_numpy(), cfg)
    if bundle.shifts:
        out = apply_shifts(out, bundle.shifts, cfg)
    return out


def val_curve(model, X, y, q: float, step: int = 50) -> list:
    """[(trees, pinball)] after every `step` trees and after the last tree."""
    y = np.asarray(y, dtype=float)
    last = model.n_iter_
    return [(i, pinball(y, pred, q)) for i, pred in enumerate(model.staged_predict(X), start=1)
            if i % step == 0 or i == last]
