"""forecast.predict: the forecaster's one public call (Task 9).

predict(sales, origin) gives q50..q95 for every store-item on origin .. origin + horizon - 1
(forecast.horizon_days, now 2: today and tomorrow), in the forecasts.parquet columns (origin,
date, store_id, item_id, q50, q80, q90, q95), sorted like the stub. Only sales dated before the
origin are read (R1); the inputs themselves are at least forecast.min_lag_days old (R2), and
horizon_days <= min_lag_days, so every input is known on the origin's morning. Days after the last sales day still get their calendar inputs (R3).

  - The model is the frozen one (Task 8) unless a bundle is passed.
  - A store-item with no sales row before the origin (not launched yet) gets the baseline,
    which is 0 on every line; how many did is in out.attrs["fallback_pairs"].
  - Lines are calibrated (numbers in the bundle) and post-processed: never crossing, never
    below 0, all 0 on 25 Dec.
"""

from pathlib import Path

import pandas as pd

from forecast.baseline import baseline_lines
from forecast.metrics import QUANTILES
from forecast.model import predict_lines
from forecast.postprocess import postprocess

KEY = ["date", "store_id", "item_id"]
PAIR = ["store_id", "item_id"]
LINES = list(QUANTILES)
CONTRACT = ["origin"] + KEY + LINES
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def predict(sales: pd.DataFrame, origin, horizon: int = None, quantiles=None, *, bundle=None,
            cfg=None, pairs=None) -> pd.DataFrame:
    from forecast.data import load_config
    cfg = load_config(DEFAULT_CONFIG) if cfg is None else cfg
    horizon = int(cfg["forecast"]["horizon_days"]) if horizon is None else int(horizon)
    if horizon > int(cfg["forecast"]["min_lag_days"]):
        raise ValueError(f"horizon {horizon} is longer than forecast.min_lag_days "
                         f"({cfg['forecast']['min_lag_days']}): the last days would need sales "
                         f"not known on the origin's morning (rule R2)")
    origin = pd.Timestamp(origin)
    if bundle is None:
        from forecast.freeze import load_frozen
        bundle = load_frozen(cfg["forecast"]["model_dir"], origin, cfg)
    want = sorted(QUANTILES.values()) if quantiles is None else sorted(float(q) for q in quantiles)
    if want != sorted(bundle.models):
        raise ValueError(f"quantiles {want} differ from the model's {sorted(bundle.models)}")

    past = sales[sales["date"] < origin]
    pairs = (sales[PAIR] if pairs is None else pairs[PAIR]).drop_duplicates()
    seen = past[PAIR].drop_duplicates()
    has = pairs.merge(seen.assign(_h=True), on=PAIR, how="left")["_h"].fillna(False).astype(bool).to_numpy()
    dates = pd.DataFrame({"date": pd.date_range(origin, periods=horizon, freq="D")})

    parts = []
    if has.any():
        t = dates.merge(pairs[has], how="cross")[KEY]
        parts.append(postprocess(predict_lines(bundle, past, t, cfg), cfg))
    if (~has).any():
        t = dates.merge(pairs[~has], how="cross")[KEY]
        parts.append(postprocess(baseline_lines(past, t, cfg, method="stub"), cfg))
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "origin", origin)
    out["origin"] = out["origin"].astype("datetime64[ns]")
    out["date"] = out["date"].astype("datetime64[ns]")
    out[LINES] = out[LINES].astype("float64")
    out = out[CONTRACT].sort_values(["origin"] + KEY, kind="stable").reset_index(drop=True)
    out.attrs["fallback_pairs"] = int((~has).sum())
    return out
