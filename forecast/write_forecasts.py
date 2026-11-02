"""The forecasts file for the order model (Task 9).

write_forecasts runs forecast.predict once per origin with the frozen model and writes every
origin x horizon_days (2) x store-item to one parquet file in the stub's exact columns, so Lump 2 swaps
forecasts_stub.parquet for forecasts.parquet and changes nothing else. Laptop only, like the
freeze (loophole 6). forecasts_for(origin) is the fast path: one origin read back from the file.

Run it on the laptop from the retailops folder (after python -m forecast.freeze):
    python -m forecast.write_forecasts --config config.yaml
origins = data.windows.test_start .. test_end (56); writes data/processed/forecasts.parquet.
"""

import sys
from pathlib import Path

import pandas as pd

from forecast.freeze import check_platform, load_frozen
from forecast.predict import predict


def default_path(cfg: dict) -> Path:
    return Path(cfg["data"]["processed_dir"]) / "forecasts.parquet"


def write_forecasts(sales: pd.DataFrame, origins, cfg: dict, path, bundle=None) -> pd.DataFrame:
    check_platform(cfg, "writing forecasts.parquet")
    origins = sorted(pd.Timestamp(o) for o in origins)
    if bundle is None:
        bundle = load_frozen(cfg["forecast"]["model_dir"], origins[0], cfg)
    pairs = sales[["store_id", "item_id"]].drop_duplicates()
    parts, fallback = [], 0
    for o in origins:
        p = predict(sales, o, bundle=bundle, cfg=cfg, pairs=pairs)
        fallback += p.attrs["fallback_pairs"]
        parts.append(p)
    fc = pd.concat(parts, ignore_index=True)
    fc.attrs = {}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(path, index=False)
    fc.attrs["fallback_pairs"] = fallback
    return fc


def forecasts_for(origin, cfg: dict, path=None) -> pd.DataFrame:
    """The horizon days x store-items of one origin, read back from the written file."""
    fc = pd.read_parquet(path or default_path(cfg), filters=[("origin", "==", pd.Timestamp(origin))])
    fc.attrs = {}
    return fc.reset_index(drop=True)


def main(argv=None):
    import argparse
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from forecast.data import load_config, load_sales, windows

    ap = argparse.ArgumentParser(description="Task 9: write forecasts.parquet")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = load_config(args.config)
    w = windows(cfg)
    origins = pd.date_range(w["test_start"], w["test_end"], freq="D")
    path = default_path(cfg)
    fc = write_forecasts(load_sales(cfg), origins, cfg, path)
    print(f"{len(origins)} origins, {len(fc)} rows, {fc.attrs['fallback_pairs']} fallback "
          f"store-item-origins; wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
