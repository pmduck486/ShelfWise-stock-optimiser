"""The frozen model (Task 8).

  - freeze: retrains every line on train_start .. cutoff - 1 day with the final settings
    (forecast.params + forecast.line_params), puts the calibration numbers (Task 7) inside, and
    saves bundle.joblib plus manifest.json: cutoff, windows, inputs, quantiles, settings per
    line, calibration numbers, seed, scikit-learn version, config fingerprint, time, platform
    and training row count.
  - Laptop only (loophole 6, user's call 2026-10-31): freeze refuses to run unless
    platform.system() equals forecast.freeze_platform ("Windows"), so the one frozen model and
    the forecasts file made from it both come from the laptop's own scikit-learn. "any" turns
    the guard off (tests only).
  - load_frozen: loads the bundle and refuses it if its cutoff is after the replay start (it
    would have seen replay days), if the forecast config changed since freezing, or if this
    scikit-learn is a different version from the one that froze it.

The config fingerprint is the sha256 of the forecast section and data.windows written as
sorted JSON, so line endings and comments in config.yaml do not change it; model_dir,
scores_dir and freeze_platform are left out (where files go does not change the model), and
the other path settings (calendar_file) count by file name only, so the same config.yaml gives
the same fingerprint in any folder (re-stamped 2026-11-02 after the project was moved).

Run it on the laptop from the retailops folder (after Task 7):
    python -m forecast.freeze --config config.yaml
cutoff = data.windows.test_start (2015-11-01); writes outputs/forecast/model/.
"""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from forecast.data import PATH_KEYS, check_cutoff, windows
from forecast.metrics import QUANTILES
from forecast.model import fit_models, model_params

NOT_HASHED = {"model_dir", "scores_dir", "freeze_platform"}


def check_platform(cfg: dict, what: str) -> None:
    want = cfg["forecast"]["freeze_platform"]
    here = platform.system()
    if want != "any" and here != want:
        raise RuntimeError(f"{what} runs on the laptop only: forecast.freeze_platform is {want!r} "
                           f"but this platform is {here!r} (loophole 6)")


def config_sha256(cfg: dict) -> str:
    f = {k: v for k, v in cfg["forecast"].items() if k not in NOT_HASHED}
    f.update({k: Path(f[k]).name for k in PATH_KEYS if k in f})   # file name only: folder-independent
    text = json.dumps({"forecast": f, "windows": cfg["data"]["windows"]}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def saved_shifts(cfg: dict) -> dict:
    p = Path(cfg["forecast"]["scores_dir"]) / "model_val_calibrated.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run python -m forecast.calibrate first (Task 7)")
    return json.loads(p.read_text(encoding="utf-8"))["shifts"]


def freeze(sales: pd.DataFrame, cfg: dict, cutoff, out_dir, shifts=None) -> Path:
    """Fit, save and describe the frozen model in out_dir. Returns out_dir."""
    check_platform(cfg, "freezing the model")
    cutoff = pd.Timestamp(cutoff)
    check_cutoff(sales, cutoff)
    shifts = saved_shifts(cfg) if shifts is None else dict(shifts)
    w = windows(cfg)
    bundle = fit_models(sales, cutoff, w["train_start"], cfg, list(QUANTILES.values()), model_params(cfg))
    bundle.shifts = shifts
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "bundle.joblib")
    manifest = {
        "cutoff": str(cutoff.date()),
        "windows": {k: str(v.date()) for k, v in w.items()},
        "features": bundle.features,
        "quantiles": sorted(bundle.models),
        "params": bundle.params,
        "shifts": shifts,
        "seed": bundle.seed,
        "sklearn_version": sklearn.__version__,
        "config_sha256": config_sha256(cfg),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.release()} / Python {platform.python_version()}",
        "train_rows": bundle.train_rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def load_frozen(model_dir, replay_start, cfg: dict):
    """The frozen Bundle, after the three guards (see the module text)."""
    d = Path(model_dir)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if pd.Timestamp(m["cutoff"]) > pd.Timestamp(replay_start):
        raise ValueError(f"frozen cutoff {m['cutoff']} is after the replay start "
                         f"{pd.Timestamp(replay_start).date()}: the model has seen replay days")
    if m["config_sha256"] != config_sha256(cfg):
        raise ValueError("the forecast config changed since the model was frozen; freeze again")
    if m["sklearn_version"] != sklearn.__version__:
        raise ValueError(f"model frozen with scikit-learn {m['sklearn_version']}, "
                         f"this is {sklearn.__version__}; freeze again")
    return joblib.load(d / "bundle.joblib")


def main(argv=None):
    import argparse
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from forecast.data import load_config, load_sales

    ap = argparse.ArgumentParser(description="Task 8: freeze the final model")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--cutoff", default=None, help="default: data.windows.test_start")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = load_config(args.config)
    check_platform(cfg, "freezing the model")
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else windows(cfg)["test_start"]
    sales = load_sales(cfg)
    out = freeze(sales[sales["date"] < cutoff], cfg, cutoff, Path(cfg["forecast"]["model_dir"]))
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({k: m[k] for k in ["cutoff", "shifts", "sklearn_version", "train_rows", "platform"]}))
    print(f"froze {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
