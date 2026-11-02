"""Scores for the slides (Task 10).

write_report reads the saved validation scores (Tasks 3, 5 and 7) and writes, into
outputs/forecast/report/:
  - summary.md: per fold, MAE and pinball of the naive baseline, the stub and the final
    (calibrated) model; coverage of each line against its target; the calibration numbers;
    the notes the slides need (25 Dec, 0 = 0 ties).
  - mae_vs_baselines.png: MAE per fold, model against both baselines (lower is better).
  - coverage.png: each line's coverage per fold against its target.
  - example_<store>_<item>.png (when an example is given and forecasts.parquet exists): one
    store-item's day-ahead lines over the test origins, with the actual sales where known.
It only reads score files and never retrains.

Run it from the retailops folder:
    python -m forecast.report --config config.yaml [--example CA_1 FOODS_3_090]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from forecast.metrics import QUANTILES  # noqa: E402

LINES = list(QUANTILES)
# validated default categorical slots 1-3 (dataviz palette), fixed order; ink for text
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
METHODS = [("naive baseline", "baseline_val.json", AQUA), ("stub (4-week average)", "stub_val.json", ORANGE)]


def _all_row(rows: list) -> dict:
    return next(r for r in rows if r["segment"] == "all")


def collect(scores_dir: Path) -> dict:
    """{fold: {method: all-products score row}}, plus the calibration report."""
    cal = json.loads((scores_dir / "model_val_calibrated.json").read_text(encoding="utf-8"))
    out = {}
    for fold, block in cal["folds"].items():
        out[fold] = {}
        for name, file, _ in METHODS:
            d = json.loads((scores_dir / file).read_text(encoding="utf-8"))
            out[fold][name] = _all_row(d["folds"][fold]["scores"])
        out[fold]["model"] = _all_row(block["after"])
    return {"scores": out, "calibration": cal}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _mae_chart(scores: dict, path: Path):
    folds = list(scores)
    names = [m[0] for m in METHODS] + ["model"]
    colors = [m[2] for m in METHODS] + [BLUE]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    _style(ax)
    w = 0.26
    for i, (name, c) in enumerate(zip(names, colors)):
        xs = [j + (i - 1) * w for j in range(len(folds))]
        vals = [scores[f][name]["mae"] for f in folds]
        ax.bar(xs, vals, width=w - 0.02, color=c, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(folds)), folds, color=INK)
    ax.set_ylabel("MAE of the middle line (q50), units per day", color=INK2)
    ax.set_title("Daily error per store-item: lower is better", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _coverage_chart(scores: dict, path: Path):
    folds = list(scores)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    _style(ax)
    w = 0.36
    for i, (fold, c) in enumerate(zip(folds, [BLUE, ORANGE])):
        xs = [j + (i - 0.5) * w for j in range(len(LINES))]
        vals = [scores[fold]["model"][f"cover_{q}"] for q in LINES]
        ax.bar(xs, vals, width=w - 0.02, color=c, label=fold)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    for j, q in enumerate(LINES):
        t = QUANTILES[q] * 100
        ax.plot([j - w - 0.04, j + w + 0.04], [t, t], color=INK, linewidth=2)
    ax.plot([], [], color=INK, linewidth=2, label="target")
    ax.set_xticks(range(len(LINES)), LINES, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of days with sales at or below the line", color=INK2)
    ax.set_title("Coverage of each line against its target (25 Dec left out)", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _example_chart(cfg: dict, store: str, item: str, path: Path) -> bool:
    fc_path = Path(cfg["data"]["processed_dir"]) / "forecasts.parquet"
    if not fc_path.exists():
        return False
    fc = pd.read_parquet(fc_path, filters=[("store_id", "==", store), ("item_id", "==", item)])
    fc = fc[fc["date"] == fc["origin"]].sort_values("date")
    sales = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "sales.parquet",
                            columns=["date", "store_id", "item_id", "units"])
    act = sales[(sales["store_id"] == store) & (sales["item_id"] == item)
                & sales["date"].between(fc["date"].min(), fc["date"].max())]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.fill_between(fc["date"], fc["q50"], fc["q95"], color=BLUE, alpha=0.18, linewidth=0, label="q50 to q95")
    ax.plot(fc["date"], fc["q90"], color=BLUE, linewidth=1, linestyle="--", label="q90")
    ax.plot(fc["date"], fc["q50"], color=BLUE, linewidth=2, label="q50")
    ax.plot(act["date"], act["units"], color=ORANGE, linewidth=2, marker="o", markersize=3, label="actual")
    ax.set_ylabel("units per day", color=INK2)
    ax.set_title(f"{store} {item}: day-ahead lines over the test origins", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=4)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def _summary(data: dict, example_file) -> str:
    scores, cal = data["scores"], data["calibration"]
    tol = cal["tolerance_pts"]
    out = ["# Forecaster scores for the slides", "",
           "All numbers are on the two validation folds (the test window is never scored here). "
           "Scores leave out 25 Dec: stores are closed, every line is 0 and there are no sales.", "",
           "## Error (MAE and pinball, lower is better)", ""]
    head = "| fold | method | MAE | " + " | ".join(f"pinball {q}" for q in LINES) + " |"
    out += [head, "|" + "---|" * (3 + len(LINES))]
    for fold, by in scores.items():
        for name, r in by.items():
            out.append(f"| {fold} | {name} | {r['mae']:.3f} | "
                       + " | ".join(f"{r[f'pinball_{q}']:.3f}" for q in LINES) + " |")
    out += ["", "Model MAE against the stub (the order model's placeholder):", ""]
    for fold, by in scores.items():
        m, s = by["model"]["mae"], by["stub (4-week average)"]["mae"]
        out.append(f"- {fold}: {m:.3f} vs {s:.3f}, {100 * (s - m) / s:.1f}% lower")
    out += ["", "## Coverage (share of days with sales at or below each line)", "",
            "| fold | " + " | ".join(f"{q} (target {QUANTILES[q] * 100:.0f}%)" for q in LINES) + " |",
            "|" + "---|" * (1 + len(LINES))]
    for fold, by in scores.items():
        out.append(f"| {fold} | " + " | ".join(f"{by['model'][f'cover_{q}']:.1f}%" for q in LINES) + " |")
    out += ["", f"Calibration numbers (Task 7, tolerance {tol} points): "
            + ", ".join(f"{k} = {v:g}" for k, v in cal["shifts"].items())
            + (". All lines were already within tolerance, so nothing was stretched."
               if all(v == (0.0 if k == "q50" else 1.0) for k, v in cal["shifts"].items())
               else ". Coverage above is after these numbers were applied."), "",
            "## Notes", "",
            "- 25 Dec: left out of every score, for the model and both baselines alike. The "
            "forecasts file still has 25 Dec rows, with all four lines set to 0.",
            "- q50 coverage above 50% on the festive fold comes from 0 = 0 ties: on a day with no "
            "sales and a q50 of 0 the day counts as covered. Counting ties as half gives about 51%.",
            "- MAE = 2 x the q50 pinball, so the two q50 columns rank methods the same way."]
    if example_file:
        out += ["", f"Example store-item chart: {example_file}"]
    out += ["", "Charts: mae_vs_baselines.png, coverage.png"]
    return "\n".join(out) + "\n"


def write_report(cfg: dict, scores_dir=None, out_dir=None, example=None) -> Path:
    scores_dir = Path(scores_dir or cfg["forecast"]["scores_dir"])
    out_dir = Path(out_dir or Path(cfg["forecast"]["scores_dir"]).parent / "report")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = collect(scores_dir)
    _mae_chart(data["scores"], out_dir / "mae_vs_baselines.png")
    _coverage_chart(data["scores"], out_dir / "coverage.png")
    example_file = None
    if example:
        store, item = example
        name = f"example_{store}_{item}.png"
        if _example_chart(cfg, store, item, out_dir / name):
            example_file = name
    path = out_dir / "summary.md"
    path.write_text(_summary(data, example_file), encoding="utf-8")
    return path


def main(argv=None):
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from forecast.data import load_config

    ap = argparse.ArgumentParser(description="Task 10: scores and charts for the slides")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--example", nargs=2, metavar=("STORE", "ITEM"), default=None)
    args = ap.parse_args(argv)
    print("wrote", write_report(load_config(args.config), example=args.example))


if __name__ == "__main__":
    main()
