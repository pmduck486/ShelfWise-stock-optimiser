"""Tests for the trained forecaster (Phase C): the q50 model, the experiment log and the
both-folds rule (Task 4), post-processing and the review tools (Task 5).

They train on tiny made-up data (2 stores x 5 items, a few months) with small trees, so the
whole file runs in a few seconds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast.data import load_calendar, load_config  # noqa: E402

CONFIG = ROOT / "config.yaml"
KEY = ["date", "store_id", "item_id"]
SALES_COLS = ["date", "store_id", "item_id", "dept_id", "units", "price", "is_perishable",
              "event_name"]
TINY_PARAMS = {"learning_rate": 0.1, "max_iter": 20, "max_leaf_nodes": 7, "min_samples_leaf": 5}
START = pd.Timestamp("2015-01-01")
CUTOFF = pd.Timestamp("2015-05-01")


@pytest.fixture
def cfg():
    return load_config(CONFIG)


def tiny_sales(cfg, start=START, days=150, stores=("CA_1", "TX_1"), n_items=5, seed=0):
    """Made-up daily sales: 2 stores x 5 items, a weekly pattern, the real calendar's events."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    weekly = 1 + 0.5 * (dates.dayofweek >= 5)
    frames = []
    for st in stores:
        for i in range(n_items):
            frames.append(pd.DataFrame({
                "date": dates, "store_id": st, "item_id": f"FOODS_3_{i:03d}", "dept_id": "FOODS_3",
                "units": rng.poisson((3 + 2 * i) * weekly).astype("int32"),
                "price": 2.0 + 0.5 * i, "is_perishable": i == 0,
            }))
    sales = pd.concat(frames, ignore_index=True)
    cal = load_calendar(cfg)[["date", "event_name"]]
    return sales.merge(cal, on="date", how="left")[SALES_COLS]


def targets_between(sales, start, end):
    d = sales["date"]
    return sales.loc[(d >= start) & (d <= end), KEY].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 4: the q50 model
# ---------------------------------------------------------------------------
def test_fit_refuses_rows_after_cutoff(cfg):
    """Training must never see a row on or after its cutoff (R1): refuse, don't trim."""
    from forecast.model import fit_models
    sales = tiny_sales(cfg)                      # runs past CUTOFF
    with pytest.raises(ValueError, match="cutoff"):
        fit_models(sales, CUTOFF, START, cfg, [0.5], TINY_PARAMS)


def test_q50_runs_on_tiny_data(cfg):
    from forecast.model import Bundle, fit_models, predict_lines
    sales = tiny_sales(cfg)
    past = sales[sales["date"] < CUTOFF]
    bundle = fit_models(past, CUTOFF, START, cfg, [0.5], TINY_PARAMS)
    assert isinstance(bundle, Bundle)
    assert bundle.cutoff == CUTOFF and set(bundle.models) == {0.5}
    assert bundle.shifts == {}
    targets = targets_between(sales, CUTOFF, CUTOFF + pd.Timedelta(days=6))
    lines = predict_lines(bundle, sales, targets, cfg)
    assert list(lines.columns) == KEY + ["q50"]
    assert len(lines) == len(targets) == 7 * 10
    pd.testing.assert_frame_equal(lines[KEY], targets[KEY])
    assert np.isfinite(lines["q50"]).all()


def test_fit_quantile_no_early_stopping(cfg):
    """Fix 1 (2026-10-30): scikit-learn's own early stopping is off, so every tree asked for
    is built and no random tenth of the training rows is held back."""
    from forecast.features import categorical_mask
    from forecast.model import fit_quantile
    rng = np.random.default_rng(0)
    n = 20_000                                   # above sklearn's 10,000-row "auto" switch
    X = pd.DataFrame(rng.normal(size=(n, len(categorical_mask(cfg)))).astype("float32"))
    X.iloc[:, [i for i, c in enumerate(categorical_mask(cfg)) if c]] = 0
    y = rng.poisson(5, n).astype(float)
    m = fit_quantile(X, y, 0.5, TINY_PARAMS, seed=1, cat_mask=categorical_mask(cfg))
    assert m.early_stopping is False
    assert m.n_iter_ == TINY_PARAMS["max_iter"]


def test_val_curve_every_50_trees(cfg):
    """Fix 1: one fit gives the validation score after every 50 trees, and the last point
    equals the finished model's score."""
    from forecast.metrics import pinball
    from forecast.model import fit_quantile, val_curve
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(size=3000)})
    y = 5 + 2 * X["a"].to_numpy() + rng.normal(size=3000)
    params = {**TINY_PARAMS, "max_iter": 120}
    m = fit_quantile(X[:2000], y[:2000], 0.5, params, seed=1, cat_mask=[False, False])
    curve = val_curve(m, X[2000:], y[2000:], 0.5, step=50)
    assert [t for t, _ in curve] == [50, 100, 120]
    assert curve[-1][1] == pytest.approx(pinball(y[2000:], m.predict(X[2000:]), 0.5))


# ---------------------------------------------------------------------------
# Task 4: experiment log and the both-folds rule
# ---------------------------------------------------------------------------
def test_keep_change_needs_both_folds():
    from forecast.experiments import keep_change
    before = {"fold1": 2.00, "fold2": 2.10}
    assert keep_change(before, {"fold1": 1.99, "fold2": 2.09}) is True
    assert keep_change(before, {"fold1": 1.95, "fold2": 2.11}) is False   # worse on fold 2
    assert keep_change(before, {"fold1": 2.00, "fold2": 2.05}) is False   # a tie is no gain


def test_keep_change_uses_seed_mean():
    """Fix 2 (2026-10-30): with several seeds per fold, the mean is compared, so one lucky
    seed cannot carry a change."""
    from forecast.experiments import keep_change
    before = {"fold1": [2.00, 2.02, 2.01], "fold2": [2.10, 2.10, 2.10]}
    lucky = {"fold1": [1.98, 2.05, 2.04], "fold2": [2.09, 2.09, 2.09]}     # fold 1 mean 2.0233
    real = {"fold1": [1.99, 2.00, 2.00], "fold2": [2.09, 2.08, 2.09]}
    assert keep_change(before, lucky) is False
    assert keep_change(before, real) is True


def test_keep_change_refuses_missing_fold():
    from forecast.experiments import keep_change
    with pytest.raises(ValueError, match="fold"):
        keep_change({"fold1": 2.0, "fold2": 2.1}, {"fold1": 1.9})


def test_experiment_log_appends(tmp_path):
    from forecast.experiments import log_experiment
    path = tmp_path / "experiments.csv"
    log_experiment(path, {"change": "base", "fold1_pinball": 2.03, "kept": True})
    log_experiment(path, {"change": "lr 0.05", "fold1_pinball": 2.04, "kept": False})
    text = path.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("time,") for line in text) == 1           # header once
    log = pd.read_csv(path)
    assert list(log["change"]) == ["base", "lr 0.05"]
    assert list(log["kept"]) == [True, False]


def tiny_items(sales):
    items = sales[["item_id", "dept_id", "is_perishable"]].drop_duplicates("item_id").copy()
    items["rank"] = range(1, len(items) + 1)
    items["avg_daily_units"] = sales.groupby("item_id")["units"].mean().reindex(items["item_id"]).to_numpy()
    items["shelf_life_days"] = pd.array([3 if p else pd.NA for p in items["is_perishable"]], dtype="Int64")
    return items


def test_model_curve_skips_closed_days(cfg):
    """The tree curve is scored on the same rows as the fold's scores: 25 Dec left out, so
    its last point equals the reported q50 pinball."""
    from forecast.evaluate import evaluate_model_fold
    from forecast.metrics import segments
    cfg["forecast"]["params"] = TINY_PARAMS
    cfg["data"]["windows"]["train_start"] = "2014-09-01"
    sales = tiny_sales(cfg, start=pd.Timestamp("2014-09-01"), days=130)
    fold = {"name": "f", "val_start": pd.Timestamp("2014-12-15"), "val_end": pd.Timestamp("2014-12-31"),
            "peak_days": []}
    seg = segments(sales, tiny_items(sales), cfg)
    block = evaluate_model_fold(sales, fold, seg, cfg, [0.5], [1])
    run = block["seed_runs"][0]
    assert run["val_curve_q50"][-1][1] == pytest.approx(run["pinball_q50"])


def test_log_try_compares_with_last_kept(cfg, tmp_path):
    """--log: the first row is the base; a later try is kept only if it beats the last kept
    row on every fold."""
    from forecast.evaluate import log_try
    cfg["forecast"]["scores_dir"] = str(tmp_path)

    def result(f1, f2):
        return {"method": "model_q50", "seeds": [1, 2, 3], "folds": {
            "fold1": {"mean_over_seeds": {"pinball_q50": f1, "mae": 2 * f1}},
            "fold2": {"mean_over_seeds": {"pinball_q50": f2, "mae": 2 * f2}}}}
    assert log_try(cfg, result(2.07, 2.10), "base")["kept"] is True
    assert log_try(cfg, result(2.03, 2.11), "worse on fold 2")["kept"] is False
    assert log_try(cfg, result(2.03, 2.09), "better on both")["kept"] is True
    assert log_try(cfg, result(2.04, 2.08), "worse than the last kept on fold 1")["kept"] is False
    log = pd.read_csv(tmp_path / "experiments.csv")
    assert list(log["kept"]) == [True, False, True, False]


# ---------------------------------------------------------------------------
# Task 5: post-processing
# ---------------------------------------------------------------------------
LINES = ["q50", "q80", "q90", "q95"]


def lines_frame(rows, day="2015-07-01"):
    out = pd.DataFrame(rows, columns=LINES, dtype=float)
    out.insert(0, "item_id", [f"FOODS_3_{i:03d}" for i in range(len(out))])
    out.insert(0, "store_id", "CA_1")
    out.insert(0, "date", pd.Timestamp(day))
    return out


def test_postprocess_uncrosses(cfg):
    from forecast.postprocess import postprocess
    out = postprocess(lines_frame([[5, 4, 6, 3]]), cfg)
    assert out[LINES].iloc[0].tolist() == [3, 4, 5, 6]


def test_postprocess_clips_negative(cfg):
    from forecast.postprocess import postprocess
    out = postprocess(lines_frame([[-0.4, -0.1, 0.5, 2.0]]), cfg)
    assert out[LINES].iloc[0].tolist() == [0, 0, 0.5, 2.0]


def test_postprocess_closed_days_zero(cfg):
    from forecast.postprocess import postprocess
    out = postprocess(lines_frame([[5, 6, 7, 8]], day="2015-12-25"), cfg)
    assert out[LINES].iloc[0].tolist() == [0, 0, 0, 0]


def test_postprocess_refuses_nan(cfg):
    from forecast.postprocess import postprocess
    with pytest.raises(ValueError, match="missing"):
        postprocess(lines_frame([[5, np.nan, 7, 8]]), cfg)


def test_sanity_on_model_output(cfg):
    """All four models on tiny data, then post-processing: ordered, >= 0, no NaN, and 25 Dec
    (inside the targets) is 0 on every line."""
    from forecast.model import fit_models, predict_lines
    from forecast.postprocess import postprocess
    sales = tiny_sales(cfg, start=pd.Timestamp("2014-08-01"), days=170)   # to 2015-01-17
    cut = pd.Timestamp("2014-12-20")
    bundle = fit_models(sales[sales["date"] < cut], cut, pd.Timestamp("2014-08-01"), cfg,
                        [0.5, 0.8, 0.9, 0.95], TINY_PARAMS)
    targets = targets_between(sales, cut, cut + pd.Timedelta(days=6))
    out = postprocess(predict_lines(bundle, sales, targets, cfg), cfg)
    a = out[LINES].to_numpy()
    assert np.isfinite(a).all() and (a >= 0).all()
    assert (np.diff(a, axis=1) >= 0).all()
    assert (a[out["date"] == pd.Timestamp("2014-12-25")] == 0).all()
    assert (a[:, 3] > a[:, 0]).mean() > 0.5               # the 95% line is above the median


# ---------------------------------------------------------------------------
# Task 5: review tools (re-check strategies 3 and 5)
# ---------------------------------------------------------------------------
def fold_scores(q50, q80, q90, q95):
    return [{"segment": "all", "pinball_q50": q50, "pinball_q80": q80, "pinball_q90": q90,
             "pinball_q95": q95}, {"segment": "slow", "pinball_q50": 9, "pinball_q80": 9,
                                   "pinball_q90": 9, "pinball_q95": 9}]


def test_compare_lines_flags_loser():
    from forecast.review import compare_lines
    model = {"f1": fold_scores(2.0, 1.7, 1.2, 0.90)}
    naive = {"f1": fold_scores(2.4, 2.0, 1.4, 0.96)}
    stub = {"f1": fold_scores(2.2, 1.8, 1.3, 0.88)}
    t = compare_lines(model, naive, stub).set_index("line")
    assert list(t.index) == ["q80", "q90", "q95"]
    assert t.loc["q95", "model"] == 0.90 and t.loc["q95", "stub"] == 0.88
    assert t["beats_naive"].all()
    assert t.loc["q80", "beats_stub"] and t.loc["q90", "beats_stub"]
    assert not t.loc["q95", "beats_stub"]


def test_worst_items_uses_relative_error(cfg):
    """A big seller with an ordinary error ranks below a small seller with a large error
    for its size; each row carries the stub's error and the store-item's traits."""
    from forecast.review import worst_items
    days = pd.date_range("2015-07-01", periods=10, freq="D")
    rows = []
    for item, units, miss in (("BIG", 100, 10), ("SMALL", 2, 3), ("OK", 5, 0.4)):
        for d in days:
            rows.append({"date": d, "store_id": "CA_1", "item_id": item, "units": units,
                         "q50": units + miss, "stub_q50": units + 1})
    d = pd.DataFrame(rows)
    traits = pd.DataFrame({"store_id": "CA_1", "item_id": ["BIG", "SMALL", "OK"],
                           "is_perishable": [False, True, False]})
    w = worst_items(d[KEY + ["q50"]], d[KEY + ["units"]], d[KEY].assign(q50=d["stub_q50"]),
                    traits, cfg, n=2)
    assert list(w["item_id"]) == ["SMALL", "BIG"]
    first = w.iloc[0]
    assert first["mae"] == pytest.approx(3) and first["mean_units"] == pytest.approx(2)
    assert first["rel_error"] == pytest.approx(1.5) and first["stub_mae"] == pytest.approx(1)
    assert bool(first["is_perishable"]) is True


def test_worst_items_skips_closed_days(cfg):
    from forecast.review import worst_items
    d = pd.DataFrame({"date": pd.to_datetime(["2014-12-24", "2014-12-25"]), "store_id": "CA_1",
                      "item_id": "A", "units": [4, 0], "q50": [4, 9]})
    traits = pd.DataFrame({"store_id": ["CA_1"], "item_id": ["A"]})
    w = worst_items(d[KEY + ["q50"]], d[KEY + ["units"]], d[KEY + ["q50"]], traits, cfg, n=5)
    assert w.iloc[0]["mae"] == 0


def test_shuffle_importance_finds_used_input():
    """y is made from column a only: scrambling a hurts, scrambling b changes ~nothing."""
    from forecast.model import fit_quantile
    from forecast.review import shuffle_importance
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(size=4000), "b": rng.normal(size=4000)})
    y = 10 + 3 * X["a"].to_numpy() + rng.normal(scale=0.3, size=4000)
    m = fit_quantile(X[:3000], y[:3000], 0.5, {**TINY_PARAMS, "max_iter": 60}, seed=1,
                     cat_mask=[False, False])
    t = shuffle_importance(m, X[3000:], y[3000:], 0.5, n_repeats=3, seed=0).set_index("input")
    assert list(t.index[:1]) == ["a"]
    assert t.loc["a", "rise"] > 1.0
    assert abs(t.loc["b", "rise"]) < 0.05


def test_leak_check_flags_too_good(cfg):
    from forecast.review import leak_check
    floor = cfg["forecast"]["review"]["leak_mae_floor"]
    fold = cfg["forecast"]["review"]["leak_fold"]
    assert leak_check({fold: floor + 0.5}, cfg) is None
    msg = leak_check({fold: floor - 0.5}, cfg)
    assert msg and "leak" in msg


def test_item_traits_columns(cfg):
    from forecast.review import item_traits
    sales = tiny_sales(cfg, days=120)
    sales.loc[(sales["item_id"] == "FOODS_3_001") & (sales["store_id"] == "CA_1")
              & (sales["date"] == START), "price"] = 0.02
    late = sales[(sales["item_id"] == "FOODS_3_002") & (sales["date"] < START + pd.Timedelta(days=30))].index
    sales = sales.drop(late)
    t = item_traits(sales, tiny_items(sales), cfg, cutoff=START + pd.Timedelta(days=100))
    assert {"store_id", "item_id", "is_perishable", "avg_daily_units", "first_day", "late_launch",
            "long_zero_runs", "odd_price_rows"} <= set(t.columns)
    t = t.set_index(["store_id", "item_id"])
    assert t.loc[("CA_1", "FOODS_3_001"), "odd_price_rows"] == 1
    assert bool(t.loc[("CA_1", "FOODS_3_002"), "late_launch"]) is True
    assert bool(t.loc[("CA_1", "FOODS_3_003"), "late_launch"]) is False


def try_result(f1, f2, seeds=(1, 2, 3), method="model_q50", leak=None):
    return {"method": method, "seeds": list(seeds), "leak_check": leak, "folds": {
        "fold1": {"mean_over_seeds": {"pinball_q50": f1, "mae": 2 * f1}},
        "fold2": {"mean_over_seeds": {"pinball_q50": f2, "mae": 2 * f2}}}}


def test_log_try_exploring_seed_never_kept(cfg, tmp_path):
    """Fix 2: a one-seed try is logged for exploring but never kept, and is never compared
    with a 3-seed row."""
    from forecast.evaluate import log_try
    cfg["forecast"]["scores_dir"] = str(tmp_path)
    log_try(cfg, try_result(2.07, 2.10), "base")
    row = log_try(cfg, try_result(2.00, 2.00, seeds=(1,)), "one seed, looks better")
    assert row["kept"] is False and "exploring" in row["why"]


def test_log_try_same_method_only(cfg, tmp_path):
    """The full model's post-processed q50 is never compared with the median-only model."""
    from forecast.evaluate import log_try
    cfg["forecast"]["scores_dir"] = str(tmp_path)
    log_try(cfg, try_result(2.07, 2.10), "base q50")
    row = log_try(cfg, try_result(2.08, 2.11, method="model"), "first full model")
    assert row["kept"] is True and row["why"] == "base row"


def test_log_try_leak_never_kept(cfg, tmp_path):
    from forecast.evaluate import log_try
    cfg["forecast"]["scores_dir"] = str(tmp_path)
    log_try(cfg, try_result(2.07, 2.10), "base")
    row = log_try(cfg, try_result(1.70, 1.75, leak="suspected leak: ..."), "too good")
    assert row["kept"] is False and "leak" in row["why"]


def test_log_try_exact_tie_after_csv(cfg, tmp_path):
    """Rerunning the same settings (identical by R12) is a tie, not a gain, even after the
    scores went through the CSV file."""
    from forecast.evaluate import log_try
    cfg["forecast"]["scores_dir"] = str(tmp_path)
    v = 2.1300918552535633                      # reads back 1 ulp higher with the default parser
    log_try(cfg, try_result(v, v), "base")
    assert log_try(cfg, try_result(v, v), "same again")["kept"] is False


def test_log_try_tolerates_rows_without_kept(cfg, tmp_path):
    from forecast.evaluate import log_try
    from forecast.experiments import log_experiment
    cfg["forecast"]["scores_dir"] = str(tmp_path)
    log_try(cfg, try_result(2.07, 2.10), "base")
    log_experiment(tmp_path / "experiments.csv", {"change": "a note with no kept value"})
    assert log_try(cfg, try_result(2.06, 2.09), "better")["kept"] is True


def test_main_exits_on_suspected_leak(cfg, tmp_path, monkeypatch):
    """A suspected leak stops the run with an error code after writing what was found."""
    from forecast import evaluate as E
    monkeypatch.setattr(E, "load_config", lambda path: {**cfg, "forecast": {**cfg["forecast"],
                                                                            "scores_dir": str(tmp_path)}})
    monkeypatch.setattr(E, "evaluate_model", lambda c, m, s: {
        **try_result(1.7, 1.75, leak="suspected leak: test"),
        "folds": {"fold1": {"mean_over_seeds": {"pinball_q50": 1.7, "mae": 3.4},
                            "seed_runs": []}}})
    with pytest.raises(SystemExit) as e:
        E.main(["--method", "model_q50", "--log", "leaky"])
    assert e.value.code != 0
    assert (tmp_path / "model_q50_val.json").exists()


# ---------------------------------------------------------------------------
# Stopped-selling fix C+A (user's call 2026-10-31, decision 26)
# ---------------------------------------------------------------------------
def test_config_keeps_long_zero_runs_and_dead_item_rule(cfg):
    """C: long zero runs stay in training. A: the dead-item rule is on."""
    assert cfg["forecast"]["drop_long_zero_runs"] is False
    assert cfg["forecast"]["dead_item_q50"] is True


def test_dead_item_rule_zeroes_q50_only(cfg):
    """q50 = 0 where the 28-day average ending 7 days back is 0; the upper lines, rows that
    sold, and rows with too little history (NaN) are left alone."""
    from forecast.postprocess import dead_item_rule
    lines = lines_frame([[3, 5, 6, 8], [4, 6, 7, 9], [2, 3, 4, 5]])
    out = dead_item_rule(lines, np.array([0.0, 1.5, np.nan]), cfg)
    assert out[LINES].values.tolist() == [[0, 5, 6, 8], [4, 6, 7, 9], [2, 3, 4, 5]]
    assert lines["q50"].iloc[0] == 3                      # input not changed in place
    cfg["forecast"]["dead_item_q50"] = False
    off = dead_item_rule(lines, np.array([0.0, 1.5, np.nan]), cfg)
    pd.testing.assert_frame_equal(off, lines)


def test_predict_lines_applies_dead_item_rule(cfg):
    """An item with no sales in the 28 days ending 7 days back gets q50 = 0 from the model."""
    from forecast.model import fit_models, predict_lines
    sales = tiny_sales(cfg)
    dead = (sales["store_id"] == "CA_1") & (sales["item_id"] == "FOODS_3_004")
    sales.loc[dead & (sales["date"] >= CUTOFF - pd.Timedelta(days=40)), "units"] = 0
    bundle = fit_models(sales[sales["date"] < CUTOFF], CUTOFF, START, cfg, [0.5, 0.95], TINY_PARAMS)
    targets = targets_between(sales, CUTOFF, CUTOFF + pd.Timedelta(days=6))
    lines = predict_lines(bundle, sales, targets, cfg)
    is_dead = ((lines["store_id"] == "CA_1") & (lines["item_id"] == "FOODS_3_004")).to_numpy()
    assert (lines.loc[is_dead, "q50"] == 0).all()
    assert (lines.loc[~is_dead, "q50"] > 0).all()


# ---------------------------------------------------------------------------
# Phase D alignment (user's call 2026-10-31, decision 28): per-line settings, 600-tree grid
# ---------------------------------------------------------------------------
def test_model_params_per_line(cfg):
    """forecast.line_params overrides forecast.params for one line only."""
    from forecast.model import model_params
    cfg["forecast"]["line_params"] = {"q95": {"max_iter": 40}}
    p = model_params(cfg)
    assert set(p) == {0.5, 0.8, 0.9, 0.95}
    assert p[0.95]["max_iter"] == 40
    assert p[0.5] == cfg["forecast"]["params"]
    assert p[0.95]["learning_rate"] == cfg["forecast"]["params"]["learning_rate"]


def test_model_params_refuses_unknown_line(cfg):
    from forecast.model import model_params
    cfg["forecast"]["line_params"] = {"q99": {"max_iter": 40}}
    with pytest.raises(ValueError, match="q99"):
        model_params(cfg)


def test_fit_models_uses_line_params(cfg):
    """Each line's model is built with its own settings, and the bundle records them."""
    from forecast.model import fit_models
    sales = tiny_sales(cfg)
    params = {0.5: {**TINY_PARAMS, "max_iter": 10}, 0.95: {**TINY_PARAMS, "max_iter": 25}}
    b = fit_models(sales[sales["date"] < CUTOFF], CUTOFF, START, cfg, [0.5, 0.95], params)
    assert b.models[0.5].n_iter_ == 10 and b.models[0.95].n_iter_ == 25
    assert b.params == {"q50": params[0.5], "q95": params[0.95]}


def test_config_grid_reaches_600_trees(cfg):
    """The tree count is read from the curve up to 600 trees, so the grid builds 600."""
    g = cfg["forecast"]["tuning_grid"]
    assert g["max_iter"] == [600]
    assert cfg["forecast"]["line_params"] == {} or set(cfg["forecast"]["line_params"]) <= {
        "q50", "q80", "q90", "q95"}


def test_seed_runs_score_every_line(cfg):
    """The 3-seed confirm needs each line's pinball per seed and their mean."""
    from forecast.evaluate import evaluate_model_fold
    from forecast.metrics import segments
    cfg["forecast"]["params"] = TINY_PARAMS
    cfg["data"]["windows"]["train_start"] = "2015-01-01"
    sales = tiny_sales(cfg)
    fold = {"name": "f", "val_start": CUTOFF, "val_end": CUTOFF + pd.Timedelta(days=13), "peak_days": []}
    seg = segments(sales, tiny_items(sales), cfg)
    block = evaluate_model_fold(sales, fold, seg, cfg, [0.5, 0.8, 0.9, 0.95], [1, 2])
    for r in block["seed_runs"]:
        assert {"pinball_q80", "pinball_q90", "pinball_q95"} <= set(r)
    m = block["mean_over_seeds"]
    assert m["pinball_q95"] == pytest.approx(np.mean([r["pinball_q95"] for r in block["seed_runs"]]))


# ---------------------------------------------------------------------------
# Task 6: light tuning on validation only
# ---------------------------------------------------------------------------
def tune_cfg(cfg):
    cfg["forecast"]["params"] = dict(TINY_PARAMS, max_iter=100)
    cfg["forecast"]["tuning_grid"] = {"learning_rate": [0.1, 0.2], "max_leaf_nodes": [7],
                                      "max_iter": [100]}
    cfg["data"]["windows"]["train_start"] = "2015-01-01"
    cfg["forecast"]["val_folds"] = [
        {"name": "fa", "train_end": "2015-03-31", "val_start": "2015-04-01", "val_end": "2015-04-21"},
        {"name": "fb", "train_end": "2015-04-30", "val_start": "2015-05-01", "val_end": "2015-05-21"}]
    return cfg


def test_grid_size_limit(cfg):
    """At most 6 settings come from the config grid."""
    from forecast.tune import grid
    combos = grid(cfg)
    assert 1 <= len(combos) <= 6
    cfg["forecast"]["tuning_grid"] = {"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63],
                                      "max_iter": [600]}
    with pytest.raises(ValueError, match="6"):
        grid(cfg)


def test_tune_reads_no_test_rows(cfg):
    """Rows from test_start on are poisoned: the tuning table must not change."""
    from forecast.tune import tune
    cfg = tune_cfg(cfg)
    test_start = pd.Timestamp(cfg["data"]["windows"]["test_start"])
    sales = tiny_sales(cfg, days=1100)                       # runs past test_start (2015-11-01)
    assert sales["date"].max() >= test_start
    poisoned = sales.copy()
    poisoned.loc[poisoned["date"] >= test_start, "units"] = 999
    a, b = tune(sales, cfg), tune(poisoned, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.columns) >= {"learning_rate", "max_leaf_nodes", "trees", "line", "fold", "pinball"}
    assert set(a["trees"]) == {50, 100}


def tune_rows(table):
    """[(lr, trees, line, fold, pinball)] -> tuning table."""
    return pd.DataFrame([{"learning_rate": lr, "max_leaf_nodes": 31, "trees": t, "line": line,
                          "fold": fold, "pinball": p} for lr, t, line, fold, p in table])


def test_best_params_needs_both_folds():
    """The best average that loses to the current settings on one fold is not picked."""
    from forecast.tune import best_params
    current = {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 300, "min_samples_leaf": 50}
    res = tune_rows([
        (0.1, 300, "q50", "f1", 2.00), (0.1, 300, "q50", "f2", 2.00),     # current
        (0.05, 300, "q50", "f1", 1.70), (0.05, 300, "q50", "f2", 2.01),   # best mean, worse on f2
        (0.1, 100, "q50", "f1", 1.95), (0.1, 100, "q50", "f2", 1.99),     # better on both
    ])
    best = best_params(res, current)
    assert best["q50"] == {**current, "max_iter": 100}


def test_best_params_keeps_current_when_nothing_better():
    from forecast.tune import best_params
    current = {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 300, "min_samples_leaf": 50}
    res = tune_rows([
        (0.1, 300, "q95", "f1", 0.80), (0.1, 300, "q95", "f2", 0.78),
        (0.05, 300, "q95", "f1", 0.79), (0.05, 300, "q95", "f2", 0.785),
    ])
    assert best_params(res, current)["q95"] == current


def test_best_params_needs_current_in_table():
    from forecast.tune import best_params
    current = {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 300, "min_samples_leaf": 50}
    with pytest.raises(ValueError, match="current"):
        best_params(tune_rows([(0.05, 300, "q50", "f1", 2.0), (0.05, 300, "q50", "f2", 2.0)]), current)


# ---------------------------------------------------------------------------
# Task 7: calibration (scaled correction), cross-fold check (strategy 2), group check (strategy 8)
# ---------------------------------------------------------------------------
def cal_frame(q50, line, units, name="q95"):
    """Rows with a q50, one upper line, and actual units (other lines copied for ordering)."""
    n = len(units)
    d = pd.DataFrame({"date": pd.Timestamp("2015-07-01") + pd.to_timedelta(np.arange(n) % 300, "D"),
                      "store_id": "CA_1", "item_id": [f"FOODS_3_{i:05d}" for i in range(n)]})
    d["q50"] = np.asarray(q50, float)
    for c in ("q80", "q90", "q95"):
        d[c] = np.asarray(line, float)
    actual = d[KEY].assign(units=np.asarray(units, float))
    return d, actual


def test_no_shift_when_within_tolerance(cfg):
    from forecast.calibrate import fit_shifts
    rng = np.random.default_rng(0)
    y = rng.normal(10, 2, 20000)
    lines, actual = cal_frame(np.full_like(y, 10), np.full_like(y, 10 + 1.645 * 2), y)
    for c, z in (("q80", 0.84), ("q90", 1.28)):
        lines[c] = 10 + z * 2
    s = fit_shifts(lines, actual, cfg)
    assert s == {"q50": 0.0, "q80": 1.0, "q90": 1.0, "q95": 1.0}


def test_scale_fixes_coverage_on_synthetic(cfg):
    """Small and big sellers whose spread is 30% too narrow: after scaling, each group's 95%
    coverage is within 1 point of the target."""
    from forecast.calibrate import apply_shifts, fit_shifts
    rng = np.random.default_rng(1)
    n = 20000
    mu = np.where(np.arange(n) < n // 2, 2.0, 48.0)
    sd = mu * 0.3
    y = rng.normal(mu, sd)
    lines, actual = cal_frame(mu, mu + 0.7 * 1.645 * sd, y)
    lines["q80"], lines["q90"] = mu + 0.7 * 0.84 * sd, mu + 0.7 * 1.28 * sd
    s = fit_shifts(lines, actual, cfg)
    assert s["q95"] > 1.2
    out = apply_shifts(lines, s, cfg)
    for small in (True, False):
        g = (mu == 2.0) == small
        assert (y[g] <= out["q95"].to_numpy()[g]).mean() * 100 == pytest.approx(95, abs=1)


def test_scale_leaves_zero_spread_rows(cfg):
    from forecast.calibrate import apply_shifts
    lines, _ = cal_frame([3.0, 3.0], [3.0, 5.0], [0, 0])
    out = apply_shifts(lines, {"q50": 0.0, "q80": 2.0, "q90": 2.0, "q95": 2.0}, cfg)
    assert out["q95"].tolist() == [3.0, 7.0]


def test_q50_unit_shift(cfg):
    """q50 too high by 2 units everywhere: a -2 shift, applied after the upper lines are
    scaled around the original q50."""
    from forecast.calibrate import apply_shifts, fit_shifts
    rng = np.random.default_rng(2)
    y = rng.normal(10, 1, 20000)
    lines, actual = cal_frame(np.full_like(y, 12), np.full_like(y, 12 + 1.645), y)
    lines["q80"], lines["q90"] = 12 + 0.84, 12 + 1.28
    s = fit_shifts(lines, actual, cfg)
    assert s["q50"] == pytest.approx(-2, abs=0.05)
    out = apply_shifts(lines, {"q50": -2.0, "q80": 1.0, "q90": 1.0, "q95": 1.0}, cfg)
    assert out["q50"].iloc[0] == pytest.approx(10) and out["q95"].iloc[0] == pytest.approx(12 + 1.645)


def test_shifted_lines_still_ordered(cfg):
    """A big q50 shift and a k below 1 can cross lines; post-processing un-crosses them, and
    25 Dec stays 0."""
    from forecast.calibrate import apply_shifts
    lines = lines_frame([[5, 6, 7, 8], [5, 6, 7, 8]])
    lines.loc[1, "date"] = pd.Timestamp("2015-12-25")
    out = apply_shifts(lines, {"q50": 4.0, "q80": 0.5, "q90": 0.5, "q95": 0.5}, cfg)
    a = out[LINES].to_numpy()
    assert (np.diff(a[0]) >= 0).all() and (a[1] == 0).all()


def test_shifts_skip_closed_days(cfg):
    """25 Dec rows (0 sales, lines 0) are left out of the fit."""
    from forecast.calibrate import fit_shifts
    rng = np.random.default_rng(3)
    y = rng.normal(10, 2, 4000)
    lines, actual = cal_frame(np.full_like(y, 10), np.full_like(y, 10 + 1.645 * 2), y)
    lines["q80"], lines["q90"] = 10 + 0.84 * 2, 10 + 1.28 * 2
    xmas = lines.iloc[:3000].copy()
    xmas["date"] = pd.Timestamp("2014-12-25")
    xmas["item_id"] = [f"X{i}" for i in range(3000)]
    xmas[LINES] = 30.0                                    # would look over-covered
    act = pd.concat([actual, xmas[KEY].assign(units=0.0)])
    s = fit_shifts(pd.concat([lines, xmas]), act, cfg)
    assert s["q95"] == 1.0


def test_predict_lines_applies_bundle_shifts(cfg):
    """A bundle with shifts returns calibrated, post-processed lines."""
    from forecast.calibrate import apply_shifts
    from forecast.model import fit_models, predict_lines
    sales = tiny_sales(cfg)
    b = fit_models(sales[sales["date"] < CUTOFF], CUTOFF, START, cfg, [0.5, 0.8, 0.9, 0.95], TINY_PARAMS)
    targets = targets_between(sales, CUTOFF, CUTOFF + pd.Timedelta(days=6))
    raw = predict_lines(b, sales, targets, cfg)
    b.shifts = {"q50": 0.5, "q80": 1.5, "q90": 1.5, "q95": 1.5}
    pd.testing.assert_frame_equal(predict_lines(b, sales, targets, cfg), apply_shifts(raw, b.shifts, cfg))


def test_cross_fold_check_fits_on_one_judges_other(cfg):
    """Strategy 2: shifts fitted on fold A are judged on fold B (and the reverse)."""
    from forecast.calibrate import cross_fold_check, fit_shifts
    rng = np.random.default_rng(4)
    folds = {}
    for name, width in (("fa", 0.7), ("fb", 1.0)):
        y = rng.normal(10, 2, 20000)
        lines, actual = cal_frame(np.full_like(y, 10), np.full_like(y, 10 + width * 1.645 * 2), y)
        lines["q80"], lines["q90"] = 10 + width * 0.84 * 2, 10 + width * 1.28 * 2
        folds[name] = (lines, actual)
    out = cross_fold_check(folds, cfg)
    row = out[(out["fit_on"] == "fa") & (out["judged_on"] == "fb") & (out["line"] == "q95")].iloc[0]
    k = fit_shifts(*folds["fa"], cfg)["q95"]
    assert row["k"] == pytest.approx(k)
    assert row["cover_before"] == pytest.approx(95, abs=1)
    assert row["cover_after"] > 97                           # fa's wide stretch over-covers fb


def test_group_check_flags_only_both_fold_losses():
    """Strategy 8: a group is flagged only if the model loses to the stub on both folds."""
    from forecast.review import group_check
    recs = []
    for fold, (m_slow, s_slow) in (("f1", (1.2, 1.0)), ("f2", (1.1, 1.0))):
        recs += [{"fold": fold, "by": "slow", "group": "True", "line": "q95", "model": m_slow, "stub": s_slow},
                 {"fold": fold, "by": "store_id", "group": "CA_1", "line": "q95",
                  "model": 0.9 if fold == "f1" else 1.1, "stub": 1.0}]
    flagged = group_check(pd.DataFrame(recs))
    assert flagged[["by", "group", "line"]].values.tolist() == [["slow", "True", "q95"]]
