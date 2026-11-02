"""Model inputs (features) for the ML forecaster: one row per target day and store-item.

Rules (plan/forecast-training-plan.md, section 3):
  R2 (changed 2026-11-02, store-only model): store data (units, prices and anything made
      from them) is at least forecast.min_lag_days (now 2) older than the target day. The
      forecaster predicts today and tomorrow, so a 2-day gap uses everything known each
      morning for tomorrow (the day an order arriving tomorrow must cover). feature_sources
      lists, for each such input, the fewest days back it reads. (Was 7 days before 2026-11-02.)
  R3: calendar inputs (CALENDAR_FEATURES) may be for the target day itself. They come from
      calendar.csv, never from sales, so days after the last sales day are covered too.
  R4: statistics (category codes, each store-item's normal price, long zero-sales runs)
      come from rows before the model's cutoff only.

Inputs are read from a dense day grid per store-item, so "7 days back" means calendar
days, not rows. Days with no row (not on sale yet) are missing, not 0. Closed days
(forecast.closed_dates, 25 Dec) are missing too, so their near-zero sales never feed a
later day's lags or averages; their prices are kept. Averages, the 28-day spread and the
store rank use the days present and need at least half the window (4 of 7, 14 of 28); a
lag that lands on a closed day stays missing (decision 21).
"""

import numpy as np
import pandas as pd

from forecast.data import load_calendar
from forecast.metrics import closed_mask

KEY = ["date", "store_id", "item_id"]
PAIR = ["store_id", "item_id"]
CODED = ["store_id", "item_id", "dept_id"]      # category -> code maps fitted on training rows

def feature_sources(cfg: dict) -> dict:
    """For each store-data input: the fewest days back it reads (rule R2). Lags read their
    own day; every window, price and rank input ends forecast.min_lag_days before the target."""
    f = cfg["forecast"]
    lag0 = int(f["min_lag_days"])
    windows = sorted(int(w) for w in f["rolling_windows"])
    long = windows[-1]
    out = {f"lag_{int(L)}": int(L) for L in f["lags"]}
    out.update({f"roll_mean_{w}": lag0 for w in windows})
    out.update({f"roll_std_{long}": lag0, f"price_{lag0}": lag0,
                f"price_chg_{int(f['price_change_weeks'])}w": lag0, "price_rel_item": lag0,
                "momentum": lag0, f"store_rank_{long}": lag0})
    return out
CALENDAR_FEATURES = ["wday", "month", "is_event", "event_name", "snap", "days_to_event",
                     "days_since_event", "next_event_name"]
ID_FEATURES = ["store_id", "item_id", "dept_id", "is_perishable"]
CATEGORICAL = {"store_id", "item_id", "dept_id", "event_name", "next_event_name"}


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def feature_names(cfg: dict) -> list:
    """Input columns in model order: store data, then calendar, then who."""
    f = cfg["forecast"]
    windows = sorted(f["rolling_windows"])
    long = windows[-1]
    store = ([f"lag_{int(L)}" for L in f["lags"]]
             + [f"roll_mean_{w}" for w in windows]
             + [f"roll_std_{long}", f"price_{int(f['min_lag_days'])}",
                f"price_chg_{int(f['price_change_weeks'])}w", "price_rel_item", "momentum",
                f"store_rank_{long}"])
    too_new = [int(L) for L in f["lags"] if int(L) < int(f["min_lag_days"])]
    if too_new:
        raise ValueError(f"lags {too_new} are newer than forecast.min_lag_days (rule R2)")
    unknown = [n for n in store if n not in feature_sources(cfg)]
    if unknown:
        raise ValueError(f"inputs {unknown} are not in feature_sources; add them with the "
                         f"fewest days back they read (rule R2)")
    return store + CALENDAR_FEATURES + ID_FEATURES


def categorical_mask(cfg: dict) -> list:
    """True for each column of feature_names that the model treats as a category."""
    return [n in CATEGORICAL for n in feature_names(cfg)]


# ---------------------------------------------------------------------------
# Statistics from training rows (R4)
# ---------------------------------------------------------------------------
def _before(sales: pd.DataFrame, cutoff) -> pd.DataFrame:
    return sales[sales["date"] < pd.Timestamp(cutoff)]


def _code_map(values: pd.Series) -> dict:
    return {v: i for i, v in enumerate(sorted(values.dropna().unique()))}


def fit_encoders(sales: pd.DataFrame, *, cutoff) -> dict:
    """Category codes and each store-item's normal price, from rows before `cutoff` (R4;
    required, so the rows used are always stated).

    Codes for store_id, item_id, dept_id and event names (shared by event_name and
    next_event_name); a value not seen before the cutoff gets no code (NaN).
    normal_price: the store-item's mean price over those rows, for price_rel_item.
    """
    train = _before(sales, cutoff)
    enc = {c: _code_map(train[c]) for c in CODED}
    events = train["event_name"].fillna("").astype(str)
    enc["event_name"] = _code_map(events[events != ""])
    enc["normal_price"] = train.groupby(PAIR)["price"].mean()
    return enc


def zero_run_stats(sales: pd.DataFrame, cfg: dict, *, cutoff) -> dict:
    """Zero-sales runs longer than forecast.long_zero_run_days, from rows before `cutoff`
    (R4; required).

    A run counts open days only: a closed day inside it neither breaks it nor adds to it.
    A day with no row (a gap) does break it. Returns {"cutoff", "min_days", "runs"} where
    runs has one row per run: store_id, item_id, start, end, open_days.
    """
    min_days = int(cfg["forecast"]["long_zero_run_days"])
    d = _before(sales, cutoff)[KEY + ["units"]]
    cols = PAIR + ["start", "end", "open_days"]
    d = d[~closed_mask(d["date"], cfg).to_numpy()].sort_values(PAIR + ["date"])
    if d.empty:
        return {"cutoff": cutoff, "min_days": min_days, "runs": pd.DataFrame(columns=cols)}

    # open-day number: consecutive open days differ by 1, even across a closed day
    days = pd.Series(pd.date_range(d["date"].min(), d["date"].max(), freq="D"))
    open_no = pd.Series(np.cumsum(~closed_mask(days, cfg).to_numpy()), index=days)
    k = open_no.reindex(d["date"]).to_numpy()
    zero = (d["units"] == 0).to_numpy()
    pair = (d["store_id"] + "|" + d["item_id"]).to_numpy()

    new_block = np.ones(len(d), dtype=bool)
    new_block[1:] = (pair[1:] != pair[:-1]) | (np.diff(k) != 1) | (zero[1:] != zero[:-1])
    block = np.cumsum(new_block)
    z = d.assign(block=block)[zero]
    runs = z.groupby("block").agg(store_id=("store_id", "first"), item_id=("item_id", "first"),
                                  start=("date", "min"), end=("date", "max"),
                                  open_days=("date", "size"))
    runs = runs[runs["open_days"] > min_days][cols].reset_index(drop=True)
    return {"cutoff": cutoff, "min_days": min_days, "runs": runs}


# ---------------------------------------------------------------------------
# Training rows
# ---------------------------------------------------------------------------
def empty_shelf_filter(rows: pd.DataFrame, sales: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Hook for layer P7 (empty-shelf correction). Returns rows unchanged while
    forecast.layers.p7_empty_shelf is off."""
    if not cfg["forecast"]["layers"]["p7_empty_shelf"]:
        return rows
    raise NotImplementedError("layer P7 (empty-shelf correction) is not built yet")


def training_rows(sales: pd.DataFrame, start, end, cfg: dict, stats: dict) -> pd.DataFrame:
    """Target rows (date, store_id, item_id, units) dated start..end for fitting a model:
    forecast.drop_dates (25 Dec) dropped, long zero runs dropped when
    forecast.drop_long_zero_runs is on, then passed through empty_shelf_filter.

    `stats` comes from zero_run_stats; its cutoff must be after `end`, so the runs were
    found on training rows only (R4).
    """
    f = cfg["forecast"]
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if stats["cutoff"] is None:
        raise ValueError("zero-run stats have no cutoff; build them with zero_run_stats(..., cutoff=)")
    if end >= pd.Timestamp(stats["cutoff"]):
        raise ValueError(f"zero-run stats were found before {pd.Timestamp(stats['cutoff']).date()}"
                         f" but training rows run to {end.date()}")
    d = sales.loc[(sales["date"] >= start) & (sales["date"] <= end), KEY + ["units"]]
    d = d[~d["date"].dt.strftime("%m-%d").isin(f["drop_dates"]).to_numpy()]

    runs = stats["runs"]
    if f["drop_long_zero_runs"] and len(runs):
        drop = pd.concat([pd.DataFrame({"date": pd.date_range(r.start, r.end, freq="D"),
                                        "store_id": r.store_id, "item_id": r.item_id})
                          for r in runs.itertuples()], ignore_index=True)
        drop["date"] = drop["date"].astype(d["date"].dtype)
        m = d.merge(drop.assign(_drop=True), on=KEY, how="left")["_drop"].fillna(False)
        d = d[~m.to_numpy(dtype=bool)]

    rows = d.sort_values(KEY).reset_index(drop=True)
    return empty_shelf_filter(rows, sales, cfg)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _day_numbers(dates) -> np.ndarray:
    """Whole days since 1970-01-01 (works for any datetime64 unit)."""
    return pd.to_datetime(pd.Series(dates)).to_numpy().astype("datetime64[D]").astype(np.int64)


def _window_sums(a: np.ndarray, w: int):
    """Count, sum and sum of squares of the days present (not NaN) in the w days ending on
    each day (axis 1). Exact when `a` holds whole numbers (units, or prices in cents), so a
    day's inputs never depend on which other days are built with it."""
    present = ~np.isnan(a)
    x = np.where(present, a, 0.0)
    zeros = np.zeros((a.shape[0], 1))
    c_n = np.concatenate([zeros, np.cumsum(present, axis=1)], axis=1)
    c_s = np.concatenate([zeros, np.cumsum(x, axis=1)], axis=1)
    c_q = np.concatenate([zeros, np.cumsum(x * x, axis=1)], axis=1)
    hi = np.arange(1, a.shape[1] + 1)
    lo = np.maximum(hi - w, 0)
    return c_n[:, hi] - c_n[:, lo], c_s[:, hi] - c_s[:, lo], c_q[:, hi] - c_q[:, lo]


def _window(a: np.ndarray, w: int, min_present: int):
    """Mean and sample std over the w days ending on each day (axis 1), using the days
    present (not NaN); NaN where fewer than min_present days are present."""
    n, s, q = _window_sums(a, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n >= min_present, s / n, np.nan)
        var = np.where((n >= min_present) & (n >= 2), (q - s * s / n) / (n - 1), np.nan)
    return mean, np.sqrt(np.clip(var, 0.0, None))


def _calendar_inputs(cfg: dict) -> pd.DataFrame:
    """Calendar-only inputs for every calendar day (R3): event distances capped at
    forecast.event_distance_cap_days, and the next event's name (NaN past the cap)."""
    cap = int(cfg["forecast"]["event_distance_cap_days"])
    cal = load_calendar(cfg).sort_values("date").reset_index(drop=True)
    pos = np.arange(len(cal))
    is_ev = cal["is_event"].to_numpy()
    last = np.maximum.accumulate(np.where(is_ev, pos, -10**9))
    nxt = np.minimum.accumulate(np.where(is_ev, pos, 10**9)[::-1])[::-1]
    to_next = nxt - pos
    out = cal[["wday", "month", "is_event", "event_name"]].copy()
    out["day"] = _day_numbers(cal["date"])
    out["days_to_event"] = np.minimum(to_next, cap)
    out["days_since_event"] = np.minimum(pos - last, cap)
    names = cal["event_name"].to_numpy()
    out["next_event_name"] = np.where(to_next <= cap, names[np.minimum(nxt, len(cal) - 1)], "")
    for st in ("CA", "TX", "WI"):
        out[f"snap_{st}"] = cal[f"snap_{st}"].to_numpy()
    return out


def build_features(sales: pd.DataFrame, targets: pd.DataFrame, cfg: dict,
                   encoders: dict) -> pd.DataFrame:
    """One row of inputs per target row (date, store_id, item_id), in the targets' order,
    columns = feature_names(cfg), all float32. Store data is read only from days at least
    forecast.min_lag_days before each target (R2)."""
    f = cfg["forecast"]
    names = feature_names(cfg)
    lag0 = int(f["min_lag_days"])
    windows = sorted(int(w) for w in f["rolling_windows"])
    long = windows[-1]
    price_days = 7 * int(f["price_change_weeks"])

    tg = targets[KEY].reset_index(drop=True)
    t_day = _day_numbers(tg["date"])
    reach = max(max(int(L) for L in f["lags"]), lag0 + long - 1, lag0 + price_days)
    g0, g1 = t_day.min() - reach, t_day.max() - lag0          # g1 = newest day ever read
    n_days = g1 - g0 + 1

    # dense grid [store-item, day]; rows newer than g1 are never placed in it
    pairs = (pd.concat([sales[PAIR], tg[PAIR]]).drop_duplicates()
             .sort_values(PAIR).reset_index(drop=True))
    pair_no = pd.Series(np.arange(len(pairs)),
                        index=pd.MultiIndex.from_frame(pairs))
    s_day = _day_numbers(sales["date"])
    keep = (s_day >= g0) & (s_day <= g1)
    rows = pair_no.reindex(pd.MultiIndex.from_frame(sales.loc[keep, PAIR])).to_numpy()
    cols = s_day[keep] - g0
    units = np.full((len(pairs), n_days), np.nan)
    price = np.full((len(pairs), n_days), np.nan)
    units[rows, cols] = sales.loc[keep, "units"].to_numpy(dtype=float)
    price[rows, cols] = sales.loc[keep, "price"].to_numpy(dtype=float)
    grid_days = pd.Series(pd.to_datetime(np.arange(g0, g1 + 1), unit="D"))
    units[:, closed_mask(grid_days, cfg).to_numpy()] = np.nan    # closed: missing, price kept

    r = pair_no.reindex(pd.MultiIndex.from_frame(tg[PAIR])).to_numpy()
    t = t_day - g0
    end = t - lag0                                             # last day any window reads
    out = {}

    for L in f["lags"]:
        out[f"lag_{int(L)}"] = units[r, t - int(L)]
    means = {}
    for w in windows:
        mean, std = _window(units, w, (w + 1) // 2)
        means[w] = mean[r, end]
        out[f"roll_mean_{w}"] = means[w]
        if w == long:
            out[f"roll_std_{w}"] = std[r, end]
            long_mean = mean

    p_now = price[r, end]
    cents = np.rint(price * 100)                               # whole cents: exact sums
    n_prev, s_prev, _ = _window_sums(cents, price_days)
    n_prev, s_prev = n_prev[r, end - 1], s_prev[r, end - 1]    # the 4 weeks before end
    c_now = cents[r, end]
    normal = (encoders["normal_price"].reindex(pd.MultiIndex.from_frame(tg[PAIR]))
              .to_numpy(dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        ok_chg = (c_now > 0) & (s_prev > 0)
        ok_rel = (p_now > 0) & (normal > 0)
        out[f"price_{lag0}"] = p_now
        # price at end vs the mean of the 4 weeks before: (now - mean) / mean, in cents
        out[f"price_chg_{int(f['price_change_weeks'])}w"] = np.where(
            ok_chg, (c_now * n_prev - s_prev) / s_prev, 0.0)
        out["price_rel_item"] = np.where(ok_rel, p_now / normal, 1.0)
        short_m, long_m = means[windows[0]], means[long]
        out["momentum"] = np.where(np.isnan(short_m) | np.isnan(long_m), np.nan,
                                   np.where(long_m > 0, short_m / long_m, 1.0))

    # sales rank within the store over the long window: 0-1 share, 1 = top seller
    rank = np.full_like(long_mean, np.nan)
    for _, idx in pairs.groupby("store_id").indices.items():
        rank[idx] = pd.DataFrame(long_mean[idx]).rank(axis=0, pct=True).to_numpy()
    out[f"store_rank_{long}"] = rank[r, end]

    # calendar (R3)
    cal = _calendar_inputs(cfg).set_index("day")
    missing = ~np.isin(t_day, cal.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} target days are not in calendar.csv")
    c = cal.loc[t_day]
    ev = encoders["event_name"]
    out["wday"] = c["wday"].to_numpy()
    out["month"] = c["month"].to_numpy()
    out["is_event"] = c["is_event"].to_numpy()
    out["event_name"] = c["event_name"].map(ev).to_numpy(dtype=float)
    state = tg["store_id"].str[:2].to_numpy()
    snap = np.zeros(len(tg), dtype=bool)
    for st in ("CA", "TX", "WI"):
        snap[state == st] = c[f"snap_{st}"].to_numpy()[state == st]
    out["snap"] = snap
    out["days_to_event"] = c["days_to_event"].to_numpy()
    out["days_since_event"] = c["days_since_event"].to_numpy()
    out["next_event_name"] = c["next_event_name"].map(ev).to_numpy(dtype=float)

    # who
    item_info = sales.drop_duplicates("item_id").set_index("item_id")
    for col in ("store_id", "item_id"):
        out[col] = tg[col].map(encoders[col]).to_numpy(dtype=float)
    dept = tg["item_id"].map(item_info["dept_id"])
    out["dept_id"] = dept.map(encoders["dept_id"]).to_numpy(dtype=float)
    out["is_perishable"] = tg["item_id"].map(item_info["is_perishable"]).to_numpy(dtype=float)

    return pd.DataFrame({n: np.asarray(out[n], dtype=np.float32) for n in names})
