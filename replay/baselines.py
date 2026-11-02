"""The 3 simple ordering rules (plan/l4-replay-plan.md, methods 1-3).

Each rule returns the order model's own `Decision` so that her
`optimize.replay.execute_day` executes it with the same physics and accounting
as the optimizer. On day d a rule sees only: the State (stock + pipeline), the
forecast rows with origin d, the compiled cfg and a `PastSales` holding sales
strictly before d. Every number comes from cfg (`replay:` and `order_model:`).

Shared steps (all 3 rules):
 1. store target over the store cover days (today .. day before the next order that
    can arrive; 0 demand on closed days; no order at all if it would arrive on a
    closed store day, so 24 Dec orders nothing and 23 Dec covers to 25 Dec);
 2. order = target - (stock + everything on the way), never below 0, rounded up to
    whole cartons, minimum order applied;
 3. the warehouse sends at most what it holds this morning (stock + today's arrivals),
    split in proportion to the requests in whole cartons, largest remainders first;
 4. trucks = ceil(cartons for the store / truck capacity);
 5. warehouse target over lead + review days from today; position counted after
    today's dispatch; rounded up to whole pallets, minimum applied;
 6. supplier order scaled down to the warehouse budget (whole pallets, floor, then the
    leftover budget by largest remainder).
With replay.warehouse off (store-only, 2026-11-02) steps 3, 5 and 6 are skipped: the
warehouse always holds enough, and no supplier orders are placed.
"""

from datetime import timedelta
import math

import numpy as np
import pandas as pd

from optimize import Decision, OptimizerPolicy
from optimize.policy import reschedule_pipeline


def warehouse_on(cfg):
    """replay.warehouse: off = store-only (the warehouse always delivers, no supplier orders)."""
    value = cfg['replay'].get('warehouse', 'on')
    if value in (False, 'off'):
        return False
    if value in (True, 'on'):
        return True
    raise ValueError(f"replay.warehouse must be on or off, not {value!r}")


def _closed(day, cfg, kind='store'):
    key = 'closed_store_month_days' if kind == 'store' else 'closed_dc_month_days'
    return day.strftime('%m-%d') in set(cfg['order_model']['calendar'][key])


def store_cover_window(today, cfg):
    """(can_order_today, cover days). Cover = today .. the day before the next order
    that can arrive lands (orders placed every store_review_days)."""
    lead = cfg['order_model']['lead_time_store_days']
    review = cfg['replay']['store_review_days']
    can_order = not _closed(today + timedelta(days=lead), cfg)
    for k in range(review, 367 + review, review):
        arrival = today + timedelta(days=k + lead)
        if not _closed(arrival, cfg):
            break
    else:
        raise ValueError('No open store arrival day within a year')
    return can_order, [today + timedelta(days=t) for t in range((arrival - today).days)]


def warehouse_cover_window(today, cfg):
    """(can_order_today, cover days) = today .. today + supplier lead + review - 1."""
    lead = cfg['order_model']['lead_time_supplier_days']
    n = lead + cfg['replay']['warehouse_review_days']
    can_order = not _closed(today + timedelta(days=lead), cfg, 'dc')
    return can_order, [today + timedelta(days=t) for t in range(n)]


def split_whole_cartons(requests, available):
    """Split `available` whole cartons in proportion to `requests` (largest
    remainders first, ties to the earlier store). Never above a request."""
    requests = [int(r) for r in requests]
    available = int(available)
    total = sum(requests)
    if total <= available:
        return requests
    if available <= 0:
        return [0] * len(requests)
    shares = [available * r / total for r in requests]
    base = [math.floor(s) for s in shares]
    left = available - sum(base)
    order = sorted(range(len(requests)), key=lambda i: (-(shares[i] - base[i]), i))
    for i in order[:left]:
        base[i] += 1
    return base


def scale_to_budget(pallets, pallet_costs, budget):
    """Scale every line by budget / spend, floor to whole pallets, then hand the
    leftover budget out one pallet per line by largest remainder (ties: earlier
    line), skipping a line whose pallet no longer fits. Never above a request or
    the budget. Returns (pallets, factor).

    Floor alone is degenerate with big pallets: when every line wants 1 pallet
    and the factor is below 1, all lines floor to 0 every day (Gate 1 found it)."""
    pallets = [int(p) for p in pallets]
    spend = sum(p * c for p, c in zip(pallets, pallet_costs))
    if spend <= budget:
        return pallets, 1.0
    factor = budget / spend
    shares = [p * factor for p in pallets]
    scaled = [math.floor(s) for s in shares]
    while sum(p * c for p, c in zip(scaled, pallet_costs)) > budget:      # floating-point guard
        i = max(range(len(scaled)), key=lambda j: scaled[j] * pallet_costs[j])
        scaled[i] -= 1
    left = budget - sum(p * c for p, c in zip(scaled, pallet_costs))
    for i in sorted(range(len(pallets)), key=lambda j: (-(shares[j] - math.floor(shares[j])), j)):
        if scaled[i] < pallets[i] and pallet_costs[i] <= left:
            scaled[i] += 1
            left -= pallet_costs[i]
    return scaled, factor


def _round_up(need, pack, minimum):
    count = np.ceil(np.maximum(need, 0) / pack - 1e-9).astype(int)
    count = np.maximum(count, 0)
    return np.where((count > 0) & (count < minimum), minimum, count)


class SimplePolicy:
    """Shared order-up-to machinery; subclasses give the demand signal."""
    name = None
    uses_past_sales = True

    # --- demand signals: arrays over the given keys -------------------------------
    def store_targets(self, ctx, keys, window):
        raise NotImplementedError

    def warehouse_targets(self, ctx, dc_keys, window):
        raise NotImplementedError

    # --- shared ----------------------------------------------------------------------
    def decide(self, state, fc, cfg, past):
        today = pd.Timestamp(state.date).date()
        if past.before != today:
            raise ValueError('Past sales must end the day before the decision day')
        om, rp = cfg['order_model'], cfg['replay']
        if rp['rounding'] != 'up':
            raise ValueError(f"Unsupported rounding {rp['rounding']!r}")
        if rp['shortage_split'] != 'proportional_largest_remainder' or rp['budget_rule'] != 'scale_down_largest_remainder':
            raise ValueError('Unsupported shortage or budget rule')
        items = om['parameters']['items']
        homes = cfg['network']['home_dc']
        pipeline, adjustments = reschedule_pipeline(state.pipeline, cfg)
        fc = fc.loc[pd.to_datetime(fc.origin).dt.date == today].copy()
        fc['date'] = pd.to_datetime(fc.date).dt.date
        ctx = dict(today=today, cfg=cfg, past=past, fc=fc.set_index(['store_id', 'item_id', 'date']))

        stores = state.store_stock[['store_id', 'item_id', 'units']].reset_index(drop=True)
        keys = list(zip(stores.store_id, stores.item_id))
        ctx['store_keys'] = keys
        on_way = pipeline[pipeline.location_type == 'store'].groupby(['location_id', 'item_id']).units.sum()
        position = stores.units.to_numpy(float) + np.array([float(on_way.get(k, 0.0)) for k in keys])
        upc = np.array([items[p]['units_per_case'] for _, p in keys], dtype=float)
        can_order, window = store_cover_window(today, cfg)
        if can_order:
            target = self.store_targets(ctx, keys, window)
            cases = _round_up(target - position, upc, om['store_min_order_cases'])
        else:
            cases = np.zeros(len(keys), dtype=int)

        # Warehouse dispatch limit: what it physically holds this morning.
        dcs = state.dc_stock[['dc_id', 'item_id', 'units']].reset_index(drop=True)
        dc_keys = list(zip(dcs.dc_id, dcs.item_id))
        dc_pipe = pipeline[pipeline.location_type == 'dc']
        today_in = dc_pipe[dc_pipe.arrival_date == today].groupby(['location_id', 'item_id']).units.sum()
        later_in = dc_pipe[dc_pipe.arrival_date > today].groupby(['location_id', 'item_id']).units.sum()
        morning = {k: float(u) + float(today_in.get(k, 0.0)) for k, u in zip(dc_keys, dcs.units)}
        short_lines = 0
        store_only = not warehouse_on(cfg)
        for (d, p) in ([] if store_only else dc_keys):    # store-only: the warehouse never runs short
            idx = [i for i, (s, q) in enumerate(keys) if q == p and homes[s] == d]
            asked = [int(cases[i]) for i in idx]
            have = int(math.floor(morning[(d, p)] / items[p]['units_per_case'] + 1e-9))
            if sum(asked) > have:
                short_lines += 1
                given = split_whole_cartons(asked, have)
                given = [g if g == 0 or g >= om['store_min_order_cases'] else 0 for g in given]
                for i, g in zip(idx, given):
                    cases[i] = g
        cases = cases.astype(float)
        units = cases * upc
        store_orders = pd.DataFrame(dict(date=today, store_id=stores.store_id, item_id=stores.item_id,
                                         cases=cases, units=units))
        per_store = store_orders.groupby('store_id').cases.sum()
        cap = om['truck_capacity_cases']
        store_ids = list(dict.fromkeys(stores.store_id))
        deliveries = pd.DataFrame([dict(date=today, store_id=s, dc_id=homes[s], cases_total=float(per_store[s]),
                                        trucks=float(math.ceil(per_store[s] / cap))) for s in store_ids])

        # Warehouse (supplier) order, position after today's dispatch.
        sent = store_orders.assign(dc_id=store_orders.store_id.map(homes)).groupby(['dc_id', 'item_id']).units.sum()
        dc_position = np.array([morning[k] - float(sent.get(k, 0.0)) + float(later_in.get(k, 0.0)) for k in dc_keys])
        pallet_units = np.array([items[p]['units_per_case'] * items[p]['cases_per_pallet'] for _, p in dc_keys], float)
        dc_can, dc_window = warehouse_cover_window(today, cfg)
        if dc_can and not store_only:
            dc_target = self.warehouse_targets(ctx, dc_keys, dc_window)
            pallets = _round_up(dc_target - dc_position, pallet_units, om['dc_min_order_pallets'])
        else:
            pallets = np.zeros(len(dc_keys), dtype=int)
        factors = {}
        for d in dict.fromkeys(dcs.dc_id):
            idx = [i for i, (dd, _) in enumerate(dc_keys) if dd == d]
            costs = [pallet_units[i] * items[dc_keys[i][1]]['buy_cost'] for i in idx]
            scaled, factor = scale_to_budget([pallets[i] for i in idx], costs, om['parameters']['dc_budgets'][d])
            scaled = [x if x == 0 or x >= om['dc_min_order_pallets'] else 0 for x in scaled]
            factors[d] = factor
            for i, x in zip(idx, scaled):
                pallets[i] = x
        pallets = pallets.astype(float)
        dc_orders = pd.DataFrame(dict(date=today, dc_id=dcs.dc_id, item_id=dcs.item_id,
                                      pallets=pallets, units=pallets * pallet_units))
        audit = dict(passed=True, physical_sales_checks=0, method=self.name, store_can_order=can_order,
                     store_cover_days=len(window), dc_short_lines=short_lines, budget_factor=factors,
                     budget_scaled=any(f < 1.0 for f in factors.values()))
        return Decision(today, store_orders, dc_orders, deliveries, pipeline, adjustments, None, audit)


def _open_mask(window, cfg):
    return np.array([0.0 if _closed(w, cfg) else 1.0 for w in window])


def _by_dc(store_values, keys, dc_keys, cfg):
    """Sum a (rows x store-keys) array into (rows x dc-keys) over each DC's stores."""
    homes = cfg['network']['home_dc']
    col = {k: j for j, k in enumerate(dc_keys)}
    out = np.zeros((store_values.shape[0], len(dc_keys)))
    for i, (s, p) in enumerate(keys):
        out[:, col[(homes[s], p)]] += store_values[:, i]
    return out


class LastWeek(SimplePolicy):
    """Method 1 (habit): a day's demand = recorded sales lag days earlier (0 on closed days)."""
    name = 'last_week'

    def _daily(self, ctx, keys, window):
        lag = ctx['cfg']['replay']['last_week_lag_days']
        past = ctx['past'].matrix([w - timedelta(days=lag) for w in window], keys)
        return past * _open_mask(window, ctx['cfg'])[:, None]

    def store_targets(self, ctx, keys, window):
        return self._daily(ctx, keys, window).sum(axis=0)

    def warehouse_targets(self, ctx, dc_keys, window):
        keys = ctx['store_keys']
        return _by_dc(self._daily(ctx, keys, window), keys, dc_keys, ctx['cfg']).sum(axis=0)


class TopupAverage(SimplePolicy):
    """Method 2: mu*D + z*sigma*sqrt(D), mu/sigma from the window ending yesterday."""
    name = 'topup_avg'

    def _history(self, ctx, keys):
        rp, today = ctx['cfg']['replay'], ctx['today']
        days = [today - timedelta(days=k) for k in range(rp['average_window_days'], 0, -1)]
        if rp['exclude_closed_days_from_history']:
            days = [d for d in days if not _closed(d, ctx['cfg'])]
        return ctx['past'].matrix(days, keys)

    def _z(self, ctx, items):
        rp, params = ctx['cfg']['replay'], ctx['cfg']['order_model']['parameters']['items']
        line = {float(lvl): name for lvl, name in rp['service_line_by_level']}
        return np.array([rp['z_by_line'][line[float(params[p]['service_level'])]] for p in items])

    def _target(self, ctx, hist, items, window):
        ddof = ctx['cfg']['replay']['sigma_ddof']
        mu, sigma = hist.mean(axis=0), hist.std(axis=0, ddof=ddof)
        n = _open_mask(window, ctx['cfg']).sum()
        return mu * n + self._z(ctx, items) * sigma * math.sqrt(n)

    def store_targets(self, ctx, keys, window):
        return self._target(ctx, self._history(ctx, keys), [p for _, p in keys], window)

    def warehouse_targets(self, ctx, dc_keys, window):
        keys = ctx['store_keys']
        summed = _by_dc(self._history(ctx, keys), keys, dc_keys, ctx['cfg'])
        return self._target(ctx, summed, [p for _, p in dc_keys], window)


class TopupForecast(SimplePolicy):
    """Method 3: the product's service line (q80/q90/q95) summed over the cover days."""
    name = 'topup_forecast'

    def _daily(self, ctx, keys, window):
        rp, params = ctx['cfg']['replay'], ctx['cfg']['order_model']['parameters']['items']
        line = {float(lvl): name for lvl, name in rp['service_line_by_level']}
        fc = ctx['fc']
        out = np.zeros((len(window), len(keys)))
        for j, (s, p) in enumerate(keys):
            col = line[float(params[p]['service_level'])]
            for i, w in enumerate(window):
                if _closed(w, ctx['cfg']):
                    continue
                try:
                    out[i, j] = float(fc.at[(s, p, w), col])
                except KeyError:
                    raise ValueError(f'No forecast with origin {ctx["today"]} for {s} {p} {w}') from None
        return out

    def store_targets(self, ctx, keys, window):
        return self._daily(ctx, keys, window).sum(axis=0)

    def warehouse_targets(self, ctx, dc_keys, window):
        keys = ctx['store_keys']
        return _by_dc(self._daily(ctx, keys, window), keys, dc_keys, ctx['cfg']).sum(axis=0)


POLICIES = {'last_week': LastWeek, 'topup_avg': TopupAverage, 'topup_forecast': TopupForecast,
            'optimizer': OptimizerPolicy}


def make_policy(name):
    if name in ('store_model', 'store_ilp') and name not in POLICIES:
        from .store_model import StoreILP, StoreModel
        POLICIES.update(store_model=StoreModel, store_ilp=StoreILP)
    if name not in POLICIES:
        raise ValueError(f"Unknown method {name!r}; choose from {sorted(set(POLICIES) | {'store_model', 'store_ilp'})}")
    return POLICIES[name]()
