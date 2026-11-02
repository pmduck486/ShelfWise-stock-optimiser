"""The simplified store order model (method `store_model`, decided 2026-11-02).

Each morning, for every store-item, order up to a target that covers today and tomorrow
(the store cover days of replay.baselines.store_cover_window: 2 days normally, 23 Dec covers
to 25 Dec, nothing is ordered on 24 Dec). Three rules set the target:

 1. Cost-based stock level (the newsvendor rule). A missed sale costs
    Cu = lost_sale_penalty_share x price; a unit thrown away costs Co = buy_cost x
    (1 + waste_disposal_share). The best level is the forecast line at Cu / (Cu + Co):
      - long-life items: nothing is thrown away, so stock to q95 (the highest line we have);
      - perishables: Cu / (Cu + Co) is about 0.63, so stock between q50 and q80:
        q50 + (level - 0.5) / 0.3 x (q80 - q50).
 2. Spoilage cap (perishables): never hold more than the median forecast (q50) from today to
    today + shelf life, so stock is sold before it goes bad. Days after the forecast's last day
    use the mean of the forecast days.
 3. Nearest-carton rounding (perishables): the shared code rounds up to whole cartons, so the
    target is lowered by half a carton, which turns "round up" into "round to nearest".

Order = target - (stock + everything on the way), rounded to whole cartons, minimum order
applied (replay.baselines.SimplePolicy, shared with the 3 simple rules). The warehouse is not
part of this model: with replay.warehouse off it always delivers next day (store-only), and
with it on this model's warehouse target is the pooled stores' forecast over the warehouse
cover days, mean + 1.645 x pooled spread (the 2026-11-01 trial design, kept for later).
Settings were fixed before any run (CS2105/_checks/rebuild_trial/PREREGISTERED.md); nothing
here is tuned on the test window.
"""

from datetime import timedelta

import numpy as np

from .baselines import SimplePolicy, _closed

Z95 = 1.645          # pooled warehouse safety factor (95%)
Z90 = 1.2816         # q90 - q50 = 1.2816 x spread for a normal curve


def stock_level(cfg, item_id):
    """Cu / (Cu + Co) for one product: the share of days its stock should cover."""
    om = cfg['order_model']
    item = om['parameters']['items'][item_id]
    cu = om['lost_sale_penalty_share'] * item['price']
    co = item['buy_cost'] * (1 + om['waste_disposal_share'])
    return cu / (cu + co)


def forecast_days(fc, today):
    """How many days the forecast covers from today (2 for the store-only forecaster)."""
    return len({d for d in fc.index.get_level_values('date') if d >= today})


class StoreModel(SimplePolicy):
    name = 'store_model'
    uses_past_sales = True          # the shared machinery passes past sales; this model never reads them

    def _daily_line(self, row, item, level):
        """The target line for one store-item-day of the forecast."""
        if item['shelf_life_days'] is None:
            return float(row['q95'])
        q50, q80 = float(row['q50']), float(row['q80'])
        return q50 + (level - 0.5) / 0.3 * (q80 - q50)

    def store_targets(self, ctx, keys, window):
        """Targets with the nearest-carton shift (rule 3) for perishables."""
        items = ctx['cfg']['order_model']['parameters']['items']
        shift = np.array([0.5 * items[p]['units_per_case'] if items[p]['shelf_life_days'] is not None else 0.0
                          for _, p in keys])
        return self.raw_targets(ctx, keys, window) - shift

    def raw_targets(self, ctx, keys, window):
        """Rules 1 and 2 only: the units each store-item should have over the cover days."""
        line, cap = self.lines_and_caps(ctx, keys, window)
        return np.minimum(line, cap)

    def lines_and_caps(self, ctx, keys, window):
        """(rule 1 level line over the cover days, rule 2 spoilage cap; inf for long-life items)."""
        cfg, fc, today = ctx['cfg'], ctx['fc'], ctx['today']
        items = cfg['order_model']['parameters']['items']
        horizon = forecast_days(fc, today)
        line = np.zeros(len(keys))
        caps = np.full(len(keys), np.inf)
        for j, (s, p) in enumerate(keys):
            item = items[p]
            level = stock_level(cfg, p)
            total = 0.0
            for w in window:
                if _closed(w, cfg):
                    continue
                total += self._daily_line(fc.loc[(s, p, w)], item, level)
            life = item['shelf_life_days']
            if life is not None:
                q50 = [0.0 if _closed(today + timedelta(days=t), cfg)
                       else float(fc.at[(s, p, today + timedelta(days=t)), 'q50']) for t in range(horizon)]
                mean = sum(q50) / horizon
                caps[j] = sum((q50[t] if t < horizon else mean) for t in range(0, life + 1)
                              if not _closed(today + timedelta(days=t), cfg))
            line[j] = total
        return line, caps

    def warehouse_targets(self, ctx, dc_keys, window):
        """Only used with replay.warehouse on (needs a forecast covering the warehouse days)."""
        cfg, fc = ctx['cfg'], ctx['fc']
        homes = cfg['network']['home_dc']
        col = {k: i for i, k in enumerate(dc_keys)}
        mean = np.zeros(len(dc_keys))
        var = np.zeros(len(dc_keys))
        for (s, p) in ctx['store_keys']:
            i = col[(homes[s], p)]
            for w in window:
                if _closed(w, cfg):
                    continue
                row = fc.loc[(s, p, w)]
                q50, q90 = float(row['q50']), float(row['q90'])
                mean[i] += q50
                var[i] += (max(q90 - q50, 0.0) / Z90) ** 2
        return mean + Z95 * np.sqrt(var)


class StoreILP(StoreModel):
    """Lump 3 (2026-11-02): the store model's targets turned into whole-carton orders by the
    one-day store ILP (analysis/store_ilp.py) under truck, shelf-space and budget limits. The
    ILP replaces the rounding rule; its shadow prices are kept in self.shadow_rows."""
    name = 'store_ilp'

    def __init__(self):
        self.shadow_rows = []

    def decide(self, state, fc, cfg, past):
        self._state = state
        self._day = None
        decision = super().decide(state, fc, cfg, past)
        if self._day is not None:
            decision.audit.update(self._day.stats)
            decision.audit['ilp_trucks'] = int(sum(self._day.trucks.values()))
        return decision

    def store_targets(self, ctx, keys, window):
        from optimize.policy import reschedule_pipeline
        from analysis.store_ilp import solve_day
        cfg, fc, today = ctx['cfg'], ctx['fc'], ctx['today']
        items = cfg['order_model']['parameters']['items']
        pipeline, _ = reschedule_pipeline(self._state.pipeline, cfg)
        on_way = pipeline[pipeline.location_type == 'store'].groupby(['location_id', 'item_id']).units.sum()
        stock = self._state.store_stock.set_index(['store_id', 'item_id']).units
        position = {k: float(stock.get(k, 0.0)) + float(on_way.get(k, 0.0)) for k in keys}
        lay = cfg['analysis']['store_ilp'].get('layers', {})
        line, caps = self.lines_and_caps(ctx, keys, window)
        if lay.get('spoilage_cap_in_ilp'):
            target, cap = line, dict(zip(keys, caps))           # layer 4: the cap is an ILP constraint
        else:
            target, cap = np.minimum(line, caps), None          # rule 2 applied before the ILP
        need = {k: float(t) - position[k] for k, t in zip(keys, target)}
        floor_units = {k: float(min(l, c)) for k, l, c in zip(keys, line, caps)}
        today_q50 = {k: (0.0 if _closed(today, cfg) else float(fc.at[(k[0], k[1], today), 'q50'])) for k in keys}
        tomorrow = {k: position[k] - today_q50[k] for k in keys}
        open_days = sum(1 for w in window if not _closed(w, cfg))
        day = solve_day(cfg, keys, need, tomorrow, len(window), open_days,
                        position=position, cap=cap, floor_units=floor_units)
        self._day = day
        for r in day.shadow:
            self.shadow_rows.append(dict(date=today, constraint=r['constraint'], store_id=r['store_id'],
                                         item_id=r.get('item_id', ''), value=r['value'], limit=r['limit'],
                                         used=r['used']))
        # the shared code rounds (target - position) up to whole cartons: hand it exactly the ILP's cartons
        return np.array([position[k] + day.cartons[k] * items[k[1]]['units_per_case'] for k in keys])
