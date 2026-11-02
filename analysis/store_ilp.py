"""Lump 3 on the store-only model (2026-11-02): the one-day store ILP and its shadow prices.

Each morning the store model (replay/store_model.py) says how many units each store-item
should have (its target). This small integer program turns those targets into whole-carton
orders while respecting three store limits, and its LP gives the shadow price of each limit.

Variables (one day, every store s and product p that can order today):
    x[s,p]  cartons ordered (whole number)          z[s,p]  1 if an order line is placed
    y[s]    trucks sent to store s (whole number)    short[s,p], over[s,p] >= 0 units
Minimise
    sum  u_p * short[s,p] + o_p * over[s,p] + order_line_cost * z[s,p]  +  sum truck_trip_cost * y[s]
      u_p = lost_sale_penalty_share x price         (a unit below target risks a lost sale)
      o_p = buy_cost x (1 + waste_disposal_share)   (perishable: a unit above target risks waste)
          = holding_cost_per_unit_day x cover days  (long-life: it only waits on the shelf)
subject to
    target:  units_per_case_p * x[s,p] + short[s,p] - over[s,p] = need[s,p]    (need = target - stock - on the way)
    line:    min_cases * z[s,p] <= x[s,p] <= M[s,p] * z[s,p]
    truck:   sum_p x[s,p] <= truck_capacity_cases * y[s]
    space:   sum_p units_per_case_p * x[s,p] <= free shelf space of s tomorrow morning
    budget:  sum_p buy_cost_p * units_per_case_p * x[s,p] <= daily budget of s
Free shelf space = space_factor x sum of each product's shelf capacity (the order model's own
capacity rule: shelf_days x average daily sales, at least shelf_min_cases cartons) minus what will
be there tomorrow morning (stock + on the way - today's median forecast). Daily budget =
budget_factor x average daily spend x open cover days. Every number comes from config.yaml
(order_model: and analysis.store_ilp:); nothing is tuned on the test window.

Shadow prices (build roadmap, Lump 3: "fix integers, re-solve LP, read duals"): the trucks y and
the order lines z are fixed at the ILP's values, cartons x become continuous, the LP is re-solved
and each limit's dual is read. value = Rs saved today by one more unit of that limit (one more
carton of truck space, one more unit of shelf space, one more rupee of budget); 0 = the limit
does not bind.
"""

from dataclasses import dataclass, field
import math
from time import perf_counter

import pulp

LIMITS = ('truck', 'space', 'budget')


@dataclass
class DayResult:
    cartons: dict                         # (store, item) -> whole cartons
    trucks: dict                          # store -> trucks
    shadow: list = field(default_factory=list)   # rows: constraint, store_id, value
    stats: dict = field(default_factory=dict)


def unit_costs(cfg, item_id, cover_days):
    """(u, o): Rs per unit below target, Rs per unit above target."""
    om = cfg['order_model']
    it = om['parameters']['items'][item_id]
    u = om['lost_sale_penalty_share'] * it['price']
    if it['shelf_life_days'] is None:
        o = it['holding_cost_per_unit_day'] * cover_days
    else:
        o = it['buy_cost'] * (1 + om['waste_disposal_share'])
    return u, o


def shelf_capacity(cfg, store_id, item_id):
    """The order model's per-product shelf capacity in units (same rule as execute_day's overflow)."""
    om = cfg['order_model']
    it = om['parameters']['items'][item_id]
    days = om['shelf_days_perishable'] if it['shelf_life_days'] is not None else om['shelf_days']
    avg = om['parameters']['store_averages'][store_id][item_id]
    return max(days * avg, om['shelf_min_cases'] * it['units_per_case'])


def limits(cfg, keys, tomorrow_units, open_cover_days):
    """{store: {'truck': cartons per truck, 'space': free units, 'budget': Rs}}."""
    om, a = cfg['order_model'], cfg['analysis']['store_ilp']
    items, avg = om['parameters']['items'], om['parameters']['store_averages']
    out = {}
    for s in dict.fromkeys(s for s, _ in keys):
        ps = [p for ss, p in keys if ss == s]
        space = a['space_factor'] * sum(shelf_capacity(cfg, s, p) for p in ps)
        used = sum(max(tomorrow_units[(s, p)], 0.0) for p in ps)
        spend = sum(avg[s][p] * items[p]['buy_cost'] for p in ps)
        out[s] = dict(truck=float(om['truck_capacity_cases']), space=max(space - used, 0.0),
                      budget=a['budget_factor'] * spend * open_cover_days)
        sv = a.get('layers', {}).get('stock_value_cap', {})
        if sv.get('enabled'):                # layer 1 (soft stock cap): stock value tomorrow morning
            held = sum(items[p]['buy_cost'] * max(tomorrow_units[(s, p)], 0.0) for p in ps)
            out[s]['stockval'] = max(sv['cover_days'] * spend - held, 0.0)
    return out


def _build(cfg, keys, need, cover_days, lim, fixed=None, position=None, cap=None, floor_units=None):
    om = cfg['order_model']
    lay = cfg['analysis']['store_ilp'].get('layers', {})
    fl = lay.get('service_floor', {})
    items = om['parameters']['items']
    relax = fixed is not None
    m = pulp.LpProblem('store_day', pulp.LpMinimize)
    cat = pulp.LpContinuous if relax else pulp.LpInteger
    x, z, sh, ov, gap = {}, {}, {}, {}, {}
    for k in keys:
        s, p = k
        tag = f'{s}_{p}'
        x[k] = pulp.LpVariable(f'x_{tag}', lowBound=0, cat=cat)
        sh[k] = pulp.LpVariable(f'short_{tag}', lowBound=0)
        ov[k] = pulp.LpVariable(f'over_{tag}', lowBound=0)
        if relax:
            z[k] = fixed['z'][k]
        else:
            z[k] = pulp.LpVariable(f'z_{tag}', cat=pulp.LpBinary)
    stores = list(dict.fromkeys(s for s, _ in keys))
    y = {s: (fixed['y'][s] if relax else pulp.LpVariable(f'y_{s}', lowBound=0, cat=pulp.LpInteger))
         for s in stores}
    obj = []
    for k in keys:
        u, o = unit_costs(cfg, k[1], cover_days)
        obj += [u * sh[k], o * ov[k], om['order_line_cost'] * z[k]]
        if fl.get('enabled'):
            gap[k] = pulp.LpVariable(f'gap_{k[0]}_{k[1]}', lowBound=0)
            obj.append(fl['gap_penalty_factor'] * u * gap[k])
    obj += [om['truck_trip_cost'] * y[s] for s in stores]
    m += pulp.lpSum(obj)
    cons = {}
    for k in keys:
        s, p = k
        upc = items[p]['units_per_case']
        big = math.ceil(max(need[k], 0.0) / upc) + om['store_min_order_cases']
        m += upc * x[k] + sh[k] - ov[k] == need[k], f'target_{s}_{p}'
        m += x[k] <= big * z[k], f'lineup_{s}_{p}'
        m += x[k] >= om['store_min_order_cases'] * z[k], f'linelo_{s}_{p}'
        if fl.get('enabled'):                # layer 2: service floor (soft, a big penalty per missing unit)
            m += upc * x[k] + gap[k] >= fl['share'] * floor_units[k] - position[k], f'floor_{s}_{p}'
        if cap is not None and math.isfinite(cap[k]):     # layer 4: spoilage cap inside the ILP
            m += upc * x[k] <= max(cap[k] - position[k], 0.0), f'spoil_{s}_{p}'
    for s in stores:
        ks = [k for k in keys if k[0] == s]
        cons[('truck', s)] = pulp.lpSum(x[k] for k in ks) <= lim[s]['truck'] * y[s]
        cons[('space', s)] = pulp.lpSum(items[k[1]]['units_per_case'] * x[k] for k in ks) <= lim[s]['space']
        cons[('budget', s)] = pulp.lpSum(items[k[1]]['buy_cost'] * items[k[1]]['units_per_case'] * x[k]
                                         for k in ks) <= lim[s]['budget']
        if 'stockval' in lim[s]:
            cons[('stockval', s)] = pulp.lpSum(items[k[1]]['buy_cost'] * items[k[1]]['units_per_case'] * x[k]
                                               for k in ks) <= lim[s]['stockval']
        for (name, _), c in [(key, c) for key, c in cons.items() if key[1] == s]:
            m += c, f'{name}_{s}'
    return m, x, y, z


def _solver(cfg, mip=True):
    a = cfg['analysis']['store_ilp']
    return pulp.PULP_CBC_CMD(msg=False, mip=mip, timeLimit=a['time_limit_s'], gapRel=a['mip_gap'])


def solve_day(cfg, keys, need, tomorrow_units, cover_days, open_cover_days, position=None, cap=None,
              floor_units=None):
    """Whole-carton orders for one day plus each limit's shadow price.

    The stores share no limit, so each store is its own small ILP (same optimum as one big
    model, solved in about a tenth of the time)."""
    t0 = perf_counter()
    lim = limits(cfg, keys, tomorrow_units, open_cover_days)
    tol = cfg['order_model']['solver']['feasibility_tolerance']
    items = cfg['order_model']['parameters']['items']
    per_carton = dict(truck=lambda p: 1.0, space=lambda p: items[p]['units_per_case'],
                      budget=lambda p: items[p]['units_per_case'] * items[p]['buy_cost'],
                      stockval=lambda p: items[p]['units_per_case'] * items[p]['buy_cost'])
    extra = dict(position=position, cap=cap, floor_units=floor_units)
    if cfg['analysis']['store_ilp'].get('layers', {}).get('service_floor', {}).get('enabled') and floor_units is None:
        raise ValueError('service_floor needs position and floor_units')
    cartons, trucks, shadow = {}, {}, []
    n_vars = n_int = not_proven = 0
    objective = lp_objective = 0.0
    for s in dict.fromkeys(k[0] for k in keys):
        ks = [k for k in keys if k[0] == s]
        sub_need, sub_lim = {k: need[k] for k in ks}, {s: lim[s]}
        m, x, y, z = _build(cfg, ks, sub_need, cover_days, sub_lim, **extra)
        m.solve(_solver(cfg))
        if pulp.LpStatus[m.status] != 'Optimal':
            raise RuntimeError(f'Store ILP for {s} did not solve: {pulp.LpStatus[m.status]}')
        not_proven += int(getattr(m, 'sol_status', 1) != 1)
        n_vars += len(m.variables())
        n_int += sum(1 for v in m.variables() if v.cat == pulp.LpInteger)
        objective += float(pulp.value(m.objective))
        for k in ks:
            cartons[k] = int(round(x[k].varValue or 0.0))
        trucks[s] = int(round(y[s].varValue or 0.0))
        zval = {k: int(round(z[k].varValue or 0.0)) for k in ks}
        # shadow prices: fix trucks and order lines, relax cartons, re-solve the LP, read the duals
        lp, *_ = _build(cfg, ks, sub_need, cover_days, sub_lim, fixed=dict(y=trucks, z=zval), **extra)
        lp.solve(_solver(cfg, mip=False))
        if pulp.LpStatus[lp.status] != 'Optimal':
            raise RuntimeError(f'Fixed-integer LP for {s} did not solve')
        lp_objective += float(pulp.value(lp.objective))
        for name in [n for n in LIMITS + ('stockval',) if n in sub_lim[s]]:
            pi = lp.constraints[f'{name}_{s}'].pi
            value = 0.0 if pi is None or abs(pi) < tol else -float(pi)
            used = sum(per_carton[name](k[1]) * cartons[k] for k in ks)
            limit = lim[s][name] * trucks[s] if name == 'truck' else lim[s][name]
            shadow.append(dict(constraint=name, store_id=s, value=value, limit=float(limit), used=float(used)))
        # per-item layers: one row per store-item where the constraint has a price
        for k in ks:
            for name, sign in (('floor', 1.0), ('spoil', -1.0)):
                c = lp.constraints.get(f'{name}_{k[0]}_{k[1]}')
                if c is None or c.pi is None or abs(c.pi) < tol:
                    continue
                shadow.append(dict(constraint=name, store_id=s, item_id=k[1], value=sign * float(c.pi),
                                   limit=float(-c.constant), used=float(items[k[1]]['units_per_case'] * cartons[k])))
    stats = dict(ilp_n_vars=n_vars, ilp_n_int=n_int, ilp_status='Optimal', ilp_not_proven=not_proven,
                 ilp_objective=objective, ilp_lp_objective=lp_objective, ilp_seconds=perf_counter() - t0,
                 ilp_binding=sum(1 for r in shadow if r['value'] > 0 and r['constraint'] in LIMITS + ('stockval',)),
                 ilp_item_layer_rows=sum(1 for r in shadow if r['constraint'] in ('floor', 'spoil')))
    return DayResult(cartons, trucks, shadow, stats)
