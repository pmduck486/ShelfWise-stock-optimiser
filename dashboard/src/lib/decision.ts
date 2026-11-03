/* One store-item's decision on one day, rebuilt in the browser from the exported numbers.
   Two cost views over carton options:
   - ilpTerm: this item's own part of the store ILP objective (analysis/store_ilp.py): exact.
   - expectedCost: a newsvendor illustration over a demand distribution interpolated from the
     forecast lines; NOT what the ILP minimises (the ILP uses the target as a fixed number). */
import type { DayFile, DayRow, ItemInfo, PeriodParams } from './data';

export interface Decision {
  idx: number;
  item: ItemInfo;
  price: number;
  buy: number;
  hold: number;
  level: number; // Cu/(Cu+Co) for perishables
  today: [number, number, number, number];
  tomorrow: [number, number, number, number] | null;
  coverOpen: boolean[]; // per cover day: is the store open (25 Dec closed)
  coverQ: [number, number, number, number]; // forecast lines summed over the open cover days
  actual: number;
  pos: number; // stock on hand + arriving today, when the order is placed
  line: number | null; // rule-1 line summed over the cover days (project code)
  cap: number | null; // spoilage cap (perishables)
  target: number | null; // what the ILP aims at
  need: number | null; // target - pos
  cases: number;
  units: number;
  lost: number;
  waste: number;
  served: number;
  closing: number;
  alt: { last_week: number; topup_avg: number; topup_forecast: number; store_model: number };
  floor: { value: number; required: number; used: number } | null;
  spoil: { value: number; limit: number; used: number } | null;
  u: number; // Rs/$ per unit below target
  o: number; // per unit above target
  coverDays: number;
  baselineTarget: number | null;
  holidayTarget: number | null;
  baselineLine: number | null;
  baselineCap: number | null;
  baselineCases: number;
  holidayCases: number | null;
  isHolidayActive: boolean;
  holidayMultiplier: number | null;
}

export function decisionFor(day: DayFile, idx: number, item: ItemInfo, P: PeriodParams, holidayPolicy = false): Decision {
  const r: DayRow = day.rows[idx];
  const ip = P.items[idx % 120];
  const today: [number, number, number, number] = [r[0], r[1], r[2], r[3]];
  const tomorrow = r[4] == null ? null : ([r[4], r[5]!, r[6]!, r[7]!] as [number, number, number, number]);
  const closedMD = new Set(P.closed_days);
  const coverOpen = day.cover.map((d) => !closedMD.has(d.slice(5)));
  const coverQ: [number, number, number, number] = [0, 0, 0, 0];
  day.cover.forEach((d, i) => {
    if (!coverOpen[i]) return;
    const k = day.fc_dates.indexOf(d);
    const q = k === 0 ? today : k === 1 ? tomorrow : null;
    if (q) for (let j = 0; j < 4; j++) coverQ[j] += q[j];
  });
  const baseLine = r[10];
  const baseCap = r[11];
  const baseTarget = baseLine == null ? null : day.spoil_in_ilp || baseCap == null ? baseLine : Math.min(baseLine, baseCap);
  const baseCases = r[12];

  const hLine = (r.length > 19 && r[19] != null) ? r[19] : null;
  const hCap = (r.length > 20 && r[20] != null) ? r[20] : null;
  const hCases = (r.length > 21 && r[21] != null) ? r[21] : null;
  const hTarget = hLine == null ? null : day.spoil_in_ilp || hCap == null ? hLine : Math.min(hLine, hCap);

  const isHolidayActive = holidayPolicy && (hLine != null || hTarget != null);
  const line = isHolidayActive && hLine != null ? hLine : baseLine;
  const cap = isHolidayActive && hCap != null ? hCap : baseCap;
  const target = isHolidayActive && hTarget != null ? hTarget : baseTarget;
  const cases = isHolidayActive && hCases != null ? hCases : baseCases;

  const pos = r[9];
  const lost = r[13];
  const waste = r[14];
  const actual = r[8];
  const served = actual - lost;
  const lay = day.layers.filter((x) => x[0] === idx);
  const fl = lay.find((x) => x[1] === 0);
  const sp = lay.find((x) => x[1] === 1);
  const u = P.lost_sale_penalty_share * ip.price;
  const o = item.perishable ? ip.buy * (1 + P.waste_disposal_share) : ip.hold * day.cover.length;
  const holidayMultiplier = (baseTarget != null && target != null && baseTarget > 0.05) ? +(target / baseTarget).toFixed(2) : null;

  return {
    idx, item, price: ip.price, buy: ip.buy, hold: ip.hold, level: ip.level, today, tomorrow, coverOpen, coverQ,
    actual, pos, line, cap, target, need: target == null ? null : target - pos,
    cases, units: cases * item.upc, lost, waste, served, closing: pos - served - waste,
    alt: { last_week: r[15], topup_avg: r[16], topup_forecast: r[17], store_model: r[18] },
    floor: fl ? { value: fl[2], required: fl[3], used: fl[4] } : null,
    spoil: sp ? { value: sp[2], limit: sp[3], used: sp[4] } : null,
    u, o, coverDays: day.cover.length,
    baselineTarget: baseTarget,
    holidayTarget: hTarget,
    baselineLine: baseLine,
    baselineCap: baseCap,
    baselineCases: baseCases,
    holidayCases: hCases,
    isHolidayActive,
    holidayMultiplier,
  };
}

/** Piecewise-linear quantile function through the forecast lines (an approximation for the picture). */
export function quantileFn(q: [number, number, number, number]) {
  const [q50, q80, q90, q95] = q;
  const lo = Math.max(0, q50 - ((q80 - q50) * 0.5) / 0.3);
  const hi = q95 + ((q95 - q90) * 0.045) / 0.05;
  const ps = [0, 0.5, 0.8, 0.9, 0.95, 0.995];
  const vs = [Math.min(lo, q50), q50, Math.max(q80, q50), Math.max(q90, q80), Math.max(q95, q90), Math.max(hi, q95)];
  return (p: number) => {
    if (p >= ps[5]) return vs[5];
    let i = 0;
    while (p > ps[i + 1]) i++;
    const t = (p - ps[i]) / (ps[i + 1] - ps[i]);
    return vs[i] + t * (vs[i + 1] - vs[i]);
  };
}

const GRID = Array.from({ length: 400 }, (_, i) => (i + 0.5) / 400 * 0.995);

/** Expected cost of holding S units over the cover days (illustration). */
export function expectedCost(d: Decision, S: number, P: PeriodParams, k: number) {
  const Q = quantileFn(d.coverQ);
  let short = 0;
  let over = 0;
  for (const p of GRID) {
    const D = Q(p);
    short += Math.max(D - S, 0);
    over += Math.max(S - D, 0);
  }
  short /= GRID.length;
  over /= GRID.length;
  return { total: d.u * short + d.o * over + (k > 0 ? P.order_line_cost : 0), short, over };
}

/** This item's own term of the ILP objective for k cartons (null = not allowed by the spoilage cap). */
export function ilpTerm(d: Decision, k: number, P: PeriodParams, floorOn: boolean, floorShare: number | null, spoilInIlp: boolean) {
  if (d.need == null || d.target == null || d.line == null) return null;
  const units = k * d.item.upc;
  if (spoilInIlp && d.cap != null && units > Math.max(d.cap - d.pos, 0) + 1e-9) return null;
  const short = Math.max(d.need - units, 0);
  const over = Math.max(units - d.need, 0);
  let cost = d.u * short + d.o * over + (k > 0 ? P.order_line_cost : 0);
  let gap = 0;
  if (floorOn && floorShare != null) {
    const floorUnits = d.cap == null ? d.line : Math.min(d.line, d.cap);
    gap = Math.max(floorShare * floorUnits - d.pos - units, 0);
    cost += (P.service_floor.gap_penalty_factor ?? 0) * d.u * gap;
  }
  return { cost, short, over, gap };
}

export function cartonRange(d: Decision) {
  const top = Math.max(d.coverQ[3] * 1.15, d.line ?? 0, d.cap ?? 0, d.pos + d.units);
  const kmax = Math.max(3, Math.ceil((top - d.pos) / d.item.upc) + 1, d.cases + 2);
  return Array.from({ length: Math.min(kmax, 40) + 1 }, (_, k) => k);
}
