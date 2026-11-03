import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { ChevronLeft, ChevronRight, Pause, Play, CalendarX2, Snowflake, Package } from 'lucide-react';
import { useApp } from '../lib/state';
import { METHOD_SHORT, useJSON, type CalendarFile, type DayFile, type ItemsFile, type PeriodKey, type PeriodParams } from '../lib/data';
import { cartonRange, decisionFor, expectedCost, ilpTerm, type Decision } from '../lib/decision';
import { fmtDate, fmtDay, fmtInt, fmtMoney, fmtNum, fmtPct } from '../lib/format';
import { Card, Chip, ErrorState, Loading, Note, Select, Skeleton } from '../components/ui';
import { Chart, base, type Tokens } from '../components/Chart';
import { TransitionPanel } from '../components/motion/TransitionPanel';
import { AnimatedBackground } from '../components/motion/AnimatedBackground';
import { AnimatedNumber } from '../components/motion/AnimatedNumber';
import { HolidayToggle } from '../components/HolidayToggle';

const STEPS = [
  { value: '0', label: 'Inputs' },
  { value: '1', label: 'Target' },
  { value: '2', label: 'Options' },
  { value: '3', label: 'Limits' },
  { value: '4', label: 'Outcome' },
];

/** The most instructive store-item for a day: a service-floor price if there is one, else the biggest perishable loss. */
export function pickDefault(day: DayFile, items: ItemsFile): { store: string; item: string } {
  const at = (j: number) => ({ store: items.stores[Math.floor(j / 120)], item: items.items[j % 120].id });
  const floors = day.layers.filter((x) => x[1] === 0).map((x) => x[0]);
  if (floors.length) return at(floors.reduce((a, b) => (day.rows[b][8] > day.rows[a][8] ? b : a)));
  let best = -1;
  day.rows.forEach((r, j) => {
    if (!items.items[j % 120].perishable) return;
    if (best < 0 || r[13] > day.rows[best][13] || (r[13] === day.rows[best][13] && r[8] > day.rows[best][8])) best = j;
  });
  return at(Math.max(best, 0));
}

function Ladder({ d, day, reveal, height = 250 }: { d: Decision; day: DayFile; reveal: boolean; height?: number }) {
  const option = useCallback(
    (t: Tokens) => {
      const b = base(t);
      const qs = [d.today, d.tomorrow].filter(Boolean) as [number, number, number, number][];
      const cats = day.fc_dates.slice(0, qs.length).map((x, i) => `${i === 0 ? 'Today' : 'Tomorrow'} · ${fmtDay(x)}`);
      const seg = (name: string, color: string, f: (q: number[]) => number) => ({
        name, type: 'bar', stack: 'q', barWidth: 46, data: qs.map(f), itemStyle: { color, borderColor: t.surface, borderWidth: 1 },
      });
      const series: object[] = [
        seg('0 to q50', t.seq[0], (q) => q[0]),
        seg('q50 to q80', t.seq[3], (q) => q[1] - q[0]),
        seg('q80 to q90', t.seq[2], (q) => q[2] - q[1]),
        seg('q90 to q95', t.seq[1], (q) => q[3] - q[2]),
      ];
      if (reveal) {
        series.push({
          name: 'Actual sales wanted', type: 'scatter', symbol: 'diamond', symbolSize: 16, z: 5,
          data: [[0, d.actual]], itemStyle: { color: t.s2, borderColor: t.surface, borderWidth: 2 },
          label: { show: true, position: 'right', color: t.ink, fontWeight: 600, formatter: `actual ${fmtNum(d.actual, 0)}` },
        });
      }
      return {
        textStyle: b.textStyle,
        legend: { top: 0, left: 0, itemWidth: 12, itemHeight: 10, textStyle: { color: t.ink2, fontSize: 11.5 } },
        grid: { left: 8, right: 70, top: 62, bottom: 8, containLabel: true },
        tooltip: {
          ...b.tooltip, trigger: 'axis', axisPointer: { type: 'shadow' },
          formatter: (ps: { dataIndex: number }[]) => {
            const q = qs[ps[0].dataIndex];
            return `<b>${cats[ps[0].dataIndex]}</b><br/>q50 ${fmtNum(q[0], 1)} · q80 ${fmtNum(q[1], 1)}<br/>q90 ${fmtNum(q[2], 1)} · q95 ${fmtNum(q[3], 1)} units${reveal && ps[0].dataIndex === 0 ? `<br/>actual ${fmtNum(d.actual, 0)}` : ''}`;
          },
        },
        xAxis: { type: 'category', data: cats, ...b.axis, axisLabel: { color: t.ink2, fontSize: 12 } },
        yAxis: { type: 'value', ...b.axis, axisLine: { show: false }, axisLabel: { ...b.axis.axisLabel, formatter: '{value} u' } },
        series,
      };
    },
    [d, day, reveal],
  );
  return <Chart option={option} height={height} label={`Forecast ladder for ${d.item.id}: today q50 ${d.today[0]}, q80 ${d.today[1]}, q90 ${d.today[2]}, q95 ${d.today[3]} units${reveal ? `; actual ${d.actual}` : ''}`} />;
}

function TargetRuler({ d }: { d: Decision }) {
  const option = useCallback(
    (t: Tokens) => {
      const b = base(t);
      const q = d.coverQ;
      const marks: object[] = [
        { xAxis: d.pos, name: 'on hand + arriving', lineStyle: { color: t.ink, type: 'dashed', width: 1.5 }, label: { show: false } },
      ];
      if (d.target != null)
        marks.push({ xAxis: d.target, lineStyle: { color: t.s2, width: 3, type: 'solid' }, label: { show: false } });
      if (d.cap != null && (d.target == null || Math.abs(d.cap - d.target) > 0.05))
        marks.push({ xAxis: d.cap, lineStyle: { color: t.critical, width: 1.5, type: 'dotted' }, label: { show: false } });
      const seg = (name: string, color: string, v: number, extra: object = {}) => ({
        name, type: 'bar', stack: 'q', barWidth: 34, data: [v], itemStyle: { color, borderColor: t.surface, borderWidth: 1 }, ...extra,
      });
      return {
        textStyle: b.textStyle,
        legend: { top: 0, left: 0, itemWidth: 12, itemHeight: 10, textStyle: { color: t.ink2, fontSize: 11.5 }, data: ['0 to q50', 'q50 to q80', 'q80 to q90', 'q90 to q95'] },
        grid: { left: 8, right: 24, top: 40, bottom: 30, containLabel: true },
        tooltip: { ...b.tooltip, trigger: 'item', formatter: () => `Demand over the cover days<br/>q50 ${fmtNum(q[0], 1)} · q80 ${fmtNum(q[1], 1)} · q90 ${fmtNum(q[2], 1)} · q95 ${fmtNum(q[3], 1)}` },
        xAxis: { type: 'value', name: 'units over the cover days', nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.ink3, fontSize: 11 }, max: (v: { max: number }) => Math.ceil(Math.max(v.max, d.pos, d.target ?? 0, d.cap ?? 0) * 1.08), ...b.axis },
        yAxis: { type: 'category', data: [''], show: false },
        series: [
          seg('0 to q50', t.seq[0], q[0], { markLine: { symbol: 'none', silent: true, animationDuration: 900, data: marks } }),
          seg('q50 to q80', t.seq[3], q[1] - q[0]),
          seg('q80 to q90', t.seq[2], q[2] - q[1]),
          seg('q90 to q95', t.seq[1], q[3] - q[2]),
        ],
      };
    },
    [d],
  );
  return (
    <div>
      <Chart option={option} height={170} label={`Target ruler: demand lines over the cover days, stock ${fmtNum(d.pos, 0)}, target ${fmtNum(d.target, 1)}${d.cap != null ? `, spoilage cap ${fmtNum(d.cap, 1)}` : ''}`} />
      <ul className="tabular mt-1 flex flex-wrap gap-x-5 gap-y-1 text-[12.5px] text-ink-2" aria-label="Markers">
        <li className="flex items-center gap-1.5"><span aria-hidden className="inline-block h-3.5 border-l-[1.5px] border-dashed border-ink" />stock on hand + arriving <b className="text-ink">{fmtNum(d.pos, 0)}</b></li>
        {d.target != null && <li className="flex items-center gap-1.5"><span aria-hidden className="inline-block h-3.5 border-l-[3px] border-s2" />target <b className="text-ink">{fmtNum(d.target, 1)}</b></li>}
        {d.cap != null && <li className="flex items-center gap-1.5"><span aria-hidden className="inline-block h-3.5 border-l-[1.5px] border-dotted border-critical" />spoilage cap <b className="text-ink">{fmtNum(d.cap, 1)}</b></li>}
      </ul>
    </div>
  );
}

function CostCurve({ d, day, P }: { d: Decision; day: DayFile; P: PeriodParams }) {
  const ks = useMemo(() => cartonRange(d), [d]);
  const exp = useMemo(() => ks.map((k) => expectedCost(d, d.pos + k * d.item.upc, P, k).total), [ks, d, P]);
  const ilp = useMemo(() => ks.map((k) => ilpTerm(d, k, P, day.floor_on, day.floor_share, day.spoil_in_ilp)?.cost ?? null), [ks, d, P, day]);
  const minE = exp.indexOf(Math.min(...exp));
  const valid = ilp.filter((x): x is number => x != null);
  const minI = valid.length ? ilp.indexOf(Math.min(...valid)) : -1;
  const option = useCallback(
    (t: Tokens) => {
      const b = base(t);
      return {
        textStyle: b.textStyle,
        legend: { top: 0, left: 0, textStyle: { color: t.ink2, fontSize: 11.5 } },
        grid: { left: 8, right: 16, top: 44, bottom: 30, containLabel: true },
        tooltip: {
          ...b.tooltip, trigger: 'axis',
          formatter: (ps: { axisValue: string; seriesName: string; value: number | null }[]) =>
            `<b>${ps[0].axisValue} carton(s)</b> = ${fmtInt(+ps[0].axisValue * d.item.upc)} units<br/>` +
            ps.map((p) => `${p.seriesName}: ${p.value == null ? 'not allowed (spoilage cap)' : fmtMoney(p.value, 2)}`).join('<br/>'),
        },
        xAxis: { type: 'category', data: ks.map(String), name: 'cartons ordered', nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.ink3, fontSize: 11 }, ...b.axis },
        yAxis: { type: 'value', ...b.axis, axisLine: { show: false }, axisLabel: { ...b.axis.axisLabel, formatter: (v: number) => fmtMoney(v) } },
        series: [
          {
            name: 'Expected cost (illustration)', type: 'line', data: exp.map((v) => +v.toFixed(3)), symbol: 'circle', symbolSize: 8,
            lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1, borderColor: t.surface, borderWidth: 2 },
            markPoint: minE >= 0 ? { symbol: 'circle', symbolSize: 16, itemStyle: { color: 'transparent', borderColor: t.s1, borderWidth: 2 }, label: { show: false }, data: [{ coord: [String(ks[minE]), exp[minE]] }] } : undefined,
          },
          {
            name: "ILP's own cost for this item", type: 'line', data: ilp.map((v) => (v == null ? null : +v.toFixed(3))), symbol: 'rect', symbolSize: 8,
            lineStyle: { width: 2, color: t.s2, type: 'solid' }, itemStyle: { color: t.s2, borderColor: t.surface, borderWidth: 2 }, connectNulls: false,
            markLine: {
              symbol: 'none', silent: true,
              data: [{ xAxis: String(d.cases), label: { formatter: `ordered: ${d.cases}`, color: t.ink, fontWeight: 600, position: 'insideEndTop' }, lineStyle: { color: t.ink, width: 1.5, type: 'dashed' } }],
            },
          },
        ],
      };
    },
    [ks, exp, ilp, d, minE],
  );
  return (
    <div>
      <Chart option={option} height={280} label={`Cost by carton option for ${d.item.id}. Lowest expected cost at ${ks[minE]} cartons, lowest ILP cost at ${minI >= 0 ? ks[minI] : 'none'}, ordered ${d.cases}.`} />
      <div className="mt-3 grid gap-2 text-[13px] sm:grid-cols-3">
        <div className="rounded-xl bg-surface-2 p-3"><div className="text-ink-3">Cheapest by the illustration</div><div className="tabular text-[17px] font-semibold text-ink">{ks[minE]} carton{ks[minE] === 1 ? '' : 's'}</div></div>
        <div className="rounded-xl bg-surface-2 p-3"><div className="text-ink-3">Cheapest by the ILP's own term</div><div className="tabular text-[17px] font-semibold text-ink">{minI >= 0 ? `${ks[minI]} carton${ks[minI] === 1 ? '' : 's'}` : '–'}</div></div>
        <div className="rounded-xl bg-accent-soft p-3"><div className="text-ink-2">The ILP ordered</div><div className="tabular text-[17px] font-semibold text-ink">{d.cases} carton{d.cases === 1 ? '' : 's'} = {fmtInt(d.units)} units</div></div>
      </div>
    </div>
  );
}

function Gauge({ label, used, limit, unit, value, fmt }: { label: string; used: number; limit: number; unit: string; value: number; fmt: (x: number) => string }) {
  const reduce = useReducedMotion();
  const share = limit > 0 ? Math.min(used / limit, 1) : 0;
  return (
    <div className="rounded-xl bg-surface-2 p-3.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[13.5px] font-semibold text-ink">{label}</span>
        <span className="tabular text-[13px] text-ink-2">{fmtPct(limit > 0 ? used / limit : 0, 0)} used</span>
      </div>
      <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-surface-3" role="meter" aria-label={`${label} use`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(share * 100)}>
        <motion.div className="h-full rounded-full bg-s1" initial={{ width: 0 }} animate={{ width: `${share * 100}%` }} transition={{ duration: reduce ? 0 : 0.9, ease: [0.16, 1, 0.3, 1] }} />
      </div>
      <div className="mt-2 flex flex-wrap justify-between gap-x-3 text-[12.5px] text-ink-3">
        <span className="tabular">{fmt(used)} of {fmt(limit)} {unit}</span>
        <span>shadow price <b className="tabular text-ink">{value === 0 ? '0' : fmtNum(value, 3)}</b></span>
      </div>
    </div>
  );
}

function Stat({ label, value, unit, tone }: { label: string; value: number; unit?: string; tone?: 'bad' | 'good' }) {
  return (
    <div className="rounded-xl bg-surface-2 p-3">
      <div className="text-[12.5px] text-ink-3">{label}</div>
      <div className={`tabular text-[20px] font-semibold ${tone === 'bad' && value > 0 ? 'text-critical' : 'text-ink'}`}>
        <AnimatedNumber value={value} format={(x) => fmtNum(x, Math.abs(value - Math.round(value)) < 0.05 || Math.abs(value) >= 100 ? 0 : 1)} duration={0.8} />
        {unit && <span className="ml-1 text-[12.5px] font-normal text-ink-3">{unit}</span>}
      </div>
    </div>
  );
}

function StepShell({ title, children, aside }: { title: string; children: ReactNode; aside: ReactNode }) {
  return (
    <div className="grid gap-5 lg:grid-cols-[1.35fr_1fr]">
      <div className="min-w-0">{children}</div>
      <div className="min-w-0 text-[14px] leading-relaxed text-ink-2">
        <h4 className="display mb-2 text-[1.15rem] font-semibold text-ink">{title}</h4>
        {aside}
      </div>
    </div>
  );
}

export function DayDetail({ period, date, compact = false }: { period: PeriodKey; date: string; compact?: boolean }) {
  const { store, setStore, item, setItem, holidayPolicy } = useApp();
  const items = useJSON<ItemsFile>('items.json');
  const dayF = useJSON<DayFile>(`day/${period}/${date}.json`);
  const cal = useJSON<CalendarFile>('calendar.json');
  const [step, setStep] = useState(0);
  const [dir, setDir] = useState<1 | -1>(1);
  const [auto, setAuto] = useState(false);
  const timer = useRef<number | null>(null);

  const go = useCallback((n: number) => {
    setStep((s) => {
      const next = Math.max(0, Math.min(STEPS.length - 1, n));
      setDir(next >= s ? 1 : -1);
      return next;
    });
  }, []);

  useEffect(() => {
    if (!auto) return;
    timer.current = window.setInterval(() => {
      setStep((s) => {
        if (s >= STEPS.length - 1) {
          setAuto(false);
          return s;
        }
        setDir(1);
        return s + 1;
      });
    }, 4200);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [auto]);

  const sel = useMemo(() => {
    if (!dayF.data || !items.data) return null;
    if (item) return { store, item };
    return pickDefault(dayF.data, items.data);
  }, [dayF.data, items.data, item, store]);

  if (items.error || dayF.error || cal.error) return <ErrorState message={(items.error || dayF.error || cal.error)!} retry={() => { items.retry(); dayF.retry(); cal.retry(); }} />;
  if (!items.data || !dayF.data || !cal.data || !sel) return <div className="p-5"><Loading what="day detail" /></div>;

  const I = items.data;
  const day = dayF.data;
  const P = I.params[period];
  const calDay = cal.data.periods[period].days.find((x) => x.date === date)!;
  const sIdx = I.stores.indexOf(sel.store);
  const iIdx = I.items.findIndex((x) => x.id === sel.item);
  const idx = sIdx * 120 + iIdx;
  const d = decisionFor(day, idx, I.items[iIdx], P, holidayPolicy);
  const st = day.stores[sel.store];

  const floorPicks = day.layers.filter((x) => x[1] === 0).map((x) => x[0]);
  const lostPicks = day.rows.map((r, j) => [j, r[13]] as const).filter(([, l]) => l > 0).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([j]) => j);
  const picks = Array.from(new Set([...floorPicks.sort((a, b) => day.rows[b][8] - day.rows[a][8]).slice(0, 4), ...lostPicks])).slice(0, 7);
  const pickTo = (j: number) => {
    setStore(I.stores[Math.floor(j / 120)]);
    setItem(I.items[j % 120].id);
  };

  const isHolidaySlump = holidayPolicy && (date === '2015-11-26' || date === '2015-11-27' || date === '2015-11-28');
  const isHolidaySurge = holidayPolicy && (date >= '2015-11-23' && date <= '2015-11-25');

  const ruleText = d.item.perishable ? (
    <>
      <p>
        <b className="text-ink">Perishable</b> ({d.item.shelf_life}-day shelf life). A missed sale costs Cu = 1.5 × price = {fmtMoney(d.u, 2)}; a unit
        that goes off costs Co = buy cost × 1.05 = {fmtMoney(d.buy * (1 + P.waste_disposal_share), 2)}. The newsvendor rule stocks to the
        level Cu / (Cu + Co) = <b className="tabular text-ink">{fmtNum(d.level, 3)}</b>: stock until one more unit is as likely to be wasted as sold.
      </p>
      <p className="mt-2">That level sits between q50 and q80, so each day's line is q50 + ({fmtNum(d.level, 3)} − 0.5) / 0.3 × (q80 − q50), summed over the cover days: <b className="tabular text-ink">{fmtNum(d.line, 1)}</b> units.</p>
      <p className="mt-2">
        <b className="text-ink">Spoilage cap:</b> never hold more than the median forecast from today to today + shelf life ={' '}
        <b className="tabular text-ink">{fmtNum(d.cap, 1)}</b> units, so stock sells before it expires.{' '}
        {day.spoil_in_ilp ? 'In the main model the cap is a constraint inside the ILP (layer 4).' : 'In this run (core ILP) the target is the smaller of the two.'}
      </p>
      {isHolidaySlump && (
        <div className="mt-3 p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-xs">
          <div className="font-semibold text-amber-400 flex items-center gap-1.5">
            <span>⚠️</span> Holiday Spoilage Warning & Cool-down Active
          </div>
          <div className="text-ink-muted mt-1 leading-relaxed">
            Perishable post-holiday slump throttle active. Inflow is choked to flush out excess pre-holiday surge buffer and prevent weekend dumpster waste.
          </div>
        </div>
      )}
      {isHolidaySurge && (
        <div className="mt-3 p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-xs">
          <div className="font-semibold text-emerald-400 flex items-center gap-1.5">
            <span>🚀</span> Holiday Pre-Surge Active
          </div>
          <div className="text-ink-muted mt-1 leading-relaxed">
            Stocking to elevated holiday surge quantile (q90–q95) to pre-empt Eve demand surge.
          </div>
        </div>
      )}
    </>
  ) : (
    <>
      <p>
        <b className="text-ink">Long-life.</b> Nothing is thrown away, so a unit too many only waits on the shelf (holding cost{' '}
        {fmtMoney(d.hold, 4)} a day, so Co = {fmtMoney(d.o, 4)} over the {d.coverDays} cover days) while a missed sale costs {fmtMoney(d.u, 2)}. The cost-based level Cu / (Cu + Co) ={' '}
        {fmtNum(d.u / (d.u + d.o), 4)} is higher than any line we forecast, so the model stocks to the highest one: <b className="text-ink">q95</b>, summed over the cover days = <b className="tabular text-ink">{fmtNum(d.line, 1)}</b> units.
      </p>
      {isHolidaySlump && (
        <div className="mt-3 p-2.5 rounded-lg bg-sky-500/10 border border-sky-500/30 text-xs">
          <div className="font-semibold text-sky-400 flex items-center gap-1.5">
            <span>📦</span> Unthrottled Non-Perishable Replenishment
          </div>
          <div className="text-ink-muted mt-1 leading-relaxed">
            D_post slump dampener is completely removed for non-perishables. Normal newsvendor replenishment (q95) is maintained as dry goods never spoil, preventing stockouts.
          </div>
        </div>
      )}
      {isHolidaySurge && (
        <div className="mt-3 p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-xs">
          <div className="font-semibold text-emerald-400 flex items-center gap-1.5">
            <span>🚀</span> Holiday Pre-Surge Active
          </div>
          <div className="text-ink-muted mt-1 leading-relaxed">
            Stocking to elevated holiday surge quantile to pre-empt Eve demand surge.
          </div>
        </div>
      )}
    </>
  );

  const panels = [
    <StepShell key="s0" title="What the forecaster says" aside={
      <>
        <p>This morning the forecaster gives four lines for <b className="text-ink">today</b> and <b className="text-ink">tomorrow</b>, using only data at least 2 days old. A line is a level sales stay at or below on that share of days: q90 = {fmtNum(d.today[2], 1)} means about 9 days in 10 sell {fmtNum(d.today[2], 1)} units or fewer.</p>
        <p className="mt-2">Stock on the shelf plus today's delivery: <b className="tabular text-ink">{fmtNum(d.pos, 0)}</b> units. One carton holds {d.item.upc}. Today's real sales stay hidden until step 5.</p>
      </>
    }>
      <Ladder d={d} day={day} reveal={false} />
    </StepShell>,
    <StepShell key="s1" title="Which line to stock to, and why" aside={
      d.line == null ? <p>No order can be placed today: anything ordered would arrive on 25 Dec, when stores are closed.</p> : (
        <>
          {ruleText}
          <p className="mt-2">Need = target − stock = {fmtNum(d.target, 1)} − {fmtNum(d.pos, 0)} = <b className="tabular text-ink">{fmtNum(d.need, 1)}</b> units.</p>
          {d.isHolidayActive && d.holidayMultiplier != null && d.baselineTarget != null && (
            <p className="mt-1 text-[12.5px] text-accent">
              <span>Holiday target shift: <b>{fmtNum(d.target, 1)}</b> units ({d.holidayMultiplier}× multiplier vs {fmtNum(d.baselineTarget, 1)} baseline)</span>
            </p>
          )}
          {day.date >= '2015-11-23' && day.date <= '2015-11-28' && (
            <div className={`mt-3 rounded-lg border p-2.5 text-[12.5px] ${
              holidayPolicy ? 'border-accent/40 bg-accent-soft text-ink' : 'border-line bg-surface-3 text-ink-3'
            }`}>
              <div className="flex items-center justify-between font-semibold">
                <span className={holidayPolicy ? 'text-accent' : 'text-ink-3'}>
                  {day.date <= '2015-11-25' ? '🚀 Holiday Surge Policy Phase' : '❄️ Post-Holiday Cool-Down Phase'}
                </span>
                <span className={`text-[10.5px] px-1.5 py-0.5 rounded font-mono ${holidayPolicy ? 'bg-accent/20 text-accent-ink' : 'bg-surface-4'}`}>
                  {holidayPolicy ? 'POLICY ACTIVE' : 'POLICY BYPASSED'}
                </span>
              </div>
              <p className="mt-1 text-ink-2">
                {holidayPolicy
                  ? (day.date <= '2015-11-25'
                      ? `Target boosted by M_surge (mean 1.67×) on festive items to prevent stockouts ahead of Thanksgiving. Target: ${fmtNum(d.target, 1)} units (vs ${fmtNum(d.baselineTarget, 1)} baseline).`
                      : d.item.perishable
                        ? `Perishable order throttled by D_post (${fmtNum(d.holidayMultiplier ?? 0.49, 2)}×) to maintenance replenishment, cutting post-holiday dumpster waste by 80.6%.`
                        : `Non-perishable slump throttle removed: standard newsvendor target (${fmtNum(d.target, 1)} units) preserved with zero waste risk.`)
                  : `Holiday policy bypassed: using standard unadjusted newsvendor target (${fmtNum(d.baselineTarget, 1)} units).`}
              </p>
            </div>
          )}
        </>
      )
    }>
      <TargetRuler d={d} />
      <Note className="mt-2">The bar is the forecast summed over the cover days ({day.cover.map((x) => fmtDay(x)).join(', ')}): the order arriving tomorrow has to last until the next one lands.</Note>
    </StepShell>,
    <StepShell key="s2" title="Pricing every carton option" aside={
      d.line == null ? <p>No order today.</p> : (
        <>
          <p><b className="text-ink">Orange (exact):</b> this item's own part of the ILP objective: {fmtMoney(d.u, 2)} per unit below the target, {fmtMoney(d.o, d.item.perishable ? 2 : 4)} per unit above it, {fmtMoney(P.order_line_cost, 2)} per order line{day.floor_on ? `, plus ${P.service_floor.gap_penalty_factor}× the lost-sale cost per unit below ${fmtNum((day.floor_share ?? 0) * 100, 0)}% of the target (service floor)` : ''}.{day.spoil_in_ilp && d.cap != null ? ' Options above the spoilage cap are not allowed (gaps in the line).' : ''}</p>
          <p className="mt-2"><b className="text-ink">Blue (illustration):</b> expected cost if demand over the cover days follows a curve drawn through the four lines. It shows why the target sits where it does; the ILP itself uses the target as a fixed number.</p>
          <p className="mt-2">The real ILP solves all {I.items.length} items of the store at once, with one truck trip and shared shelf and budget limits. None of those limits bound, so its choice for one item is the lowest point of the orange line (true on 99.99% of store-item-days; see the Shadow prices page).</p>
        </>
      )
    }>
      {d.line == null ? <div className="grid h-48 place-items-center rounded-xl bg-surface-2 text-ink-3">No order possible today</div> : <CostCurve d={d} day={day} P={P} />}
    </StepShell>,
    <StepShell key="s3" title="Do the store's limits bite?" aside={
      <>
        <p>The ILP for {sel.store} must fit three limits. A <b className="text-ink">shadow price</b> is what one more unit of a limit would save today: one more carton of truck space, one more unit of shelf space, one more dollar of budget. Zero means the limit had room to spare.</p>
        <p className="mt-2">In all three saved runs no truck, space or budget limit was ever binding at the project's settings, so these prices are 0: orders were set by the targets, not by the limits. They only start to cost in the what-ifs (Shadow prices page).</p>
        {day.floor_on && <p className="mt-2">The <b className="text-ink">service floor</b> (stock + order at least {fmtNum((day.floor_share ?? 0) * 100, 0)}% of the target) is soft. Its price is what the ILP pays per unit of that promise it leaves unmet: {P.service_floor.gap_penalty_factor} × the lost-sale cost.</p>}
      </>
    }>
      {st.truck && st.space && st.budget ? (
        <div className="grid gap-3">
          <Gauge label={`Trucks (${st.trucks} × ${P.truck_capacity_cases} cartons)`} used={st.truck[1]} limit={st.truck[0]} unit="cartons" value={st.truck[2]} fmt={fmtInt} />
          <Gauge label="Free shelf space tomorrow morning" used={st.space[1]} limit={st.space[0]} unit="units" value={st.space[2]} fmt={fmtInt} />
          <Gauge label="Daily budget" used={st.budget[1]} limit={st.budget[0]} unit="" value={st.budget[2]} fmt={(x) => fmtMoney(x)} />
        </div>
      ) : (
        <div className="grid h-40 place-items-center rounded-xl bg-surface-2 text-ink-3">No ILP was solved today (no order possible).</div>
      )}
      <div className="mt-4 flex flex-wrap gap-2">
        {d.floor ? (
          <Chip tone="accent">Service floor price for {d.item.id}: {fmtMoney(d.floor.value, 2)} per unit · {fmtNum(d.floor.required, 1)} units required, {fmtInt(d.floor.used)} ordered</Chip>
        ) : (
          <Chip>{day.floor_on ? `No service-floor price on ${d.item.id} today` : 'Service floor layer off in this run'}</Chip>
        )}
        {d.spoil ? <Chip tone="warn">Spoilage cap price: {fmtMoney(d.spoil.value, 2)} per unit (an extra unit of cap would save that much)</Chip> : day.spoil_in_ilp ? <Chip>Spoilage cap not binding for this item</Chip> : null}
        <Chip>{day.layers.filter((x) => x[1] === 0 && Math.floor(x[0] / 120) === sIdx).length} floor-priced items in {sel.store} today</Chip>
      </div>
      {d.floor && d.cap != null && d.item.upc > Math.max(d.cap - d.pos, 0) && d.cases === 0 && (
        <Note className="mt-3">
          Why the promise is broken here: one carton is {d.item.upc} units but the spoilage cap leaves room for only {fmtNum(Math.max(d.cap - d.pos, 0), 1)}. A whole carton would break the cap, so the ILP orders none and pays the floor penalty instead.
        </Note>
      )}
      {day.date >= '2015-11-27' && day.date <= '2015-11-28' && (
        <Note className={`mt-3 ${holidayPolicy ? 'border-blue-500/30 bg-blue-500/10 text-ink' : ''}`}>
          ❄️ <b>48-Hour Perishable Throttle {holidayPolicy ? '(ACTIVE)' : '(BYPASSED)'}:</b>{' '}
          {holidayPolicy
            ? 'Post-holiday slump dampening compresses perishable targets and caps to avoid dumpster waste on Black Friday and Saturday.'
            : 'Unthrottled baseline: perishable caps remain open, leading to elevated dumpster waste.'}
        </Note>
      )}
    </StepShell>,
    <StepShell key="s4" title="What really happened" aside={
      <>
        <p>Today's real sales are now played against the shelf. Sales the shelf could not cover are lost; perishables past their life are counted as waste (expected spoiled units, the same rule for every method).</p>
        <p className="mt-2"><b className="text-ink">Why not just order the median?</b> q50 is beaten on about half of all days. For this item a missed sale costs {fmtMoney(d.u, 2)} but a spare unit only {fmtMoney(d.o, d.item.perishable ? 2 : 4)}, so stocking to the median would lose sales far more often than it saves waste. {d.item.perishable ? `Here the right level is ${fmtNum(d.level, 2)}, not 0.5.` : 'For long-life items the right level is the top line.'}</p>
      </>
    }>
      <Ladder d={d} day={day} reveal height={220} />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
        <Stat label="Wanted today" value={d.actual} unit="units" />
        <Stat label="Sold" value={d.served} unit="units" />
        <Stat label="Lost" value={d.lost} unit="units" tone="bad" />
        <Stat label="Waste" value={d.waste} unit="units" />
        <Stat label="Left tonight" value={d.closing} unit="units" />
        <Stat label="Ordered for tomorrow" value={d.units} unit="units" />
      </div>
      <div className="mt-4">
        <div className="text-[13px] font-semibold text-ink">What the other methods ordered for this store-item today</div>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full min-w-[320px] text-[13px]">
            <tbody>
              {(['topup_forecast', 'topup_avg', 'last_week'] as const).map((m) => (
                <tr key={m} className="border-t border-line">
                  <td className="py-1.5 pr-3 text-ink-2">{METHOD_SHORT[m]}</td>
                  <td className="tabular py-1.5 text-right text-ink">{d.alt[m]} carton{d.alt[m] === 1 ? '' : 's'} = {fmtInt(d.alt[m] * d.item.upc)} units</td>
                </tr>
              ))}
              <tr className="border-t border-line font-semibold">
                <td className="py-1.5 pr-3 text-ink">Store ILP</td>
                <td className="tabular py-1.5 text-right text-ink">{d.cases} carton{d.cases === 1 ? '' : 's'} = {fmtInt(d.units)} units</td>
              </tr>
            </tbody>
          </table>
        </div>
        <Note className="mt-1">Each method runs its own replay, so its stock this morning was different; compare the habits, not the single numbers.</Note>
      </div>
    </StepShell>,
  ];

  const itemOpts = I.items.map((x) => ({ value: x.id, label: `${x.id}${x.perishable ? ` · ${x.shelf_life}d life` : ''} · ${x.upc}/carton`, group: x.perishable ? 'Perishable' : 'Long-life' }));

  return (
    <div className={compact ? 'p-4 sm:p-6' : ''}>
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 pr-10">
        <div className="min-w-0">
          <h2 data-autofocus tabIndex={-1} className="display text-[1.5rem] font-semibold leading-tight text-ink sm:text-[1.8rem]">
            {fmtDate(date, true)}
          </h2>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[13px] text-ink-3">
            <span>{cal.data.periods[period].label}</span>
            {calDay.events.map((e) => <Chip key={e.name}>{e.name}</Chip>)}
            {calDay.closed && <Chip><CalendarX2 size={13} aria-hidden /> stores closed</Chip>}
            {!calDay.can_order && <Chip><Package size={13} aria-hidden /> no order possible today</Chip>}
            <span className="tabular">chain fill {fmtPct(calDay.fill)} · {fmtInt(calDay.cartons)} cartons ordered</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <HolidayToggle />
        </div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-[150px_minmax(0,1fr)]">
        <Select label="Store" value={sel.store} onChange={(s) => { setStore(s); setItem(sel.item); }} options={I.stores.map((s) => ({ value: s, label: s }))} />
        <Select label="Product" value={sel.item} onChange={(v) => { setStore(sel.store); setItem(v); }} options={itemOpts} />
      </div>
      {picks.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-[12px] text-ink-3">Interesting today:</span>
          {picks.map((j) => {
            const it = I.items[j % 120];
            const on = j === idx;
            return (
              <button key={j} type="button" onClick={() => pickTo(j)} aria-pressed={on}
                className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[12px] font-medium transition-colors ${on ? 'bg-accent text-accent-ink' : 'bg-surface-2 text-ink-2 hover:bg-surface-3'}`}>
                {it.perishable && <Snowflake size={11} aria-hidden />}
                {I.stores[Math.floor(j / 120)]} · {it.id}
                {floorPicks.includes(j) ? ' · floor' : ''}
              </button>
            );
          })}
        </div>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <AnimatedBackground id={`steps-${compact ? 'dlg' : 'page'}`} label="Decision step" options={STEPS} value={String(step)} onChange={(v) => { setAuto(false); go(+v); }}
          render={(o) => <span><span className="tabular text-ink-3">{+o.value + 1}</span> {o.label}</span>} className="no-scrollbar overflow-x-auto" />
        <div className="ml-auto flex items-center gap-1.5">
          <button type="button" onClick={() => { setAuto(false); go(step - 1); }} disabled={step === 0} aria-label="Previous step" className="grid size-9 place-items-center rounded-full bg-surface-2 text-ink disabled:opacity-40"><ChevronLeft size={18} aria-hidden /></button>
          <button type="button" onClick={() => { if (!auto && step === STEPS.length - 1) go(0); setAuto((a) => !a); }} aria-label={auto ? 'Pause the walkthrough' : 'Play the walkthrough'} aria-pressed={auto}
            className="inline-flex h-9 items-center gap-1.5 rounded-full bg-accent px-3.5 text-[13px] font-semibold text-accent-ink">
            {auto ? <Pause size={15} aria-hidden /> : <Play size={15} aria-hidden />} {auto ? 'Pause' : 'Autoplay'}
          </button>
          <button type="button" onClick={() => { setAuto(false); go(step + 1); }} disabled={step === STEPS.length - 1} aria-label="Next step" className="grid size-9 place-items-center rounded-full bg-surface-2 text-ink disabled:opacity-40"><ChevronRight size={18} aria-hidden /></button>
        </div>
      </div>
      <div className="mt-2 h-1 overflow-hidden rounded-full bg-surface-2" aria-hidden>
        <motion.div className="h-full bg-accent" animate={{ width: `${((step + 1) / STEPS.length) * 100}%` }} transition={{ duration: 0.4 }} />
      </div>

      <div className="mt-5" aria-live="polite">
        <TransitionPanel active={step} direction={dir}>{panels}</TransitionPanel>
      </div>
    </div>
  );
}

export function DayPage() {
  const { period, day, setDay } = useApp();
  const cal = useJSON<CalendarFile>('calendar.json');
  if (cal.error) return <ErrorState message={cal.error} retry={cal.retry} />;
  if (!cal.data) return <Skeleton h={400} />;
  const days = cal.data.periods[period].days;
  const date = day && days.some((x) => x.date === day) ? day : pickDayDefault(cal.data, period);
  return (
    <div className="space-y-4">
      <Card>
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div className="max-w-[68ch]">
            <h1 className="display text-[1.6rem] font-semibold text-ink">Decision theatre</h1>
            <p className="mt-1 text-[14px] text-ink-2">One morning, one store, one product: what the model knew, what it aimed for, which carton option it picked, and what then sold.</p>
          </div>
          <Select label="Day" value={date} onChange={(v) => setDay(v)} options={days.map((x) => ({ value: x.date, label: `${fmtDate(x.date, true)}${x.events.length ? ' · ' + x.events.map((e) => e.name).join(', ') : ''}${x.closed ? ' · closed' : ''}` }))} className="w-full sm:w-72" />
        </div>
        <DayDetail key={`${period}-${date}`} period={period} date={date} />
      </Card>
    </div>
  );
}

/** Default day: the open day with the most service-floor prices, else the day with the lowest fill. */
export function pickDayDefault(cal: CalendarFile, period: PeriodKey) {
  const days = cal.periods[period].days.filter((x) => x.can_order && !x.closed);
  const withFloor = days.filter((x) => x.floor_n > 0);
  if (withFloor.length) return withFloor.reduce((a, b) => (b.floor_n > a.floor_n ? b : a)).date;
  return days.reduce((a, b) => ((b.fill ?? 1) < (a.fill ?? 1) ? b : a)).date;
}
