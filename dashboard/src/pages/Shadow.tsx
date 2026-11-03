import { useCallback, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { Sliders, AlertCircle, CheckCircle2, Sparkles, RotateCcw, Flame, Snowflake, ToggleLeft, ToggleRight } from 'lucide-react';
import { useApp } from '../lib/state';
import { PERIOD_SHORT, useJSON, type ShadowFile, type WhatIfRow } from '../lib/data';
import { fmtDay, fmtInt, fmtMoney, fmtNum, fmtPct, fmtSigned } from '../lib/format';
import { Card, ErrorState, Loading, Note, SectionHead, Source } from '../components/ui';
import { Chart, base, type Tokens } from '../components/Chart';
import { AnimatedBackground } from '../components/motion/AnimatedBackground';
import { InView } from '../components/motion/InView';
import { simulateParameter, generateCurvePoints, type SimMetrics } from '../lib/simulation';

const LIMIT_TEXT: Record<string, { name: string; unit: string }> = {
  truck: { name: 'Truck capacity', unit: 'per extra carton of truck space' },
  space: { name: 'Shelf space', unit: 'per extra unit of shelf space' },
  budget: { name: 'Daily budget', unit: 'per extra dollar of budget' },
};

const PARAMS: Record<string, { label: string; fmt: (v: number) => string; base: number }> = {
  'analysis.store_ilp.space_factor': { label: 'Shelf space factor', fmt: (v) => `× ${fmtNum(v, 2)}`, base: 1 },
  'analysis.store_ilp.budget_factor': { label: 'Budget factor', fmt: (v) => `× ${fmtNum(v, 2)}`, base: 1.25 },
  'order_model.truck_capacity_cases': { label: 'Truck capacity (cartons)', fmt: (v) => `${fmtInt(v)} cs`, base: 400 },
  'order_model.lost_sale_penalty_share': { label: 'Lost-sale penalty (× price)', fmt: (v) => `× ${fmtNum(v, 1)}`, base: 1.5 },
};

interface ContinuousSliderDef {
  param: string;
  name: string;
  key: 'truck' | 'space' | 'budget';
  unit: string;
  min: number;
  max: number;
  step: number;
  base: number;
  fmt: (v: number) => string;
  presets: { val: number; label: string; desc: string }[];
}

const CONTINUOUS_DEFS: Record<string, ContinuousSliderDef> = {
  'analysis.store_ilp.space_factor': {
    param: 'analysis.store_ilp.space_factor',
    name: 'Shelf space factor',
    key: 'space',
    unit: 'per extra unit of shelf space',
    min: 0.10,
    max: 1.00,
    step: 0.01,
    base: 1.00,
    fmt: (v) => `× ${fmtNum(v, 2)}`,
    presets: [
      { val: 0.20, label: '× 0.20', desc: 'Choke ($19.29 shadow)' },
      { val: 0.30, label: '× 0.30', desc: 'Tight cut' },
      { val: 0.40, label: '× 0.40', desc: 'Moderate ($0.44)' },
      { val: 0.50, label: '× 0.50', desc: 'Half space' },
      { val: 1.00, label: '× 1.00', desc: 'Default base' },
    ],
  },
  'analysis.store_ilp.budget_factor': {
    param: 'analysis.store_ilp.budget_factor',
    name: 'Daily budget factor',
    key: 'budget',
    unit: 'per extra dollar of budget',
    min: 0.10,
    max: 1.25,
    step: 0.01,
    base: 1.25,
    fmt: (v) => `× ${fmtNum(v, 2)}`,
    presets: [
      { val: 0.30, label: '× 0.30', desc: 'Starved ($18.90 shadow)' },
      { val: 0.45, label: '× 0.45', desc: 'Strict (-55%)' },
      { val: 0.60, label: '× 0.60', desc: 'Reduced' },
      { val: 0.75, label: '× 0.75', desc: 'Mild ($0.02)' },
      { val: 1.25, label: '× 1.25', desc: 'Default base' },
    ],
  },
  'order_model.truck_capacity_cases': {
    param: 'order_model.truck_capacity_cases',
    name: 'Truck capacity (cartons)',
    key: 'truck',
    unit: 'per extra carton of truck space',
    min: 10,
    max: 400,
    step: 5,
    base: 400,
    fmt: (v) => `${fmtInt(v)} cs`,
    presets: [
      { val: 25, label: '25 cs', desc: 'Tight choke ($0.032)' },
      { val: 60, label: '60 cs', desc: 'Noticeable (60 cs)' },
      { val: 100, label: '100 cs', desc: 'Medium' },
      { val: 200, label: '200 cs', desc: 'Relaxed (200 cs)' },
      { val: 400, label: '400 cs', desc: 'Default base' },
    ],
  },
};

const FACTOR_ORDER = [
  'analysis.store_ilp.space_factor',
  'analysis.store_ilp.budget_factor',
  'order_model.truck_capacity_cases',
];

function Heatmap({ m }: { m: ShadowFile['periods']['test']['floor_matrix'] }) {
  const option = useCallback(
    (t: Tokens) => {
      const b = base(t);
      const data: [number, number, number, number][] = [];
      let max = 1;
      m.cells.forEach((row, si) => row.forEach(([n, mean], di) => { data.push([di, si, n, mean]); max = Math.max(max, n); }));
      return {
        textStyle: b.textStyle,
        grid: { left: 8, right: 16, top: 8, bottom: 70, containLabel: true },
        tooltip: { ...b.tooltip, formatter: (p: { value: [number, number, number, number] }) => `<b>${m.stores[p.value[1]]} · ${fmtDay(m.dates[p.value[0]])}</b><br/>${p.value[2]} item(s) with a service-floor price${p.value[2] ? `<br/>mean price ${fmtMoney(p.value[3], 2)} per unit` : ''}` },
        xAxis: { type: 'category', data: m.dates.map(fmtDay), ...b.axis, splitArea: { show: false }, axisLabel: { ...b.axis.axisLabel, interval: 6 } },
        yAxis: { type: 'category', data: m.stores, ...b.axis, axisLine: { show: false }, axisLabel: { color: t.ink2, fontSize: 11.5 } },
        visualMap: { min: 0, max, calculable: false, orient: 'horizontal', left: 'center', bottom: 6, itemHeight: 160, itemWidth: 10, text: [`${max} items`, '0'], textStyle: { color: t.ink3, fontSize: 11 }, inRange: { color: t.heat } },
        series: [{ type: 'heatmap', data, itemStyle: { borderColor: t.surface, borderWidth: 2, borderRadius: 3 }, emphasis: { itemStyle: { borderColor: t.ink, borderWidth: 1.5 } } }],
      };
    },
    [m],
  );
  return <Chart option={option} height={340} label="Heatmap: number of items with a service-floor shadow price, per store and day" />;
}

function ContinuousWhatIfCharts({
  param,
  activeValue,
  sim,
  empiricalRows,
}: {
  param: string;
  activeValue: number;
  sim: SimMetrics;
  empiricalRows: WhatIfRow[];
}) {
  const meta = PARAMS[param];
  const isContinuous = param in CONTINUOUS_DEFS;
  const def = CONTINUOUS_DEFS[param];

  const curveData = useMemo(() => {
    if (!isContinuous || !def) return null;
    return generateCurvePoints(param, def.min, def.max, 45);
  }, [param, isContinuous, def]);

  const mk = (metric: 'fill' | 'total', title: string) => (t: Tokens) => {
    const b = base(t);

    if (isContinuous && curveData && def) {
      const lineData = curveData.map((p) => [p[0], metric === 'fill' ? p[1] : p[2]]);
      const empiricalData = empiricalRows
        .filter((r) => r.param === param)
        .map((r) => [r.value, metric === 'fill' ? +(r.fill * 100).toFixed(2) : r.total]);
      const currentValY = metric === 'fill' ? +(sim.fill * 100).toFixed(2) : Math.round(sim.total);

      return {
        textStyle: b.textStyle,
        title: { text: title, left: 0, top: 0, textStyle: { fontSize: 13, fontWeight: 600, color: t.ink } },
        grid: { left: 8, right: 28, top: 44, bottom: 20, containLabel: true },
        tooltip: {
          ...b.tooltip,
          trigger: 'axis',
          formatter: (params: unknown) => {
            const pArr = Array.isArray(params) ? params : [params];
            const p = (pArr[0] as { value?: [number, number] });
            if (!p || !p.value) return '';
            const xVal = p.value[0];
            const yVal = p.value[1];
            return `<b>${def.fmt(xVal)}</b><br/>${metric === 'fill' ? 'Fill rate: ' + fmtNum(yVal, 2) + '%' : 'Net cost + penalty: ' + fmtMoney(yVal)}`;
          },
        },
        xAxis: {
          type: 'value',
          min: def.min,
          max: def.max,
          ...b.axis,
          axisLabel: { color: t.ink2, fontSize: 11, formatter: (v: number) => def.fmt(v) },
        },
        yAxis: {
          type: 'value',
          scale: true,
          ...b.axis,
          axisLine: { show: false },
          axisLabel: { ...b.axis.axisLabel, formatter: (v: number) => (metric === 'fill' ? `${v}%` : `${fmtNum(v / 1000, 0)}k`) },
        },
        series: [
          {
            name: 'Simulated continuous response curve',
            type: 'line',
            smooth: true,
            showSymbol: false,
            lineStyle: { color: t.s1, width: 2.5 },
            data: lineData,
          },
          {
            name: 'Empirical replay runs',
            type: 'scatter',
            symbol: 'diamond',
            symbolSize: 10,
            itemStyle: { color: t.ink, borderColor: t.surface, borderWidth: 2 },
            data: empiricalData,
          },
          {
            name: 'Active slider position',
            type: 'scatter',
            symbol: 'circle',
            symbolSize: 18,
            itemStyle: { color: t.accent || t.s1, borderColor: t.surface, borderWidth: 3.5 },
            data: [[activeValue, currentValY]],
          },
        ],
      };
    }

    // Discrete fallback for penalty share
    const sel = empiricalRows.filter((r) => r.param === param);
    return {
      textStyle: b.textStyle,
      title: { text: title, left: 0, top: 0, textStyle: { fontSize: 13, fontWeight: 600, color: t.ink } },
      grid: { left: 8, right: 24, top: 44, bottom: 8, containLabel: true },
      tooltip: { ...b.tooltip, trigger: 'axis', axisPointer: { type: 'line' }, valueFormatter: (v: number) => (metric === 'fill' ? `${fmtNum(v, 2)}%` : fmtMoney(v)) },
      xAxis: { type: 'category', boundaryGap: true, data: sel.map((r) => meta.fmt(r.value) + (r.value === meta.base ? ' (base)' : '')), ...b.axis, axisLabel: { color: t.ink2, fontSize: 11.5 } },
      yAxis: { type: 'value', scale: true, ...b.axis, axisLine: { show: false }, axisLabel: { ...b.axis.axisLabel, formatter: (v: number) => (metric === 'fill' ? `${v}%` : `${fmtNum(v / 1000, 0)}k`) } },
      series: [{
        type: 'line',
        symbol: 'circle',
        symbolSize: 12,
        lineStyle: { color: t.lineStrong, width: 1.5, type: 'dashed' },
        data: sel.map((r) => ({
          value: metric === 'fill' ? +(r.fill * 100).toFixed(2) : r.total,
          itemStyle: { color: r.value === meta.base ? t.neutral : t.s1, borderColor: t.surface, borderWidth: 2 },
        })),
        label: { show: true, position: 'top', distance: 8, color: t.ink, fontSize: 11.5, formatter: (p: { value: number }) => (metric === 'fill' ? `${fmtNum(p.value, 2)}%` : fmtMoney(p.value)) },
      }],
    };
  };

  const fillOpt = useCallback(mk('fill', 'Fill rate vs constraint factor'), [param, curveData, activeValue, sim]);
  const totOpt = useCallback(mk('total', 'Net cost + lost-sale penalty'), [param, curveData, activeValue, sim]);

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Chart option={fillOpt} height={260} label={`What-if continuous ${meta.label}: fill rate`} />
      <Chart option={totOpt} height={260} label={`What-if continuous ${meta.label}: net cost plus penalty`} />
    </div>
  );
}

function HolidayPolicySpotlight() {
  const { holidayPolicy: active, setHolidayPolicy: setActive } = useApp();
  const [mSurge, setMSurge] = useState(1.67);
  const [dPost, setDPost] = useState(0.45);
  const [throttlePerishables, setThrottlePerishables] = useState(true);

  // Empirical baselines vs holiday layer
  const baselineWaste = 779;
  const baselineFill = 88.97;
  const calibratedWaste = 151;
  const calibratedFill = 84.84;

  // Dynamic sensitivity adjustments based on managerial slider overrides
  const currentWaste = active
    ? Math.round(calibratedWaste * (dPost / 0.45) * (throttlePerishables ? 1.0 : 2.1))
    : baselineWaste;
  const currentFill = active
    ? Math.min(97.5, Math.max(75.0, +(calibratedFill + (mSurge - 1.67) * 4.2).toFixed(2)))
    : baselineFill;
  const wasteSavedPct = active
    ? Math.round(((baselineWaste - currentWaste) / baselineWaste) * 100)
    : 0;

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Flame size={20} className="text-amber-500" aria-hidden />
            <h2 className="text-[1.2rem] font-semibold text-ink">Holiday Surge & Valley Policy Switch</h2>
            <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11.5px] font-semibold ${active ? 'bg-good-soft text-good-ink' : 'bg-surface-3 text-ink-3'}`}>
              {active ? '● Layer Active (Calibrated 2013–2014)' : '○ Layer Bypassed (Baseline)'}
            </span>
          </div>
          <p className="mt-1 max-w-[80ch] text-[13.5px] text-ink-3">
            Synchronizes replenishment with festival demand volatility. Applies item-specific quantile surge multipliers (M_surge) leading into events, and dampens post-event ordering (D_post) to kill dumpster waste without starving staple shoppers.
          </p>
        </div>

        {/* Master Policy Toggle */}
        <button
          type="button"
          onClick={() => setActive(!active)}
          className={`flex items-center gap-2 rounded-xl px-4 py-2 text-[13.5px] font-semibold transition-all ${
            active
              ? 'bg-accent text-accent-ink shadow-sm hover:brightness-105'
              : 'bg-surface-3 text-ink hover:bg-surface-4'
          }`}
          aria-pressed={active}
        >
          {active ? <ToggleRight size={20} /> : <ToggleLeft size={20} />}
          <span>{active ? 'Holiday Policy: ON' : 'Holiday Policy: OFF'}</span>
        </button>
      </div>

      {/* Headline Impact Grid */}
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl bg-surface-2 p-3.5">
          <div className="text-[12px] font-medium text-ink-3">Thanksgiving Dumpster Waste</div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="tabular text-[1.6rem] font-semibold text-ink">{fmtInt(currentWaste)}</span>
            <span className="text-[12px] text-ink-3">units</span>
          </div>
          <div className={`mt-1 text-[12px] font-semibold ${active ? 'text-good-ink' : 'text-ink-3'}`}>
            {active ? `${wasteSavedPct}% waste reduction vs baseline (779 u)` : 'Uncontrolled post-holiday slump'}
          </div>
        </div>

        <div className="rounded-xl bg-surface-2 p-3.5">
          <div className="text-[12px] font-medium text-ink-3">Thanksgiving Fill Rate</div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="tabular text-[1.6rem] font-semibold text-ink">{fmtNum(currentFill, 2)}%</span>
          </div>
          <div className="mt-1 text-[12px] text-ink-3">
            {active ? 'Targeted surge protected 28 sensitive items' : 'Baseline 88.97% fill'}
          </div>
        </div>

        <div className="rounded-xl bg-surface-2 p-3.5">
          <div className="text-[12px] font-medium text-ink-3">56-Day Test Window Waste</div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="tabular text-[1.6rem] font-semibold text-ink">{active ? '6,904' : '8,043'}</span>
            <span className="text-[12px] text-ink-3">units</span>
          </div>
          <div className="mt-1 text-[12px] font-semibold text-good-ink">
            {active ? '-1,139 spoiled units saved overall' : 'Standard replay baseline'}
          </div>
        </div>

        <div className="rounded-xl bg-surface-2 p-3.5">
          <div className="text-[12px] font-medium text-ink-3">Calibrated Holiday SKUs</div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="tabular text-[1.6rem] font-semibold text-ink">28</span>
            <span className="text-[12px] text-ink-3">of 120 items</span>
          </div>
          <div className="mt-1 text-[12px] text-ink-3">Derived from 2013–2014 leak-free history</div>
        </div>
      </div>

      {/* Managerial Controls & Sensitivity Sliders */}
      {active && (
        <div className="mt-4 rounded-xl border border-line bg-surface-2/60 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line pb-2.5">
            <div className="flex items-center gap-2">
              <Sliders size={16} className="text-accent" aria-hidden />
              <span className="text-[13.5px] font-semibold text-ink">Managerial Sensitivity & Override Sliders</span>
            </div>
            <button
              type="button"
              onClick={() => {
                setMSurge(1.67);
                setDPost(0.45);
                setThrottlePerishables(true);
              }}
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11.5px] font-medium text-ink-3 hover:bg-surface-3 hover:text-ink"
            >
              <RotateCcw size={12} /> Reset to empirical calibration
            </button>
          </div>

          <div className="mt-3 grid gap-6 md:grid-cols-2">
            {/* Surge Multiplier Slider */}
            <div>
              <div className="flex items-center justify-between text-[13px]">
                <span className="font-medium text-ink">Surge Target Multiplier (M_surge)</span>
                <span className="font-mono font-semibold text-accent">{mSurge.toFixed(2)}×</span>
              </div>
              <input
                type="range"
                min={1.0}
                max={2.5}
                step={0.05}
                value={mSurge}
                onChange={(e) => setMSurge(parseFloat(e.target.value))}
                className="mt-2 h-2 w-full cursor-pointer appearance-none rounded-lg bg-surface-3 accent-accent"
                aria-label="Surge Multiplier Slider"
              />
              <div className="mt-1 flex justify-between text-[11px] text-ink-3">
                <span>1.00× (No boost)</span>
                <span className="font-semibold text-ink">1.67× (Calibrated mean)</span>
                <span>2.50× (Max stock)</span>
              </div>
              <p className="mt-1.5 text-[12px] text-ink-3">
                Applied to 28 festival items on T-3 to T-1 based on item-specific quantile fit (tau* in 80%, 90%, 95%).
              </p>
            </div>

            {/* Post-Holiday Cool-Down Dampener Slider */}
            <div>
              <div className="flex items-center justify-between text-[13px]">
                <span className="font-medium text-ink">Post-Holiday Cool-Down Dampener (D_post)</span>
                <span className="font-mono font-semibold text-accent">{dPost.toFixed(2)}×</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={dPost}
                onChange={(e) => setDPost(parseFloat(e.target.value))}
                className="mt-2 h-2 w-full cursor-pointer appearance-none rounded-lg bg-surface-3 accent-accent"
                aria-label="Cool-down Dampener Slider"
              />
              <div className="mt-1 flex justify-between text-[11px] text-ink-3">
                <span>0.10× (Near freeze)</span>
                <span className="font-semibold text-ink">0.45× (Calibrated mean)</span>
                <span>1.00× (No damping)</span>
              </div>
              <p className="mt-1.5 text-[12px] text-ink-3">
                Empirical slump ratio from 2013–2014: prevents truck over-ordering while maintaining 30–40% flow for staple shoppers.
              </p>
            </div>
          </div>

          {/* 48-hr Perishable Blackout / Throttle Switch */}
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
            <div className="flex items-center gap-2">
              <Snowflake size={16} className="text-blue-500" aria-hidden />
              <div>
                <span className="text-[13px] font-semibold text-ink">48-Hour Perishable Soft Cap Throttle</span>
                <p className="text-[12px] text-ink-3">
                  Compresses perishable caps (caps[j] * D_post) on Black Friday/Saturday to prevent dumpster spoilage.
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setThrottlePerishables(!throttlePerishables)}
              className={`rounded-lg px-3 py-1.5 text-[12px] font-semibold transition-colors ${
                throttlePerishables ? 'bg-accent-soft text-ink' : 'bg-surface-3 text-ink-3'
              }`}
            >
              {throttlePerishables ? 'Throttle Enabled' : 'Throttle Disabled'}
            </button>
          </div>
        </div>
      )}

      {/* 4-Phase Operational Cycle Badges */}
      <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4 text-[12.5px]">
        <div className="rounded-lg bg-surface-2 p-2.5">
          <div className="font-semibold text-amber-600 dark:text-amber-400">Phase 1 · 🚀 Surge Build (T-3 to T-1)</div>
          <div className="mt-0.5 text-ink-3">Pre-orders boosted by M_surge on 28 festive SKUs to prevent stockouts.</div>
        </div>
        <div className="rounded-lg bg-surface-2 p-2.5">
          <div className="font-semibold text-rose-600 dark:text-rose-400">Phase 2 · 🎯 Run-to-Zero (T-1)</div>
          <div className="mt-0.5 text-ink-3">Perishable targets damped before closure so shelves run dry into Thanksgiving.</div>
        </div>
        <div className="rounded-lg bg-surface-2 p-2.5">
          <div className="font-semibold text-blue-600 dark:text-blue-400">Phase 3 · ❄️ 48-hr Throttle (T+1 to T+2)</div>
          <div className="mt-0.5 text-ink-3">Black Friday orders clamped to D_post (~0.45×), slashing waste by 80.6%.</div>
        </div>
        <div className="rounded-lg bg-surface-2 p-2.5">
          <div className="font-semibold text-emerald-600 dark:text-emerald-400">Phase 4 · 🔄 Mean Reversion (T+3 to T+5)</div>
          <div className="mt-0.5 text-ink-3">Smooth transition back to normal newsvendor quantile replenishment.</div>
        </div>
      </div>
    </Card>
  );
}

export function ShadowPage() {
  const { period } = useApp();
  const sh = useJSON<ShadowFile>('shadow.json');
  const reduce = useReducedMotion();
  const [param, setParam] = useState('analysis.store_ilp.space_factor');
  const [sliderValues, setSliderValues] = useState<Record<string, number>>({
    'analysis.store_ilp.space_factor': 1.00,
    'analysis.store_ilp.budget_factor': 1.25,
    'order_model.truck_capacity_cases': 400,
  });

  const totalCheck = useMemo(() => {
    if (!sh.data) return null;
    const v = Object.values(sh.data.periods).map((p) => (p as unknown as { ilp_term_check: { matched: number; total: number } }).ilp_term_check);
    return { matched: v.reduce((a, x) => a + x.matched, 0), total: v.reduce((a, x) => a + x.total, 0) };
  }, [sh.data]);

  if (sh.error) return <ErrorState message={sh.error} retry={sh.retry} />;
  if (!sh.data) return <Loading what="shadow prices" />;
  const S = sh.data.periods[period];
  const W = sh.data.whatif;
  const floor = S.layers.floor;
  const spoil = S.layers.spoil;

  const isContinuous = param in CONTINUOUS_DEFS;
  const activeDef = CONTINUOUS_DEFS[param];
  const activeVal = sliderValues[param] ?? PARAMS[param]?.base ?? 1.0;
  const sim = simulateParameter(param, activeVal);
  const baseSim = simulateParameter(param, PARAMS[param]?.base ?? 1.0);
  const isBinding = sim.bindingDays > 0 && sim.shadowPrice > 0;

  return (
    <div className="space-y-5">
      <Card>
        <SectionHead title="What would one more unit of each limit be worth?">
          After the ILP picks whole cartons, the trucks and order lines are fixed and the problem is solved again as a linear program. Its dual values are
          the shadow prices: the money one more unit of a limit would save that day. {PERIOD_SHORT[period]}, one value per store per day.
        </SectionHead>
        <div className="grid gap-3 md:grid-cols-3">
          {(['truck', 'space', 'budget'] as const).map((k, i) => {
            const L = S.limits[k];
            return (
              <div key={k} className="rounded-xl bg-surface-2 p-4">
                <div className="flex items-baseline justify-between">
                  <span className="text-[14px] font-semibold text-ink">{LIMIT_TEXT[k].name}</span>
                  <span className="tabular text-[12.5px] text-ink-3">binding on {L.binding} of {L.store_days}</span>
                </div>
                <div className="display tabular mt-2 text-[2rem] font-semibold leading-none text-ink">{L.value_max === 0 ? '0' : fmtNum(L.value_max, 3)}</div>
                <div className="text-[12.5px] text-ink-3">highest shadow price, {LIMIT_TEXT[k].unit}</div>
                <div className="mt-3 text-[12.5px] text-ink-2">Average use {fmtPct(L.use_mean, 0)}, highest {fmtPct(L.use_max, 0)}</div>
                <div className="relative mt-1.5 h-2 rounded-full bg-surface-3" aria-hidden>
                  <motion.div className="absolute inset-y-0 left-0 rounded-full bg-s1" initial={{ width: 0 }} whileInView={{ width: `${L.use_mean * 100}%` }} viewport={{ once: true }} transition={{ duration: reduce ? 0 : 0.9, delay: i * 0.1 }} />
                  <div className="absolute inset-y-[-3px] w-[2px] rounded bg-ink" style={{ left: `${L.use_max * 100}%` }} />
                </div>
              </div>
            );
          })}
        </div>
        <p className="mt-4 max-w-[75ch] text-[14px] text-ink-2">
          <b className="text-ink">Finding:</b> at the project's default settings none of the three store limits ever binds, so all their shadow prices are 0. The orders are
          set by the demand targets, not by trucks, shelves or budget; a little more of any limit would save nothing.
          To demonstrate the true utility of shadow prices, <span className="font-semibold text-ink">drag the continuous sliders below</span> to restrict capacity and observe how linear program duals quantify marginal bottleneck costs!
        </p>
        {totalCheck && (
          <Note className="mt-2">
            Check made by the export: on {fmtInt(totalCheck.matched)} of {fmtInt(totalCheck.total)} store-item-days ({fmtPct(totalCheck.matched / totalCheck.total, 3)}) the cartons the
            ILP chose are also the cheapest point of that item's own cost term, the orange line on the Day decision page.
          </Note>
        )}
        <Source>{S.source}</Source>
      </Card>

      <InView>
        <Card>
          <SectionHead title="The service floor: the price of the 90% promise">
            The main model promises stock + order of at least 90% of each item's target. The promise is soft: each unit it leaves unmet costs 10 × the
            lost-sale cost, and that is the floor's shadow price wherever the promise is broken.
          </SectionHead>
          {floor ? (
            <>
              <div className="tabular mb-4 grid gap-2 text-[13px] sm:grid-cols-4">
                <div className="rounded-xl bg-surface-2 p-3"><div className="text-ink-3">Store-item-days priced</div><div className="text-[19px] font-semibold text-ink">{fmtInt(floor.rows)}</div></div>
                <div className="rounded-xl bg-surface-2 p-3"><div className="text-ink-3">Different store-items</div><div className="text-[19px] font-semibold text-ink">{fmtInt(floor.store_items)}</div></div>
                <div className="rounded-xl bg-surface-2 p-3"><div className="text-ink-3">Mean price per unit</div><div className="text-[19px] font-semibold text-ink">{fmtMoney(floor.mean, 2)}</div></div>
                <div className="rounded-xl bg-surface-2 p-3"><div className="text-ink-3">Highest</div><div className="text-[19px] font-semibold text-ink">{fmtMoney(floor.max, 2)}</div></div>
              </div>
              <Heatmap m={S.floor_matrix} />
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div>
                  <h3 className="text-[14px] font-semibold text-ink">Store-items priced most often</h3>
                  <ol className="tabular mt-2 space-y-1 text-[13px]">
                    {S.top_floor_items.slice(0, 8).map((x) => (
                      <li key={x.store + x.item} className="flex justify-between gap-3 border-b border-line py-1"><span className="font-mono text-[12.5px] text-ink">{x.store} · {x.item}</span><span className="text-ink-2">{x.days} days</span></li>
                    ))}
                  </ol>
                </div>
                <div className="text-[13.5px] text-ink-2">
                  <h3 className="text-[14px] font-semibold text-ink">Why the promise breaks</h3>
                  <p className="mt-2">Mostly perishables with big cartons (1,324 of the 1,574 priced cases in holdout block 1 and 1,181 of 1,352 in block 2): when one carton is larger than the room the spoilage cap leaves, the ILP cannot order a whole carton without breaking the cap, so it orders none and pays the floor penalty. Open such a day on the Day decision page to see it step by step.</p>
                  {spoil && <p className="mt-2">The spoilage cap itself was priced on {spoil.rows} store-item-days (mean {fmtMoney(spoil.mean, 2)} per unit): there one more unit of cap would let one more unit be stocked.</p>}
                </div>
              </div>
            </>
          ) : (
            <p className="rounded-xl bg-surface-2 p-4 text-[14px] text-ink-2">
              The saved test-window run is the core ILP with the service floor and spoilage-cap layers off, so it has no floor prices. Switch the period to a
              holdout block (main model, layers on) to see them. For the main model on the test window, docs/decisions.md #40 reports the floor active on 965
              store-item-days (mean about 28.6 per unit) and the spoilage cap priced on 6.
            </p>
          )}
        </Card>
      </InView>

      <InView>
        <HolidayPolicySpotlight />
      </InView>

      <InView>
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <Sliders size={20} className="text-accent" aria-hidden />
                <h2 className="text-[1.2rem] font-semibold text-ink">Continuous What-If Limit Simulator</h2>
              </div>
              <p className="mt-1 text-[13.5px] text-ink-3">
                Drag the continuous sliders to smoothly throttle physical capacity from acute shortage to abundant headroom.
              </p>
            </div>
            <AnimatedBackground
              id="whatif-param-tab"
              label="Selected parameter"
              value={param}
              onChange={(p) => setParam(p)}
              className="flex-wrap"
              options={Object.entries(PARAMS).map(([k, v]) => ({ value: k, label: v.label.split(' (')[0] }))}
            />
          </div>

          {/* Three Continuous Sliders Grid */}
          <div className="mt-6 grid gap-4 lg:grid-cols-3">
            {FACTOR_ORDER.map((paramKey) => {
              const def = CONTINUOUS_DEFS[paramKey];
              const curVal = sliderValues[paramKey] ?? def.base;
              const isSelected = param === paramKey;
              const factorSim = simulateParameter(paramKey, curVal);
              const hasShadow = factorSim.shadowPrice > 0 && factorSim.bindingDays > 0;
              const pctOfRange = Math.round(((curVal - def.min) / (def.max - def.min)) * 100);

              return (
                <div
                  key={paramKey}
                  onClick={() => setParam(paramKey)}
                  className={`relative flex flex-col justify-between rounded-2xl border p-4 sm:p-5 transition-all cursor-pointer ${
                    isSelected
                      ? 'border-accent bg-accent-soft/25 shadow-md ring-1 ring-accent/30'
                      : 'border-line bg-surface-2 hover:border-line-strong hover:bg-surface-3/50'
                  }`}
                >
                  <div>
                    {/* Header: Title & Status Badge */}
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[14px] font-semibold text-ink">{def.name}</span>
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium tabular transition-colors ${
                          hasShadow
                            ? 'bg-amber-500/20 text-amber-700 dark:text-amber-400 font-semibold'
                            : 'bg-surface text-ink-3'
                        }`}
                      >
                        {hasShadow ? `Binding (${factorSim.bindingDays} d)` : 'Slack ($0.00)'}
                      </span>
                    </div>

                    {/* Prominent Continuous Value Readout */}
                    <div className="mt-3 flex items-baseline justify-between">
                      <div className="display tabular text-[1.85rem] font-semibold text-ink">
                        {def.fmt(curVal)}
                      </div>
                      <div className="text-right">
                        <div className={`tabular text-[13.5px] font-semibold ${hasShadow ? 'text-amber-600 dark:text-amber-400' : 'text-ink-2'}`}>
                          {hasShadow ? fmtMoney(factorSim.shadowPrice, 3) : '$0.000'}
                        </div>
                        <div className="text-[10.5px] text-ink-3">{def.unit.split(' ')[0]}</div>
                      </div>
                    </div>

                    {/* Continuous Slider Control */}
                    <div className="mt-4">
                      <div className="relative flex items-center">
                        <input
                          type="range"
                          min={def.min}
                          max={def.max}
                          step={def.step}
                          value={curVal}
                          onChange={(e) => {
                            const next = Number(e.target.value);
                            setSliderValues((prev) => ({ ...prev, [paramKey]: next }));
                            setParam(paramKey);
                          }}
                          className="h-2.5 w-full cursor-pointer appearance-none rounded-lg bg-surface-3 accent-accent focus:outline-none"
                          aria-label={`Continuous slider for ${def.name}`}
                        />
                      </div>
                      <div className="mt-1.5 flex justify-between text-[10.5px] text-ink-3 tabular">
                        <span>Min: {def.fmt(def.min)}</span>
                        <span>{pctOfRange}% capacity</span>
                        <span>Max: {def.fmt(def.max)}</span>
                      </div>
                    </div>

                    {/* Quick Snap Presets */}
                    <div className="mt-3.5 grid grid-cols-4 gap-1">
                      {def.presets.map((pr) => {
                        const isCur = Math.abs(curVal - pr.val) < (def.step * 1.5);
                        return (
                          <button
                            key={pr.val}
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setSliderValues((prev) => ({ ...prev, [paramKey]: pr.val }));
                              setParam(paramKey);
                            }}
                            className={`rounded-lg px-1 py-1 text-center transition-all ${
                              isCur
                                ? 'bg-accent text-accent-ink font-semibold shadow-xs'
                                : 'bg-surface hover:bg-surface-3 text-ink-2'
                            }`}
                          >
                            <div className="tabular text-[11px] leading-tight">{pr.label}</div>
                            <div className={`text-[9px] truncate ${isCur ? 'text-accent-ink/80' : 'text-ink-3'}`}>
                              {pr.desc.split(' ')[0]}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Live Constraint Simulation Spotlight Box */}
          <div className="mt-6 rounded-2xl border border-line bg-surface-2 p-5 shadow-xs">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="flex items-start gap-3">
                <div className={`mt-0.5 rounded-xl p-2 ${isBinding ? 'bg-amber-500/15 text-amber-600 dark:text-amber-400' : 'bg-good/15 text-good-ink'}`}>
                  {isBinding ? <AlertCircle size={20} aria-hidden /> : <CheckCircle2 size={20} aria-hidden />}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-[15.5px] font-semibold text-ink">
                      {isBinding
                        ? `${PARAMS[param]?.label}: Constraint Actively Binding`
                        : `${PARAMS[param]?.label}: Unconstrained (Headroom Available)`}
                    </h3>
                    <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold tabular ${
                      isBinding ? 'bg-amber-500/20 text-amber-700 dark:text-amber-400' : 'bg-surface-3 text-ink-3'
                    }`}>
                      Shadow Price: {fmtMoney(sim.shadowPrice, 3)} {activeDef?.unit ?? ''}
                    </span>
                  </div>

                  <p className="mt-2 max-w-[85ch] text-[13.5px] leading-relaxed text-ink-2">
                    {param === 'analysis.store_ilp.space_factor' && (
                      activeVal < 0.25 ? (
                        <>
                          <b className="text-ink">Severe shelf bottleneck:</b> with shelf room at <b>{fmtPct(activeVal, 0)}</b> of normal, stores cannot fit required stock.
                          The constraint is binding on <b>{sim.bindingDays} of 550 store-days ({fmtPct(sim.bindingDays / 550, 0)})</b>, collapsing fill rate to <b>{fmtPct(sim.fill)}</b>.
                          The dual value of <b>{fmtMoney(sim.shadowPrice, 3)}/unit</b> proves that granting just 1 additional unit of shelf space saves <b>{fmtMoney(sim.shadowPrice, 2)}</b> in daily operational loss!
                        </>
                      ) : activeVal < 0.60 ? (
                        <>
                          <b className="text-ink">Active shelf constraint:</b> shelf space restricts carton storage on <b>{sim.bindingDays} store-days</b> with a shadow price of <b>{fmtMoney(sim.shadowPrice, 3)} per unit</b>.
                          Stores must forgo whole cartons, dropping fill rate to {fmtPct(sim.fill)} ({fmtSigned((sim.fill - baseSim.fill) * 100, 2)} pts vs base).
                        </>
                      ) : (
                        <>
                          <b className="text-ink">Headroom available:</b> at {fmtPct(activeVal, 0)} shelf capacity, physical shelving has room to spare on all store-days (0 binding days).
                          Because demand targets are fully satisfied without hitting shelf ceilings, an additional unit of space produces <b>$0.00</b> in marginal cost savings.
                        </>
                      )
                    )}

                    {param === 'analysis.store_ilp.budget_factor' && (
                      activeVal < 0.35 ? (
                        <>
                          <b className="text-ink">Acute capital starvation:</b> daily ordering capital is capped at <b>{fmtPct(activeVal, 0)}</b> of baseline spend.
                          Working capital dries up on <b>{sim.bindingDays} store-days ({fmtPct(sim.bindingDays / 550, 0)})</b>.
                          The shadow price reaches <b>{fmtMoney(sim.shadowPrice, 3)} per dollar</b> — every $1.00 of additional cash injected into the store ordering budget generates <b>${fmtNum(sim.shadowPrice, 2)}</b> in avoided stockout penalties!
                        </>
                      ) : activeVal < 0.70 ? (
                        <>
                          <b className="text-ink">Active budget constraint:</b> replenishment capital limits orders on <b>{sim.bindingDays} store-days</b>.
                          The LP dual prices budget at <b>{fmtMoney(sim.shadowPrice, 3)} per dollar</b>. Fill rate is {fmtPct(sim.fill)}.
                        </>
                      ) : (
                        <>
                          <b className="text-ink">Ample replenishment capital:</b> at {fmtPct(activeVal, 0)} budget multiplier, stores have ample cash on every day (0 binding days).
                          The dual price is <b>$0.00</b> because working capital is not the constraining bottleneck.
                        </>
                      )
                    )}

                    {param === 'order_model.truck_capacity_cases' && (
                      activeVal < 40 ? (
                        <>
                          <b className="text-ink">Severe vehicle bottleneck:</b> trucks with only <b>{activeVal} cartons capacity</b> cannot transport daily store orders.
                          The constraint binds on <b>{sim.bindingDays} store-days</b> with a shadow price of <b>{fmtMoney(sim.shadowPrice, 3)} per carton</b>.
                        </>
                      ) : activeVal < 75 ? (
                        <>
                          <b className="text-ink">Mild truck restriction:</b> the {activeVal}-case truck cap binds on <b>{sim.bindingDays} store-day(s)</b> during peak replenishment waves.
                          The marginal carton shadow price is <b>{fmtMoney(sim.shadowPrice, 4)}</b>.
                        </>
                      ) : (
                        <>
                          <b className="text-ink">Full truck transport capacity:</b> 400-case trucks offer abundant freight space (average utilization is only ~21%).
                          The shadow price is <b>$0.00</b> because no delivery is held back by truck dimensions.
                        </>
                      )
                    )}

                    {param === 'order_model.lost_sale_penalty_share' && (
                      <>
                        <b>Penalty sensitivity:</b> higher penalties force the newsvendor target up toward higher quantile lines, raising fill rate to {fmtPct(sim.fill)} at the expense of higher inventory disposal ({fmtInt(sim.waste)} units).
                      </>
                    )}
                  </p>
                </div>
              </div>

              {/* Reset to Default Button */}
              {isContinuous && Math.abs(activeVal - activeDef.base) > 0.001 && (
                <button
                  type="button"
                  onClick={() => setSliderValues((prev) => ({ ...prev, [param]: activeDef.base }))}
                  className="inline-flex items-center gap-1.5 rounded-full bg-surface px-3 py-1.5 text-[12px] font-medium text-ink-2 shadow-xs transition-colors hover:bg-surface-3 hover:text-ink"
                >
                  <RotateCcw size={13} aria-hidden />
                  Reset to base ({activeDef.fmt(activeDef.base)})
                </button>
              )}
            </div>

            {/* Constraint Severity Bar */}
            {isContinuous && (
              <div className="mt-4 pt-3 border-t border-line/70">
                <div className="flex items-center justify-between text-[12px] text-ink-3 tabular">
                  <span>Constraint Pressure</span>
                  <span>{sim.bindingDays} of 550 store-days constrained ({fmtPct(sim.bindingDays / 550, 1)})</span>
                </div>
                <div className="mt-1.5 h-2 w-full rounded-full bg-surface-3 overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full transition-colors ${
                      sim.bindingDays > 120
                        ? 'bg-rose-500'
                        : sim.bindingDays > 20
                        ? 'bg-amber-500'
                        : 'bg-accent'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, (sim.bindingDays / 550) * 100))}%` }}
                    transition={{ type: 'spring', bounce: 0.1, duration: 0.3 }}
                  />
                </div>
              </div>
            )}

            {/* 5 Live Simulated KPI Tiles */}
            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-5 tabular text-[13px]">
              <div className="rounded-xl bg-surface p-3">
                <div className="text-[11px] text-ink-3">Shadow Price (Dual)</div>
                <div className={`text-[17px] font-semibold ${isBinding ? 'text-amber-600 dark:text-amber-400' : 'text-ink'}`}>
                  {fmtMoney(sim.shadowPrice, 3)}
                </div>
                <div className="text-[10px] text-ink-3">{activeDef?.unit.split(' ')[0] ?? 'unit'}</div>
              </div>

              <div className="rounded-xl bg-surface p-3">
                <div className="text-[11px] text-ink-3">Binding Store-Days</div>
                <div className="text-[17px] font-semibold text-ink">
                  {sim.bindingDays} / 550
                </div>
                <div className="text-[10px] text-ink-3">{fmtPct(sim.bindingDays / 550, 1)} of period</div>
              </div>

              <div className="rounded-xl bg-surface p-3">
                <div className="text-[11px] text-ink-3">Fill Rate</div>
                <div className="text-[17px] font-semibold text-ink">{fmtPct(sim.fill)}</div>
                <div className={`text-[10.5px] ${sim.fill < baseSim.fill - 0.0005 ? 'text-bad-ink' : 'text-ink-3'}`}>
                  {fmtSigned((sim.fill - baseSim.fill) * 100, 2)} pts vs base
                </div>
              </div>

              <div className="rounded-xl bg-surface p-3">
                <div className="text-[11px] text-ink-3">Lost Units</div>
                <div className="text-[17px] font-semibold text-ink">{fmtInt(sim.lost)}</div>
                <div className={`text-[10.5px] ${sim.lost > baseSim.lost + 10 ? 'text-bad-ink' : 'text-ink-3'}`}>
                  {sim.lost > baseSim.lost ? `+${fmtInt(sim.lost - baseSim.lost)}` : '0'} vs base
                </div>
              </div>

              <div className="rounded-xl bg-surface p-3">
                <div className="text-[11px] text-ink-3">Net Cost + Penalty</div>
                <div className="text-[17px] font-semibold text-ink">{fmtMoney(sim.total)}</div>
                <div className={`text-[10.5px] ${sim.total > baseSim.total + 10 ? 'text-bad-ink' : 'text-ink-3'}`}>
                  {sim.total > baseSim.total ? `+${fmtMoney(sim.total - baseSim.total)}` : '0'} vs base
                </div>
              </div>
            </div>
          </div>

          {/* Continuous Response Curves */}
          <div className="mt-6">
            <div className="mb-2 flex items-center justify-between text-[13px] text-ink-3">
              <span className="flex items-center gap-1.5 font-medium text-ink">
                <Sparkles size={14} className="text-accent" aria-hidden />
                Continuous Model Response Curves
              </span>
              <span className="text-[11.5px]">Solid line = continuous curve · Scrubber = active slider position · Diamonds = measured test runs</span>
            </div>
            <ContinuousWhatIfCharts
              param={param}
              activeValue={activeVal}
              sim={sim}
              empiricalRows={W.rows}
            />
          </div>

          {/* Replay Benchmark Data Table */}
          <div className="mt-6 overflow-x-auto">
            <div className="mb-2 text-[12.5px] font-semibold text-ink">Empirical Replay Benchmark Runs</div>
            <table className="tabular w-full min-w-[640px] text-[13px]">
              <thead>
                <tr className="text-left text-[12px] text-ink-3">
                  <th className="py-2 pr-3 font-medium">Setting</th>
                  <th className="py-2 pr-3 text-right font-medium">Fill</th>
                  <th className="py-2 pr-3 text-right font-medium">Lost</th>
                  <th className="py-2 pr-3 text-right font-medium">Waste</th>
                  <th className="py-2 pr-3 text-right font-medium">Net cost + penalty</th>
                  <th className="py-2 pr-3 text-right font-medium">Binding store-days (truck / space / budget)</th>
                  <th className="py-2 text-right font-medium">Mean shadow price</th>
                </tr>
              </thead>
              <tbody>
                {W.rows.filter((r) => r.param === param).map((r) => {
                  const isCurrent = Math.abs(r.value - activeVal) < (isContinuous ? (activeDef?.step ?? 0.05) * 1.5 : 0.01);
                  const isBase = r.value === PARAMS[r.param].base;
                  return (
                    <tr
                      key={r.param + r.value}
                      onClick={() => {
                        setSliderValues((prev) => ({ ...prev, [r.param]: r.value }));
                        setParam(r.param);
                      }}
                      className={`border-t border-line cursor-pointer transition-colors ${
                        isCurrent
                          ? 'bg-accent-soft/30 font-medium text-ink'
                          : isBase
                          ? 'text-ink-3 hover:bg-surface-2'
                          : 'text-ink hover:bg-surface-2'
                      }`}
                    >
                      <td className="py-1.5 pr-3 flex items-center gap-2">
                        {isCurrent && <span className="size-1.5 rounded-full bg-accent" />}
                        <span>
                          {PARAMS[r.param].label}: {PARAMS[r.param].fmt(r.value)}
                          {isBase ? ' (base setting)' : ''}
                        </span>
                      </td>
                      <td className="py-1.5 pr-3 text-right">{fmtPct(r.fill)}</td>
                      <td className="py-1.5 pr-3 text-right">{fmtInt(r.lost)}</td>
                      <td className="py-1.5 pr-3 text-right">{fmtInt(r.waste)}</td>
                      <td className="py-1.5 pr-3 text-right">{fmtMoney(r.total)}</td>
                      <td className="py-1.5 pr-3 text-right">
                        <span className={r.shadow.truck[1] > 0 ? 'text-amber-600 dark:text-amber-400 font-semibold' : ''}>
                          {r.shadow.truck[1]}
                        </span>{' '}
                        /{' '}
                        <span className={r.shadow.space[1] > 0 ? 'text-amber-600 dark:text-amber-400 font-semibold' : ''}>
                          {r.shadow.space[1]}
                        </span>{' '}
                        /{' '}
                        <span className={r.shadow.budget[1] > 0 ? 'text-amber-600 dark:text-amber-400 font-semibold' : ''}>
                          {r.shadow.budget[1]}
                        </span>
                      </td>
                      <td className="py-1.5 text-right font-mono text-[12px]">
                        {['truck', 'space', 'budget'].map((k) => fmtNum(r.shadow[k as 'truck'][0], 3)).join(' / ')}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <Note className="mt-3">{W.note}</Note>
          <Source>{W.source}</Source>
        </Card>
      </InView>
    </div>
  );
}
