import { useCallback } from 'react';
import { motion } from 'motion/react';
import { CheckCircle2, Info, Flame, ArrowRight } from 'lucide-react';
import { useApp } from '../lib/state';
import { HolidayToggle } from '../components/HolidayToggle';
import { METHOD_SHORT, useJSON, type CalendarFile, type MethodId, type MethodRow, type Summary } from '../lib/data';
import { fmtDay, fmtInt, fmtMoney, fmtNum, fmtPct, fmtSigned } from '../lib/format';
import { Card, ErrorState, Loading, Note, SectionHead, Source } from '../components/ui';
import { Chart, base, type Tokens } from '../components/Chart';
import { TextEffect } from '../components/motion/TextEffect';
import { AnimatedNumber } from '../components/motion/AnimatedNumber';
import { AnimatedGroup, groupItem } from '../components/motion/AnimatedGroup';
import { Spotlight } from '../components/motion/Spotlight';
import { Tilt } from '../components/motion/Tilt';
import { InView } from '../components/motion/InView';

const ORDER: MethodId[] = ['store_ilp', 'topup_forecast', 'topup_avg', 'last_week'];

export function methodColor(t: Tokens, id: MethodId) {
  if (id === 'store_ilp') return t.s1;
  if (id === 'store_model') return t.s1soft;
  return t.neutral;
}

function Kpi({ label, value, format, sub }: { label: string; value: number; format: (x: number) => string; sub: React.ReactNode }) {
  return (
    <motion.div variants={groupItem}>
      <Tilt className="h-full">
        <div className="flex h-full flex-col justify-between gap-3 rounded-2xl bg-surface p-4 shadow-card sm:p-5">
          <div className="text-[13px] font-medium text-ink-2">{label}</div>
          <AnimatedNumber value={value} format={format} className="display tabular text-[2rem] font-semibold leading-none text-ink sm:text-[2.35rem]" />
          <div className="text-[12.5px] leading-snug text-ink-3">{sub}</div>
        </div>
      </Tilt>
    </motion.div>
  );
}

function MetricBars({ rows, metric, title, fmt, lowerBetter }: { rows: MethodRow[]; metric: 'fill' | 'waste' | 'stock' | 'total'; title: string; fmt: (x: number) => string; lowerBetter: boolean }) {
  const option = useCallback(
    (t: Tokens) => {
      const b = base(t);
      const data = [...ORDER].reverse().map((id) => rows.find((r) => r.id === id)!);
      const vals = data.map((r) => (metric === 'fill' ? r.fill * 100 : r[metric]));
      const min = 0; // bars always start at zero
      return {
        textStyle: b.textStyle,
        grid: { left: 8, right: 70, top: 4, bottom: 4, containLabel: true },
        tooltip: { ...b.tooltip, trigger: 'item', formatter: (p: { name: string; value: number }) => `${p.name}<br/><b>${metric === 'fill' ? fmtNum(p.value, 2) + '%' : fmt(p.value)}</b>` },
        xAxis: { type: 'value', min, max: metric === 'fill' ? 100 : undefined, show: false },
        yAxis: {
          type: 'category',
          data: data.map((r) => METHOD_SHORT[r.id]),
          ...b.axis,
          axisLine: { show: false },
          axisLabel: { color: t.ink2, fontSize: 12, fontWeight: 500 },
        },
        series: [
          {
            type: 'bar',
            barWidth: 14,
            data: data.map((r, i) => ({ value: vals[i], itemStyle: { color: methodColor(t, r.id), borderRadius: [0, 4, 4, 0] } })),
            label: {
              show: true,
              position: 'right',
              color: t.ink,
              fontSize: 12,
              fontFamily: 'IBM Plex Sans',
              formatter: (p: { value: number }) => (metric === 'fill' ? `${fmtNum(p.value, 2)}%` : fmt(p.value)),
            },
          },
        ],
      };
    },
    [rows, metric, fmt],
  );
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <h3 className="text-[14px] font-semibold text-ink">{title}</h3>
        <span className="text-[11.5px] text-ink-3">{lowerBetter ? 'lower is better' : 'higher is better'}</span>
      </div>
      <Chart option={option} height={190} label={`${title} by method. ${rows.map((r) => `${r.label}: ${metric === 'fill' ? fmtPct(r.fill) : fmt(r[metric])}`).join('; ')}`} />
    </div>
  );
}

export function Overview() {
  const { period, go, holidayPolicy } = useApp();
  const sum = useJSON<Summary>('summary.json');
  const cal = useJSON<CalendarFile>('calendar.json');

  const daily = useCallback(
    (t: Tokens) => {
      const b = base(t);
      const P = cal.data!.periods[period];
      const best = sum.data!.periods[period].best_simple;
      const x = P.days.map((d) => fmtDay(d.date));
      const isHoli = holidayPolicy && period === 'test';
      const s = (id: MethodId) => P.days.map((d) => {
        if (id === 'store_ilp' && isHoli && d.holiday && d.holiday.fill != null) {
          return +(d.holiday.fill * 100).toFixed(2);
        }
        return d.methods[id][0] == null ? null : +(d.methods[id][0]! * 100).toFixed(2);
      });
      const sBaseIlp = P.days.map((d) => (d.methods['store_ilp'][0] == null ? null : +(d.methods['store_ilp'][0]! * 100).toFixed(2)));
      const closed = P.days.filter((d) => d.closed).map((d) => fmtDay(d.date));
      return {
        textStyle: b.textStyle,
        grid: { left: 8, right: 16, top: 36, bottom: 8, containLabel: true },
        legend: { top: 0, left: 0, itemWidth: 14, itemHeight: 3, textStyle: { color: t.ink2, fontSize: 12 }, icon: 'rect' },
        tooltip: { ...b.tooltip, trigger: 'axis', valueFormatter: (v: number | null) => (v == null ? 'closed' : `${fmtNum(v, 2)}%`) },
        xAxis: { type: 'category', data: x, ...b.axis, axisLabel: { ...b.axis.axisLabel, interval: 6 } },
        yAxis: { type: 'value', min: (v: { min: number }) => Math.floor(v.min - 2), max: 100, ...b.axis, axisLine: { show: false }, axisLabel: { ...b.axis.axisLabel, formatter: '{value}%' } },
        series: isHoli
          ? [
              {
                name: 'Store ILP (Lead-Aware Holiday Model)',
                type: 'line', data: s('store_ilp'), symbol: 'circle', symbolSize: 5, showSymbol: false,
                lineStyle: { width: 2.5, color: '#f59e0b' }, itemStyle: { color: '#f59e0b' }, connectNulls: false,
                markArea: closed.length ? { silent: true, itemStyle: { color: t.surface2 }, label: { show: true, color: t.ink3, fontSize: 11, position: 'insideTop', formatter: 'closed' }, data: closed.map((c) => [{ xAxis: c }, { xAxis: c }]) } : undefined,
              },
              {
                name: 'Store ILP (Baseline Uncontrolled)',
                type: 'line', data: sBaseIlp, symbol: 'emptyCircle', symbolSize: 4, showSymbol: false,
                lineStyle: { width: 1.5, color: t.s1, type: 'dashed' }, itemStyle: { color: t.s1 }, connectNulls: false,
              },
            ]
          : [
              {
                name: METHOD_SHORT.store_ilp,
                type: 'line', data: s('store_ilp'), symbol: 'circle', symbolSize: 5, showSymbol: false,
                lineStyle: { width: 2.5, color: t.s1 }, itemStyle: { color: t.s1 }, connectNulls: false,
                markArea: closed.length ? { silent: true, itemStyle: { color: t.surface2 }, label: { show: true, color: t.ink3, fontSize: 11, position: 'insideTop', formatter: 'closed' }, data: closed.map((c) => [{ xAxis: c }, { xAxis: c }]) } : undefined,
              },
              {
                name: `${METHOD_SHORT[best]} (best simple rule)`,
                type: 'line', data: s(best), symbol: 'circle', symbolSize: 5, showSymbol: false,
                lineStyle: { width: 2, color: t.s2 }, itemStyle: { color: t.s2 },
              },
            ],
      };
    },
    [cal.data, sum.data, period, holidayPolicy],
  );

  if (sum.error || cal.error) return <ErrorState message={(sum.error || cal.error)!} retry={() => { sum.retry(); cal.retry(); }} />;
  if (!sum.data || !cal.data) return <Loading what="overview" />;

  const P = sum.data.periods[period];
  const ilp = P.methods.find((m) => m.id === 'store_ilp')!;
  const best = P.methods.find((m) => m.id === P.best_simple)!;
  const allPassed = P.methods.every((m) => m.audit_passed && m.rules_passed);

  const holidayRow = sum.data?.layers?.rows?.find((r) => r.id === 'holiday');
  const isHolidayActive = holidayPolicy && period === 'test' && !!holidayRow;

  const activeFill = isHolidayActive ? holidayRow!.fill : ilp.fill * 100;
  const activeWaste = isHolidayActive ? holidayRow!.waste : ilp.waste;
  const activeTotal = isHolidayActive ? holidayRow!.total : ilp.total;
  const activeLost = isHolidayActive ? holidayRow!.lost : ilp.lost;

  return (
    <div className="space-y-6">
      <Spotlight className="rounded-3xl bg-surface shadow-card">
        <div className="grid gap-6 p-5 sm:p-8 lg:grid-cols-[1.25fr_1fr] lg:gap-10">
          <div>
            <TextEffect as="h1" className="display text-[2rem] font-semibold leading-[1.08] text-ink sm:text-[2.75rem]">
              Every morning: how many cartons should each store get?
            </TextEffect>
            <p className="prose-measure mt-4 text-[15.5px] leading-relaxed text-ink-2">
              Send too few and shelves go empty and sales are lost. Send too many and money sits in stock and fresh food
              spoils. Shelfwise makes that call for 10 stores × 120 food products, using real daily sales from the Walmart M5
              dataset, and replays it day by day against what really sold.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              <button type="button" onClick={() => go('workflow')} className="rounded-full bg-accent px-4 py-2 text-[13.5px] font-semibold text-accent-ink transition-transform hover:-translate-y-px">
                Watch the replay calendar
              </button>
              <button type="button" onClick={() => go('how')} className="rounded-full bg-surface-2 px-4 py-2 text-[13.5px] font-semibold text-ink transition-colors hover:bg-surface-3">
                How it works
              </button>
            </div>
          </div>
          <dl className="grid content-start gap-4 text-[14px]">
            <div>
              <dt className="font-semibold text-ink">1 · A forecaster that gives ranges, not one number</dt>
              <dd className="mt-1 text-ink-2">Gradient-boosted trees predict four lines for today and tomorrow: q50, q80, q90, q95. “q90 = 14” means sales stay at or below 14 on about 9 days in 10.</dd>
            </div>
            <div>
              <dt className="font-semibold text-ink">2 · A target per product, then a small ILP per store</dt>
              <dd className="mt-1 text-ink-2">Long-life items are stocked to q95; perishables to a cost-based level between q50 and q80, capped so they sell before spoiling. An integer program turns targets into whole cartons under truck, shelf and budget limits.</dd>
            </div>
            <div>
              <dt className="font-semibold text-ink">3 · Shadow prices from the LP</dt>
              <dd className="mt-1 text-ink-2">Fix the whole-number choices, re-solve as a linear program, and read what one more unit of each limit would save.</dd>
            </div>
          </dl>
        </div>
      </Spotlight>

      <section aria-label={`Main model results, ${P.label}`}>
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="display text-[1.35rem] font-semibold text-ink">
            Store ILP · {P.label} <span className="text-[14px] font-normal text-ink-3">{P.range}</span>
          </h2>
          <div className="flex items-center gap-2.5">
            <HolidayToggle compact />
            <span className="text-[12.5px] text-ink-3">layers {isHolidayActive ? `${P.store_ilp_layers} + holiday policy` : P.store_ilp_layers}</span>
          </div>
        </div>
        <AnimatedGroup className="grid grid-cols-1 gap-3 min-[420px]:grid-cols-2 lg:grid-cols-4" key={`${period}-${isHolidayActive}`}>
          <Kpi label="Fill rate" value={activeFill} format={(x) => `${fmtNum(x, 2)}%`} sub={<>units sold ÷ units wanted. Best simple rule ({METHOD_SHORT[best.id]}): {fmtPct(best.fill)}</>} />
          <Kpi label="Waste" value={activeWaste} format={(x) => `${fmtInt(x)} units`} sub={<>{isHolidayActive ? 'Holiday policy active: -1,139 spoiled units saved vs 8,043 baseline' : `expected spoiled units. Best simple rule: ${fmtInt(best.waste)} units`}</>} />
          <Kpi label="Net cost + lost-sale penalty" value={activeTotal} format={(x) => fmtMoney(x)} sub={<>{fmtMoney(ilp.net_cost)} net cost + {fmtMoney(ilp.penalty)} penalty for {fmtInt(activeLost)} lost units</>} />
          <Kpi label="vs best simple rule" value={ilp.vs_best_simple} format={(x) => fmtSigned(x, 1)} sub={<>{METHOD_SHORT[best.id]}: {fmtMoney(best.total)} on the same measure</>} />
        </AnimatedGroup>
        <div className="mt-3 flex flex-wrap items-start gap-x-6 gap-y-2">
          {allPassed && (
            <p className="flex items-center gap-1.5 text-[13px] text-good-ink">
              <CheckCircle2 size={15} aria-hidden /> All 4 methods passed every daily rule check and the independent audit.
            </p>
          )}
          <p className="flex items-start gap-1.5 text-[13px] text-ink-3">
            <Info size={15} className="mt-0.5 shrink-0" aria-hidden /> {P.note}
          </p>
        </div>
      </section>

      <div className="rounded-2xl border border-line bg-surface p-4 shadow-card sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className="rounded-xl bg-amber-500/15 p-2 text-amber-600 dark:text-amber-400">
              <Flame size={20} aria-hidden />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-[15px] font-semibold text-ink">Holiday Surge & Valley Policy Layer</h3>
                <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${isHolidayActive ? 'bg-good-soft text-good-ink' : 'bg-surface-3 text-ink-3'}`}>
                  {isHolidayActive ? '● Plan Active: -80.6% Thanksgiving Waste' : '○ Plan Bypassed'}
                </span>
              </div>
              <p className="mt-1 max-w-[80ch] text-[13px] text-ink-2">
                Solves festival demand whipsaws: pre-holiday quantile surge boosts (M_surge) on 28 festive products protect fill, while post-holiday maintenance dampening (D_post) prevents dumpster waste. Thanksgiving week waste dropped from 779 down to 151 units.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <HolidayToggle />
            <button
              type="button"
              onClick={() => go('shadow')}
              className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3.5 py-1.5 text-[12.5px] font-semibold text-ink hover:bg-surface-3 transition-colors cursor-pointer"
            >
              <span>Explore Simulator</span>
              <ArrowRight size={14} aria-hidden />
            </button>
          </div>
        </div>
      </div>

      <InView>
        <Card label="Method comparison">
          <SectionHead title="Four ways to order, same days, same shelves">
            Three simple rules a store manager might use and the store ILP. Every method starts from the
            same stock and is scored on the same sales.
          </SectionHead>
          <div className="grid gap-x-10 gap-y-6 md:grid-cols-2">
            <MetricBars rows={P.methods} metric="total" title="Net cost + lost-sale penalty" fmt={(x) => fmtMoney(x)} lowerBetter />
            <MetricBars rows={P.methods} metric="fill" title="Fill rate" fmt={(x) => fmtNum(x, 2)} lowerBetter={false} />
            <MetricBars rows={P.methods} metric="waste" title="Waste (expected spoiled units)" fmt={(x) => fmtInt(x)} lowerBetter />
            <MetricBars rows={P.methods} metric="stock" title="Average store stock value" fmt={(x) => fmtMoney(x)} lowerBetter />
          </div>
          <Note className="mt-4">
            {(() => {
              const tf = P.methods.find((m) => m.id === 'topup_forecast')!;
              return `Top-up to our forecast sells the most (${fmtPct(tf.fill)} fill) but throws away ${fmtNum(tf.waste / ilp.waste, 1)} times as much as the store ILP, whose shelves hold ${fmtPct(Math.abs(ilp.stock / tf.stock - 1), 0)} ${ilp.stock >= tf.stock ? 'more' : 'less'} stock value. `;
            })()}
            The store ILP trades a little fill for much less waste and lands lowest on total cost. Total ={' '}
            {sum.data.totals_definition}
          </Note>
          <Source>{ilp.source}{P.methods.some((m) => m.source !== ilp.source) ? `; simple rules: ${P.methods.find((m) => m.source !== ilp.source)!.source}` : ''}</Source>
        </Card>
      </InView>

      <InView>
        <Card label="Daily fill rate">
          <SectionHead
            title="Day by day comparator"
            right={<HolidayToggle compact />}
          >
            Daily fill rate of the store ILP against the best simple rule for this period ({METHOD_SHORT[P.best_simple]}).
            {period === 'test' && ' The deepest dips are around Thanksgiving (25 Nov, the busiest day of the window, and 26 Nov) and Christmas Eve, when no order could be placed because nothing is delivered on 25 Dec.'}
          </SectionHead>
          <Chart option={daily} height={300} label={`Daily fill rate, store ILP vs ${METHOD_SHORT[P.best_simple]}, ${P.label}`} />
          <Source>store_ilp/daily_metrics.csv of each method, via calendar.json</Source>
        </Card>
      </InView>

      <InView>
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <h3 className="text-[15px] font-semibold text-ink">The problem</h3>
            <p className="mt-2 text-[14px] text-ink-2">10 Walmart stores in California, Texas and Wisconsin, 120 food products (20 perishable). Each morning each store orders whole cartons that arrive the next day. Demand is uncertain, cartons are lumpy, perishables spoil.</p>
          </Card>
          <Card>
            <h3 className="text-[15px] font-semibold text-ink">How it is tested</h3>
            <p className="mt-2 text-[14px] text-ink-2">A replay: every day each method sees only the past, places its order, and the real sales of that day are then played against its shelves. 8 test weeks around Thanksgiving and Christmas, then two sealed holdout blocks of 7 weeks, run once with frozen code.</p>
          </Card>
          <Card>
            <h3 className="text-[15px] font-semibold text-ink">Units</h3>
            <p className="mt-2 text-[14px] text-ink-2">{sum.data.money.note}</p>
          </Card>
        </div>
      </InView>
    </div>
  );
}
