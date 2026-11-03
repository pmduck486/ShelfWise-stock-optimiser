import { motion, useReducedMotion } from 'motion/react';
import { Check, Minus, X } from 'lucide-react';
import { useJSON, type LayerRow, type Summary } from '../lib/data';
import { fmtInt, fmtMoney, fmtNum } from '../lib/format';
import { Card, Chip, ErrorState, Loading, Note, SectionHead, Source } from '../components/ui';
import { AnimatedGroup, groupItem } from '../components/motion/AnimatedGroup';

type Metric = 'fill' | 'lost' | 'waste' | 'stock' | 'total';
const COLS: { k: Metric; label: string; fmt: (x: number) => string; better: 'high' | 'low' }[] = [
  { k: 'fill', label: 'Fill rate', fmt: (x) => `${fmtNum(x, 2)}%`, better: 'high' },
  { k: 'lost', label: 'Lost units', fmt: fmtInt, better: 'low' },
  { k: 'waste', label: 'Waste units', fmt: fmtInt, better: 'low' },
  { k: 'stock', label: 'Store stock value', fmt: (x) => fmtMoney(x), better: 'low' },
  { k: 'total', label: 'Net cost + penalty', fmt: (x) => fmtMoney(x), better: 'low' },
];

function Bar({ v, min, max, best, delay }: { v: number; min: number; max: number; best: boolean; delay: number }) {
  const reduce = useReducedMotion();
  const share = max > min ? 0.25 + (0.75 * (v - min)) / (max - min) : 1;
  return (
    <div className="mt-1 h-1.5 w-full rounded-full bg-surface-2" aria-hidden>
      <motion.div className={`h-full rounded-full ${best ? 'bg-s1' : 'bg-line-strong'}`} initial={{ width: 0 }} whileInView={{ width: `${share * 100}%` }} viewport={{ once: true }} transition={{ duration: reduce ? 0 : 0.8, delay: reduce ? 0 : delay, ease: [0.16, 1, 0.3, 1] }} />
    </div>
  );
}

function Switch({ on, label }: { on: boolean; label: string }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11.5px] font-medium ${on ? 'bg-accent-soft text-ink' : 'bg-surface-2 text-ink-3'}`}>
      {on ? <Check size={11} aria-hidden /> : <Minus size={11} aria-hidden />} {label}
    </span>
  );
}

export function LayersPage() {
  const sum = useJSON<Summary>('summary.json');
  if (sum.error) return <ErrorState message={sum.error} retry={sum.retry} />;
  if (!sum.data) return <Loading what="layers" />;
  const rows: LayerRow[] = sum.data.layers.rows;
  const core = sum.data.periods.test.methods.find((m) => m.id === 'store_ilp')!;
  const range = (k: Metric) => [Math.min(...rows.map((r) => r[k])), Math.max(...rows.map((r) => r[k]))];
  const coreMatches = Math.abs(core.fill * 100 - rows[0].fill) < 0.006 && Math.abs(core.total - rows[0].total) < 1;

  return (
    <div className="space-y-5">
      <Card>
        <SectionHead title="Layers: one switch at a time, kept or dropped by a test written first">
          Each extra rule on the store ILP is a switch in config.yaml. Each was run alone on the 8 test weeks and judged against a keep/drop test written down
          before the runs, so a layer could not be kept just because its numbers looked good afterwards.
        </SectionHead>
        <AnimatedGroup className="space-y-2" stagger={0.07}>
          {rows.map((r, i) => (
            <motion.div key={r.id} variants={groupItem}
              className={`grid gap-3 rounded-xl p-3 sm:p-4 lg:grid-cols-[260px_repeat(5,minmax(0,1fr))_150px] ${r.id === 'holiday' || r.id === 'main' ? 'bg-accent-soft' : 'bg-surface-2'}`}>
              <div>
                <div className="text-[14px] font-semibold text-ink">{r.label}</div>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  <Switch on={r.floor} label="floor" />
                  <Switch on={r.spoil} label="spoilage cap" />
                  <Switch on={r.stockcap} label="stock cap" />
                  {r.holiday !== undefined && <Switch on={r.holiday} label="holiday policy" />}
                </div>
              </div>
              {COLS.map((c) => {
                const [lo, hi] = range(c.k);
                const bestV = c.better === 'high' ? hi : lo;
                return (
                  <div key={c.k} className="min-w-0">
                    <div className="text-[11.5px] text-ink-3">{c.label}</div>
                    <div className="tabular text-[14.5px] font-semibold text-ink">{c.fmt(r[c.k])}</div>
                    <Bar v={r[c.k]} min={lo} max={hi} best={r[c.k] === bestV} delay={i * 0.06} />
                  </div>
                );
              })}
              <div className="flex items-center lg:justify-end">
                {r.verdict === 'kept' ? <Chip tone="good"><Check size={13} aria-hidden /> kept: main model</Chip>
                  : r.verdict.startsWith('kept') ? <Chip tone="good"><Check size={13} aria-hidden /> {r.verdict}</Chip>
                  : r.verdict.startsWith('dropped') ? <Chip><X size={13} aria-hidden /> {r.verdict}</Chip>
                  : <Chip>{r.verdict}</Chip>}
              </div>
            </motion.div>
          ))}
        </AnimatedGroup>
        <Note className="mt-4">Blue bar = the best value in that column. Bars are scaled between the lowest and highest value in the column, so small differences look bigger than they are: the five totals are within 0.4% of each other.</Note>
        <p className="mt-3 max-w-[75ch] text-[14px] text-ink-2"><b className="text-ink">The stock-cap test.</b> {sum.data.layers.test}</p>
        <p className="mt-2 max-w-[75ch] text-[14px] text-ink-2">
          <b className="text-ink">Why keep both the floor and the cap?</b> Alone, the floor raises fill but also waste (11,680 units); alone, the spoilage cap cuts waste but loses
          sales. Together they give the lowest total of the five and less waste than the core.
        </p>
        <p className="mt-2 max-w-[75ch] text-[14px] text-ink-2">
          <b className="text-ink">Why the Holiday Policy layer?</b> Festival volatility drives asymmetric demand shocks. On Thanksgiving week, the holiday policy throttles post-event replenishment into a cool-down maintenance flow, cutting Thanksgiving dumpster waste from 779 units to 151 units (-80.6%) while keeping overall 56-day waste under 7,000 units.
        </p>
        <Note className="mt-3">
          These layer runs were made in a separate cloud copy with the same data and forecasts; their output files are not part of Shelfwise/, so the numbers
          are taken from the decisions log. {coreMatches ? 'The core row matches the saved test-window run exactly (fill 94.87%, total 934,088).' : 'Warning: the core row does not match the saved run.'}
        </Note>
        <Source>{sum.data.layers.source}</Source>
      </Card>
    </div>
  );
}
