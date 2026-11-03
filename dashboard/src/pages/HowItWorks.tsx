import { useState, type ReactNode } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { Database, LineChart, Target, Boxes, Scale, ClipboardCheck } from 'lucide-react';
import { useJSON, type ItemsFile, type Summary } from '../lib/data';
import { fmtMoney, fmtNum, fmtPct } from '../lib/format';
import { Card, ErrorState, Loading, Note } from '../components/ui';
import { TextEffect } from '../components/motion/TextEffect';

const EXAMPLE = 'FOODS_3_090';

export function HowItWorks() {
  const items = useJSON<ItemsFile>('items.json');
  const sum = useJSON<Summary>('summary.json');
  const [open, setOpen] = useState(0);
  const reduce = useReducedMotion();
  if (items.error || sum.error) return <ErrorState message={(items.error || sum.error)!} retry={() => { items.retry(); sum.retry(); }} />;
  if (!items.data || !sum.data) return <Loading what="pipeline" />;

  const I = items.data;
  const i = I.items.findIndex((x) => x.id === EXAMPLE);
  const it = I.items[i];
  const p = I.params.test.items[i];
  const P = I.params.test;
  const cu = P.lost_sale_penalty_share * p.price;
  const co = p.buy * (1 + P.waste_disposal_share);
  const ilp = sum.data.periods.test.methods.find((m) => m.id === 'store_ilp')!;

  const NODES: { icon: typeof Database; title: string; short: string; body: ReactNode }[] = [
    {
      icon: Database, title: 'Data', short: 'Walmart M5 daily sales',
      body: (
        <>
          <p>Real daily unit sales from the Walmart M5 dataset (Kaggle): 10 stores in California, Texas and Wisconsin, the 120 best-selling food products, 20 of them treated as perishable with a shelf life of 2 to 7 days. Prices come from M5's price file. Data is split by date: training from 2013, two validation folds (Nov–Dec 2014 and Jul–Oct 2015), an 8-week test window (1 Nov to 26 Dec 2015) and a sealed holdout after it. The final forecaster was retrained on everything before 1 Nov 2015 and frozen.</p>
          <p className="mt-2"><b className="text-ink">Milk example:</b> our stand-in for milk is {EXAMPLE}, a perishable food sold at {fmtMoney(p.price, 2)} a unit with a {it.shelf_life}-day shelf life, delivered in cartons of {it.upc}; it sold about {fmtNum(it.avg_daily, 0)} units per store per day before the test window.</p>
        </>
      ),
    },
    {
      icon: LineChart, title: 'Forecaster', short: 'q50, q80, q90, q95',
      body: (
        <>
          <p><b className="text-ink">Quantile gradient boosting:</b> many small decision trees, each correcting the last, trained to predict a <i>quantile</i> rather than an average. A quantile line q80 is the level sales stay at or below on 80% of days. Four models give four lines for today and tomorrow, per store and product.</p>
          <p className="mt-2">Inputs are past sales (always at least 2 days old, so the model never sees the day it predicts), weekday, holidays, food-stamp days and prices. It beats both baselines on every line on both validation folds.</p>
          <p className="mt-2"><b className="text-ink">Milk:</b> rather than "tomorrow 40 cartons", the forecaster says something like "half the time below q50, 9 times in 10 below q90", which is exactly what a stocking decision needs.</p>
        </>
      ),
    },
    {
      icon: Target, title: 'Targets', short: 'newsvendor level per product',
      body: (
        <>
          <p><b className="text-ink">Newsvendor rule:</b> stock up to the level where one more unit is as likely to be wasted as to be sold, weighted by what each mistake costs. With Cu the cost of a missed sale and Co the cost of a wasted unit, the best level is the quantile Cu / (Cu + Co).</p>
          <p className="mt-2"><b className="text-ink">Milk:</b> Cu = {P.lost_sale_penalty_share} × price = {fmtMoney(cu, 2)}; Co = buy cost × {1 + P.waste_disposal_share} = {fmtMoney(co, 2)}; level = {fmtNum(cu, 2)} / ({fmtNum(cu, 2)} + {fmtNum(co, 2)}) = <b className="text-ink">{fmtNum(p.level, 3)}</b>, between the q50 and q80 lines. A spoilage cap keeps stock below the median sales over its {it.shelf_life}-day life. Long-life products waste nothing, so their level is above 0.99 and they are stocked to q95.</p>
        </>
      ),
    },
    {
      icon: Boxes, title: 'One-day ILP', short: 'whole cartons per store',
      body: (
        <>
          <p><b className="text-ink">Integer linear program (ILP):</b> a linear cost to minimise with some variables forced to whole numbers. One small ILP per store each morning chooses how many cartons of each product to order (whole numbers), whether each order line is placed (yes/no) and how many trucks go (whole number).</p>
          <p className="mt-2">It minimises Cu × units short of the target + Co × units over it + {fmtMoney(P.order_line_cost, 2)} per order line + {fmtMoney(P.truck_trip_cost, 0)} per truck, subject to truck capacity ({P.truck_capacity_cases} cartons), free shelf space and a daily budget. In the main model two layers join: a soft service floor (at least 90% of the target) and the spoilage cap as a hard constraint.</p>
          <p className="mt-2"><b className="text-ink">Milk:</b> the target might be 52 units but a carton holds {it.upc}: one carton falls short, two overshoot. The ILP prices both and takes the cheaper, unless the spoilage cap forbids the second carton. Each daily solve has up to {ilp.n_vars ? ilp.n_vars.toLocaleString('en-US') : '–'} variables ({ilp.n_int?.toLocaleString('en-US')} whole-number) across the 10 stores in the saved test run; CBC solves them in about 2.5 s a day.</p>
        </>
      ),
    },
    {
      icon: Scale, title: 'LP duals', short: 'shadow prices',
      body: (
        <>
          <p><b className="text-ink">Duality:</b> every linear program has a twin whose variables price the constraints. After the ILP, the trucks and order lines are fixed at its choices and the cartons are relaxed to fractions, giving an LP. Its dual values are the <b className="text-ink">shadow prices</b>: how much the day's cost would fall with one more unit of each limit.</p>
          <p className="mt-2">Result: truck, space and budget prices are 0 on every store-day at the project's settings; the limits have room to spare. They only become positive when the limits are cut hard in the what-ifs (for example shelf space × 0.3). The service floor has a price where the 90% promise is broken, often because a whole carton of milk would exceed its spoilage cap.</p>
        </>
      ),
    },
    {
      icon: ClipboardCheck, title: 'Replay & scorecard', short: 'day by day on real sales',
      body: (
        <>
          <p><b className="text-ink">Replay:</b> start every method from the same stock on 1 Nov 2015. Each morning the method sees only the past and orders; its cartons arrive the next day; that day's real sales are played against its shelves. Lost sales, expected waste, stock and costs are recorded, and every day passes rule checks and an independent audit.</p>
          <p className="mt-2">Scorecard: fill rate, lost units, waste, stock value and net cost + lost-sale penalty. Over the 8 test weeks the store ILP reaches {fmtPct(ilp.fill)} fill and {fmtNum(ilp.vs_best_simple, 1)}% total cost against the best simple rule; the two holdout blocks were then run once with frozen code.</p>
        </>
      ),
    },
  ];

  return (
    <div className="space-y-5">
      <Card>
        <TextEffect as="h1" className="display text-[1.8rem] font-semibold leading-tight text-ink sm:text-[2.2rem]">From sales history to tomorrow's cartons</TextEffect>
        <p className="prose-measure mt-2 text-[15px] text-ink-2">Six stages run every morning of the replay. Pick a stage to read what it does, in plain words, with the milk example carried through.</p>

        <div className="relative mt-6">
          <svg className="pointer-events-none absolute left-0 top-[34px] hidden h-[2px] w-full lg:block" aria-hidden preserveAspectRatio="none" viewBox="0 0 100 2">
            <motion.line x1="4" y1="1" x2="96" y2="1" stroke="var(--line-strong)" strokeWidth="2" strokeDasharray="1.2 1.2" vectorEffect="non-scaling-stroke"
              initial={{ pathLength: reduce ? 1 : 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.4, ease: 'easeInOut' }} />
          </svg>
          <ol className="relative grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
            {NODES.map((n, k) => {
              const Icon = n.icon;
              const on = k === open;
              return (
                <li key={n.title}>
                  <motion.button type="button" onClick={() => setOpen(k)} aria-expanded={on} aria-controls="stage-panel"
                    initial={reduce ? false : { opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: reduce ? 0 : 0.15 + k * 0.12, duration: 0.45 }}
                    className={`relative flex w-full flex-col items-center gap-1.5 rounded-2xl p-3 text-center transition-colors ${on ? 'text-ink' : 'text-ink-2 hover:bg-surface-2'}`}>
                    {on && <motion.span layoutId="stage-hl" className="absolute inset-0 rounded-2xl bg-accent-soft" transition={{ type: 'spring', bounce: 0.15, duration: 0.45 }} />}
                    <span className={`relative grid size-12 place-items-center rounded-full border-2 ${on ? 'border-accent bg-surface text-accent' : 'border-line-strong bg-surface text-ink-3'}`}>
                      <Icon size={20} aria-hidden />
                    </span>
                    <span className="relative text-[13.5px] font-semibold">{k + 1}. {n.title}</span>
                    <span className="relative text-[12px] leading-tight text-ink-3">{n.short}</span>
                  </motion.button>
                </li>
              );
            })}
          </ol>
        </div>

        <div id="stage-panel" className="mt-4 overflow-hidden rounded-2xl bg-surface-2" aria-live="polite">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={open} className="prose-measure p-5 text-[14.5px] leading-relaxed text-ink-2"
              initial={reduce ? { opacity: 0 } : { opacity: 0, y: 10, filter: 'blur(4px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={reduce ? { opacity: 0 } : { opacity: 0, y: -8, filter: 'blur(4px)' }} transition={{ duration: reduce ? 0.01 : 0.28 }}>
              <h2 className="display mb-2 text-[1.25rem] font-semibold text-ink">{open + 1}. {NODES[open].title}</h2>
              {NODES[open].body}
            </motion.div>
          </AnimatePresence>
        </div>
        <Note className="mt-3">The milk numbers use {EXAMPLE}'s settings in the test-window run (items.json). The "target might be 52 units" line is an illustrative figure for explaining carton rounding, not a result.</Note>
      </Card>
    </div>
  );
}
