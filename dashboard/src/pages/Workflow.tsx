import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { BrainCircuit, CalendarX2, Flag, PackageX, Pause, Play, RotateCcw, ShieldCheck, Truck, LineChart } from 'lucide-react';
import { useApp } from '../lib/state';
import { useJSON, type CalendarDay, type CalendarFile } from '../lib/data';
import { DOW_LABELS, fmtDate, fmtInt, fmtNum, fmtPct, parseISO } from '../lib/format';
import { Card, ErrorState, Loading, Note, SectionHead } from '../components/ui';
import { MorphingDialog } from '../components/motion/MorphingDialog';
import { AnimatedNumber } from '../components/motion/AnimatedNumber';
import { HolidayToggle } from '../components/HolidayToggle';
import { DayDetail } from './DayDetail';

const BINS = [
  { lo: -1, label: 'below 88%' },
  { lo: 0.88, label: '88–92%' },
  { lo: 0.92, label: '92–95%' },
  { lo: 0.95, label: '95–97%' },
  { lo: 0.97, label: '97% and up' },
];
const binOf = (f: number) => BINS.reduce((acc, b, i) => (f >= b.lo ? i : acc), 0) + 1;

const LOOP = [
  { icon: LineChart, title: 'Forecast', text: 'q50 to q95 for today and tomorrow, from data at least 2 days old' },
  { icon: BrainCircuit, title: 'Decide', text: 'targets, then one small ILP per store picks whole cartons' },
  { icon: Truck, title: 'Deliver & sell', text: "tomorrow's cartons ship; today's real sales hit the shelf" },
  { icon: ShieldCheck, title: 'Learn', text: 'actual sales join the history for the next forecast' },
];

function getHolidayPhase(dateStr: string): { name: string; icon: string; badge: string } | null {
  if (dateStr >= '2015-11-23' && dateStr <= '2015-11-25') return { name: 'Holiday Surge (M_surge)', icon: '🚀', badge: 'Surge' };
  if (dateStr === '2015-11-26') return { name: 'Thanksgiving Day', icon: '🦃', badge: 'Holiday' };
  if (dateStr >= '2015-11-27' && dateStr <= '2015-11-28') return { name: 'Post-Holiday Throttle (D_post)', icon: '❄️', badge: 'Throttle' };
  if (dateStr === '2015-12-24') return { name: 'Christmas Eve Surge', icon: '🎄', badge: 'Surge' };
  if (dateStr === '2015-12-25') return { name: 'Christmas Day (Closed)', icon: '🎁', badge: 'Closed' };
  if (dateStr === '2015-12-26') return { name: 'Post-Christmas Cool-Down', icon: '❄️', badge: 'Cool-Down' };
  return null;
}

function DayCell({ d, period, holidayOn, active, onOpen, onHover }: { d: CalendarDay; period: string; holidayOn: boolean; active: boolean; onOpen: () => void; onHover: (on: boolean, el: HTMLElement | null) => void }) {
  const p = parseISO(d.date);
  const hp = holidayOn ? getHolidayPhase(d.date) : null;
  const hMetrics = holidayOn && d.holiday ? d.holiday : null;
  const fillVal = hMetrics ? hMetrics.fill : d.fill;
  const b = fillVal == null ? 0 : binOf(fillVal);
  const style = d.closed || fillVal == null ? undefined : { background: `var(--cal-${b})`, color: `var(--cal-ink-${b})` };
  const label = `${fmtDate(d.date, true)}${d.events.length ? ', ' + d.events.map((e) => e.name).join(', ') : ''}${hp ? `, ${hp.name}` : ''}${d.closed ? ', stores closed' : `, fill rate ${fmtPct(fillVal, 1)}`}${d.floor_n ? `, service floor priced on ${d.floor_n} items` : ''}${!d.can_order ? ', no order possible' : ''}. Open the day's decision.`;
  return (
    <motion.button
      type="button"
      layoutId={`day-${period}-${d.date}`}
      onClick={onOpen}
      onMouseEnter={(e) => onHover(true, e.currentTarget)}
      onMouseLeave={() => onHover(false, null)}
      onFocus={(e) => onHover(true, e.currentTarget)}
      onBlur={() => onHover(false, null)}
      aria-label={label}
      className={`relative flex aspect-square min-w-0 flex-col justify-between overflow-hidden rounded-lg p-1 text-left transition-all duration-300 sm:aspect-[1.15] sm:rounded-xl sm:p-2 ${d.closed ? 'hatch bg-surface-2 text-ink-2' : ''} ${active ? 'ring-2 ring-ink ring-offset-2 ring-offset-surface' : 'hover:ring-2 hover:ring-line-strong'}`}
      style={{ ...style, borderRadius: 12 }}
    >
      <div className="flex items-start justify-between gap-0.5">
        <span className="tabular text-[11px] font-semibold leading-none sm:text-[13px]">{p.d === 1 ? `${p.d} ${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][p.m - 1]}` : p.d}</span>
        <span className="flex items-center gap-0.5">
          {hp && <span className="text-[10px] sm:text-[12px]" title={hp.name}>{hp.icon}</span>}
          {d.events.length > 0 && !hp && <Flag size={11} aria-hidden className="sm:size-[13px]" />}
          {!d.can_order && <PackageX size={11} aria-hidden className="sm:size-[13px]" />}
          {d.closed && <CalendarX2 size={11} aria-hidden className="sm:size-[13px]" />}
        </span>
      </div>
      <div className="flex items-end justify-between gap-1">
        <span className="tabular hidden text-[12px] font-medium leading-none opacity-90 min-[480px]:inline sm:text-[13px]">{d.closed ? 'closed' : fmtPct(fillVal, 1)}</span>
        {d.floor_n > 0 && (
          <span className="tabular hidden items-center gap-0.5 rounded-full bg-black/15 px-1.5 py-px text-[10.5px] font-semibold leading-tight sm:inline-flex" title="service-floor prices">
            <ShieldCheck size={10} aria-hidden />{d.floor_n}
          </span>
        )}
      </div>
    </motion.button>
  );
}

export function Workflow() {
  const { period, day, setDay, holidayPolicy } = useApp();
  const cal = useJSON<CalendarFile>('calendar.json');
  const reduce = useReducedMotion();
  const [cursor, setCursor] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [phase, setPhase] = useState(0);
  const [hover, setHover] = useState<{ d: CalendarDay; x: number; y: number; w: number } | null>(null);
  const [open, setOpen] = useState(false);
  const gridRef = useRef<HTMLDivElement>(null);

  const days = cal.data?.periods[period].days ?? [];
  const n = days.length;

  useEffect(() => {
    setCursor(-1);
    setPlaying(false);
  }, [period]);

  useEffect(() => {
    if (!playing) return;
    const stepMs = reduce ? 1400 : 1000;
    const id = window.setInterval(() => {
      setPhase((ph) => {
        if (ph < 3) return ph + 1;
        setCursor((c) => {
          if (c + 1 >= n) {
            setPlaying(false);
            return c;
          }
          return c + 1;
        });
        return 0;
      });
    }, stepMs / 4);
    return () => window.clearInterval(id);
  }, [playing, n, reduce]);

  const totals = useMemo(() => {
    const upto = days.slice(0, cursor + 1);
    const getD = (x: CalendarDay) => (holidayPolicy && period === 'test' && x.holiday ? x.holiday : x);
    const dem = upto.reduce((a, x) => a + (x.closed ? 0 : x.demand), 0);
    const srv = upto.reduce((a, x) => a + (x.closed ? 0 : getD(x).served), 0);
    return {
      dem,
      srv,
      lost: upto.reduce((a, x) => a + getD(x).lost, 0),
      waste: upto.reduce((a, x) => a + getD(x).waste, 0),
      cartons: upto.reduce((a, x) => a + getD(x).cartons, 0),
      floor: upto.reduce((a, x) => a + x.floor_n, 0),
    };
  }, [days, cursor, holidayPolicy, period]);

  if (cal.error) return <ErrorState message={cal.error} retry={cal.retry} />;
  if (!cal.data) return <Loading what="calendar" />;
  const C = cal.data.periods[period];
  const offset = parseISO(days[0].date).dow;
  const cur = cursor >= 0 ? days[cursor] : null;
  const openDay = (date: string) => {
    setPlaying(false);
    setDay(date);
    setOpen(true);
  };
  const onHover = (on: boolean, el: HTMLElement | null, d?: CalendarDay) => {
    if (!on || !el || !d || !gridRef.current) return setHover(null);
    const g = gridRef.current.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    setHover({ d, x: r.left - g.left + r.width / 2, y: r.bottom - g.top + 6, w: g.width });
  };

  return (
    <div className="space-y-5">
      <Card>
        <SectionHead
          title="The rolling replay, one morning at a time"
          right={
            <div className="flex flex-wrap items-center gap-2">
              <HolidayToggle />
              <button type="button" onClick={() => { if (cursor >= n - 1) setCursor(-1); setPhase(0); setPlaying((p) => !p); }} aria-pressed={playing}
                className="inline-flex h-10 items-center gap-2 rounded-full bg-accent px-4 text-[13.5px] font-semibold text-accent-ink cursor-pointer hover:brightness-105 transition-all">
                {playing ? <Pause size={16} aria-hidden /> : <Play size={16} aria-hidden />} {playing ? 'Pause' : cursor >= 0 && cursor < n - 1 ? 'Resume' : 'Play the replay'}
              </button>
              <button type="button" onClick={() => { setPlaying(false); setCursor(-1); setPhase(0); }} aria-label="Reset the replay" className="grid size-10 place-items-center rounded-full bg-surface-2 text-ink-2 hover:text-ink cursor-pointer">
                <RotateCcw size={16} aria-hidden />
              </button>
            </div>
          }
        >
          {C.label}, {C.range}: {n} mornings, 1,200 store-products each. Every morning the model only sees the past, decides, and then that day's
          real sales are played against the shelves. Colour = share of wanted units that were sold (fill rate). Click any day to open its decision.
          Store ILP layers: {C.layers}.
        </SectionHead>

        <ol className="mb-5 grid grid-cols-2 gap-2 lg:grid-cols-4" aria-label="Daily loop">
          {LOOP.map((s, i) => {
            const on = playing && phase === i;
            const Icon = s.icon;
            return (
              <li key={s.title} className="relative overflow-hidden rounded-xl bg-surface-2 p-3">
                {on && <motion.span layoutId="loop-hl" className="absolute inset-0 rounded-xl bg-accent-soft" transition={{ type: 'spring', bounce: 0.15, duration: 0.35 }} />}
                <div className="relative flex items-center gap-2 text-[13.5px] font-semibold text-ink">
                  <Icon size={16} aria-hidden className={on ? 'text-accent' : 'text-ink-3'} />
                  {s.title}
                </div>
                <p className="relative mt-1 text-[12.5px] leading-snug text-ink-2">{s.text}</p>
              </li>
            );
          })}
        </ol>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_300px]">
          <div className="relative min-w-0" ref={gridRef}>
            <div className="mb-1.5 grid grid-cols-7 gap-1 sm:gap-1.5" aria-hidden>
              {DOW_LABELS.map((x) => <div key={x} className="text-center text-[11.5px] font-medium text-ink-3">{x}</div>)}
            </div>
            <div className="grid grid-cols-7 gap-1 sm:gap-1.5" role="grid" aria-label={`${C.label} calendar, ${n} days`}>
              {Array.from({ length: offset }, (_, i) => <div key={`pad${i}`} />)}
              {days.map((d, i) => (
                <DayCell key={d.date} d={d} period={period} holidayOn={holidayPolicy} active={i === cursor || (open && day === d.date)} onOpen={() => openDay(d.date)} onHover={(on, el) => onHover(on, el, d)} />
              ))}
            </div>
            {hover && (
              <div role="tooltip" className="pointer-events-none absolute z-20 w-60 rounded-xl border border-line bg-surface p-3 text-[12.5px] shadow-lift"
                style={{ left: Math.max(0, Math.min(hover.x - 120, hover.w - 240)), top: hover.y }}>
                <div className="font-semibold text-ink">{fmtDate(hover.d.date, true)}</div>
                {hover.d.events.map((e) => <div key={e.name} className="text-ink-2">{e.name} ({e.type})</div>)}
                {holidayPolicy && getHolidayPhase(hover.d.date) && (
                  <div className="mt-1 inline-flex items-center gap-1 rounded bg-accent-soft px-1.5 py-0.5 text-[11px] font-semibold text-accent-ink">
                    <span>{getHolidayPhase(hover.d.date)!.icon}</span> {getHolidayPhase(hover.d.date)!.name}
                  </div>
                )}
                {!holidayPolicy && getHolidayPhase(hover.d.date) && (
                  <div className="mt-1 inline-flex items-center gap-1 rounded bg-surface-3 px-1.5 py-0.5 text-[10.5px] font-medium text-ink-3">
                    <span>{getHolidayPhase(hover.d.date)!.icon}</span> Policy Bypassed (Baseline Model)
                  </div>
                )}
                {(() => {
                  const hMet = holidayPolicy && period === 'test' && hover.d.holiday ? hover.d.holiday : null;
                  const dFill = hMet ? hMet.fill : hover.d.fill;
                  const dLost = hMet ? hMet.lost : hover.d.lost;
                  const dWaste = hMet ? hMet.waste : hover.d.waste;
                  const dCartons = hMet ? hMet.cartons : hover.d.cartons;
                  return (
                    <dl className="tabular mt-1.5 grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5 text-ink-2">
                      <dt>Units wanted</dt><dd className="text-right text-ink">{fmtInt(hover.d.demand)}</dd>
                      <dt>Fill rate</dt><dd className="text-right text-ink font-semibold">{hover.d.closed ? 'closed' : fmtPct(dFill, 2)}</dd>
                      <dt>Lost units</dt><dd className="text-right text-ink">{fmtInt(dLost)}</dd>
                      <dt>Waste units</dt><dd className="text-right text-ink font-semibold">{fmtInt(dWaste)}</dd>
                      <dt>Cartons ordered</dt><dd className="text-right text-ink">{fmtInt(dCartons)}</dd>
                      <dt>Floor-priced items</dt><dd className="text-right text-ink">{hover.d.floor_n}</dd>
                      <dt>ILP size</dt><dd className="text-right text-ink">{hover.d.n_vars ? `${fmtInt(hover.d.n_vars)} vars` : 'no solve'}</dd>
                      <dt>Solve time</dt><dd className="text-right text-ink">{hover.d.ilp_seconds != null ? `${fmtNum(hover.d.ilp_seconds, 2)} s` : '–'}</dd>
                    </dl>
                  );
                })()}
              </div>
            )}
            <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-[12px] text-ink-2" aria-label="Legend">
              <span className="font-medium text-ink-3">Fill rate</span>
              {BINS.map((b, i) => (
                <span key={b.label} className="inline-flex items-center gap-1.5">
                  <span className="size-3.5 rounded" style={{ background: `var(--cal-${i + 1})` }} aria-hidden />{b.label}
                </span>
              ))}
              <span className="inline-flex items-center gap-1.5"><span className="hatch size-3.5 rounded bg-surface-2" aria-hidden />closed</span>
              <span className="inline-flex items-center gap-1"><Flag size={13} aria-hidden /> holiday or event</span>
              <span className="inline-flex items-center gap-1"><span>🚀/❄️</span> holiday surge & throttle phases</span>
              <span className="inline-flex items-center gap-1"><PackageX size={13} aria-hidden /> no order possible</span>
              <span className="inline-flex items-center gap-1"><ShieldCheck size={13} aria-hidden /> items with a service-floor price</span>
            </div>
          </div>

          <aside aria-live="polite" className="rounded-2xl bg-surface-2 p-4">
            <div className="text-[12.5px] font-medium text-ink-3">{cur ? 'This morning' : 'Press play to step through the days'}</div>
            <div className="display mt-1 text-[1.25rem] font-semibold text-ink">{cur ? fmtDate(cur.date, true) : C.range}</div>
            {cur?.events.map((e) => <div key={e.name} className="text-[13px] text-ink-2">{e.name}</div>)}
            <dl className="tabular mt-3 grid grid-cols-2 gap-2 text-[12.5px]">
              {(() => {
                const curMet = cur && holidayPolicy && period === 'test' && cur.holiday ? cur.holiday : cur;
                return [
                  ['Fill today', curMet ? (cur?.closed ? 'closed' : fmtPct(curMet.fill, 1)) : '–'],
                  ['Cartons ordered', curMet ? fmtInt(curMet.cartons) : '–'],
                  ['Lost today', curMet ? fmtInt(curMet.lost) : '–'],
                  ['Waste today', curMet ? fmtInt(curMet.waste) : '–'],
                ].map(([k, v]) => (
                  <div key={k} className="rounded-lg bg-surface p-2">
                    <dt className="text-ink-3">{k}</dt>
                    <dd className="text-[15px] font-semibold text-ink">{v}</dd>
                  </div>
                ));
              })()}
            </dl>
            {cur && cur.date >= '2015-11-23' && cur.date <= '2015-11-28' && (
              <div className={`mt-3 rounded-xl border p-2.5 text-[12px] ${
                holidayPolicy
                  ? 'border-amber-500/40 bg-amber-500/10 text-ink'
                  : 'border-line bg-surface-3 text-ink-3'
              }`}>
                <div className="flex items-center justify-between font-semibold">
                  <span>Thanksgiving Slump Control</span>
                  <span className={`text-[10.5px] px-1.5 py-0.5 rounded font-mono ${holidayPolicy ? 'bg-amber-500/20 text-amber-900 dark:text-amber-200' : 'bg-surface-4'}`}>
                    {holidayPolicy ? 'PLAN ON' : 'PLAN OFF'}
                  </span>
                </div>
                <div className="mt-1 leading-snug">
                  {holidayPolicy
                    ? 'Dumpster waste throttled to 151 units (-80.6% vs 779 baseline).'
                    : 'Uncontrolled baseline: 779 units dumped during Thanksgiving week.'}
                </div>
              </div>
            )}
            <div className="mt-4 border-t border-line pt-3 text-[12.5px] font-medium text-ink-3">Running total so far</div>
            <dl className="tabular mt-2 space-y-1.5 text-[13px]">
              <div className="flex justify-between"><dt className="text-ink-2">Fill rate</dt><dd className="font-semibold text-ink"><AnimatedNumber value={totals.dem ? (totals.srv / totals.dem) * 100 : 0} format={(x) => `${fmtNum(x, 2)}%`} duration={0.5} /></dd></div>
              <div className="flex justify-between"><dt className="text-ink-2">Lost units</dt><dd className="font-semibold text-ink"><AnimatedNumber value={totals.lost} format={fmtInt} duration={0.5} /></dd></div>
              <div className="flex justify-between"><dt className="text-ink-2">Waste units</dt><dd className="font-semibold text-ink"><AnimatedNumber value={totals.waste} format={fmtInt} duration={0.5} /></dd></div>
              <div className="flex justify-between"><dt className="text-ink-2">Cartons ordered</dt><dd className="font-semibold text-ink"><AnimatedNumber value={totals.cartons} format={fmtInt} duration={0.5} /></dd></div>
              <div className="flex justify-between"><dt className="text-ink-2">Floor-priced store-items</dt><dd className="font-semibold text-ink"><AnimatedNumber value={totals.floor} format={fmtInt} duration={0.5} /></dd></div>
            </dl>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-surface-3" aria-hidden>
              <motion.div className="h-full bg-accent" animate={{ width: `${((cursor + 1) / n) * 100}%` }} transition={{ duration: 0.3 }} />
            </div>
            <Note className="mt-2">Day {Math.max(cursor + 1, 0)} of {n}. At the end the running fill rate equals the run's headline fill rate.</Note>
          </aside>
        </div>
      </Card>

      <MorphingDialog open={open && !!day} onClose={() => setOpen(false)} layoutId={`day-${period}-${day}`} label={`Decision for ${day ? fmtDate(day) : ''}`}>
        {day && <DayDetail key={`${period}-${day}`} period={period} date={day} compact />}
      </MorphingDialog>
    </div>
  );
}
