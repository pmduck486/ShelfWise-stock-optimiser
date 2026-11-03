import type { ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import { AnimatedBackground } from './motion/AnimatedBackground';
import { useApp } from '../lib/state';
import { PERIOD_SHORT, type PeriodKey } from '../lib/data';

export function Card({ children, className = '', as: Tag = 'section', label }: { children: ReactNode; className?: string; as?: 'section' | 'div' | 'article'; label?: string }) {
  return (
    <Tag aria-label={label} className={`rounded-2xl bg-surface p-4 shadow-card sm:p-6 ${className}`}>
      {children}
    </Tag>
  );
}

export function SectionHead({ title, children, right }: { title: string; children?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
      <div className="min-w-0 max-w-[68ch]">
        <h2 className="display text-[1.35rem] font-semibold leading-tight text-ink sm:text-[1.5rem]">{title}</h2>
        {children && <div className="mt-1.5 text-[14px] text-ink-2">{children}</div>}
      </div>
      {right && <div className="max-w-full">{right}</div>}
    </div>
  );
}

export function Skeleton({ h = 240, className = '' }: { h?: number; className?: string }) {
  return <div className={`skeleton ${className}`} style={{ height: h }} aria-hidden />;
}

export function Loading({ what }: { what: string }) {
  return (
    <div role="status" aria-live="polite" className="space-y-4">
      <span className="sr-only">Loading {what}…</span>
      <Skeleton h={120} />
      <div className="grid gap-4 sm:grid-cols-2">
        <Skeleton h={260} />
        <Skeleton h={260} />
      </div>
    </div>
  );
}

export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-3 rounded-2xl border border-line bg-surface p-6">
      <div className="flex items-center gap-2 font-semibold text-ink">
        <AlertTriangle size={18} className="text-critical" aria-hidden /> The data did not load
      </div>
      <p className="text-[14px] text-ink-2">
        {message} The dashboard reads JSON from <code className="font-mono text-[13px]">public/data/</code>; run{' '}
        <code className="font-mono text-[13px]">python export_data.py</code> to create it, then reload.
      </p>
      {retry && (
        <button type="button" onClick={retry} className="inline-flex items-center gap-2 rounded-full bg-accent px-4 py-2 text-[13px] font-semibold text-accent-ink">
          <RotateCcw size={14} aria-hidden /> Try again
        </button>
      )}
    </div>
  );
}

export function Note({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <p className={`text-[13px] leading-relaxed text-ink-3 ${className}`}>{children}</p>;
}

export function Source({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 text-[12px] text-ink-3">
      Source: <span className="font-mono text-[11.5px] [overflow-wrap:anywhere]">{children}</span>
    </p>
  );
}

export function PeriodSwitch({ id = 'period' }: { id?: string }) {
  const { period, setPeriod } = useApp();
  const opts: { value: PeriodKey; label: string }[] = (['test', 'b1', 'b2'] as PeriodKey[]).map((p) => ({ value: p, label: PERIOD_SHORT[p] }));
  return <AnimatedBackground id={id} label="Dataset period" options={opts} value={period} onChange={setPeriod} optionClass="px-2.5 sm:px-3.5" />;
}

export function Chip({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'good' | 'warn' | 'bad' }) {
  const cls = {
    neutral: 'bg-surface-2 text-ink-2',
    accent: 'bg-accent-soft text-ink',
    good: 'bg-surface-2 text-good-ink',
    warn: 'bg-surface-2 text-ink',
    bad: 'bg-surface-2 text-ink',
  }[tone];
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[12px] font-medium ${cls}`}>{children}</span>;
}

export function Select<T extends string>({ label, value, onChange, options, className = '' }: { label: string; value: T; onChange: (v: T) => void; options: { value: T; label: string; group?: string }[]; className?: string }) {
  const groups = Array.from(new Set(options.map((o) => o.group ?? '')));
  return (
    <label className={`flex min-w-0 flex-col gap-1 text-[12px] font-medium text-ink-3 ${className}`}>
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="h-9 min-w-0 rounded-lg border border-line bg-surface px-2.5 text-[14px] font-normal text-ink hover:border-line-strong"
      >
        {groups.map((g) =>
          g ? (
            <optgroup key={g} label={g}>
              {options.filter((o) => o.group === g).map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </optgroup>
          ) : (
            options.filter((o) => !o.group).map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))
          ),
        )}
      </select>
    </label>
  );
}
