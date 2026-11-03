import { motion, useReducedMotion } from 'motion/react';
import type { ReactNode } from 'react';

/** A highlight that slides between options: a keyboard-accessible segmented control (radiogroup). */
export function AnimatedBackground<T extends string>({
  id,
  options,
  value,
  onChange,
  label,
  className = '',
  optionClass = '',
  render,
}: {
  id: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  className?: string;
  optionClass?: string;
  render?: (o: { value: T; label: string }, active: boolean) => ReactNode;
}) {
  const reduce = useReducedMotion();
  return (
    <div role="radiogroup" aria-label={label} className={`relative inline-flex max-w-full rounded-full bg-surface-2 p-1 ${className}`}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            onKeyDown={(e) => {
              const i = options.findIndex((x) => x.value === value);
              let next = -1;
              if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (i + 1) % options.length;
              if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (i - 1 + options.length) % options.length;
              if (next >= 0) {
                e.preventDefault();
                onChange(options[next].value);
                const btns = (e.currentTarget.parentElement as HTMLElement).querySelectorAll('button');
                (btns[next] as HTMLButtonElement | undefined)?.focus();
              }
            }}
            tabIndex={active ? 0 : -1}
            className={`relative z-0 whitespace-nowrap rounded-full px-3 py-1.5 text-[13px] font-medium transition-colors duration-200 ${
              active ? 'text-ink' : 'text-ink-3 hover:text-ink'
            } ${optionClass}`}
          >
            {active && (
              <motion.span
                layoutId={`bg-${id}`}
                className="absolute inset-0 -z-10 rounded-full bg-surface shadow-card"
                transition={reduce ? { duration: 0 } : { type: 'spring', bounce: 0.18, duration: 0.45 }}
              />
            )}
            {render ? render(o, active) : o.label}
          </button>
        );
      })}
    </div>
  );
}
