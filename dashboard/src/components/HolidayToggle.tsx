import { Flame, ToggleLeft, ToggleRight } from 'lucide-react';
import { useApp } from '../lib/state';

interface HolidayToggleProps {
  compact?: boolean;
  className?: string;
  showStatusText?: boolean;
}

export function HolidayToggle({ compact = false, className = '', showStatusText = true }: HolidayToggleProps) {
  const { holidayPolicy, setHolidayPolicy } = useApp();

  return (
    <button
      type="button"
      onClick={() => setHolidayPolicy(!holidayPolicy)}
      className={`inline-flex items-center gap-1.5 rounded-full transition-all cursor-pointer ${
        compact ? 'px-2.5 py-1 text-[11.5px]' : 'px-3.5 py-1.5 text-[12.5px]'
      } font-semibold shadow-xs ${
        holidayPolicy
          ? 'bg-amber-500/15 text-amber-900 border border-amber-500/40 hover:bg-amber-500/25 dark:text-amber-200'
          : 'bg-surface-2 text-ink-3 border border-line hover:bg-surface-3 hover:text-ink'
      } ${className}`}
      aria-pressed={holidayPolicy}
      aria-label="Toggle Holiday Plan Layer"
      title="Toggle Holiday Surge & Valley Policy Layer"
    >
      <Flame size={compact ? 13 : 15} className={holidayPolicy ? 'text-amber-600 dark:text-amber-400' : 'text-ink-3'} aria-hidden />
      <span>{compact ? 'Holiday Plan' : 'Holiday Policy Switch'}</span>
      {holidayPolicy ? (
        <ToggleRight size={compact ? 16 : 18} className="text-amber-600 dark:text-amber-400" aria-hidden />
      ) : (
        <ToggleLeft size={compact ? 16 : 18} className="text-ink-3" aria-hidden />
      )}
      {showStatusText && (
        <span className={`text-[10.5px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded ${
          holidayPolicy ? 'bg-amber-500/20 text-amber-800 dark:text-amber-300' : 'bg-surface-3 text-ink-3'
        }`}>
          {holidayPolicy ? 'ON' : 'OFF'}
        </span>
      )}
    </button>
  );
}
