import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import type { PeriodKey } from './data';

export const PAGES = [
  { id: 'overview', label: 'Overview' },
  { id: 'workflow', label: 'Workflow calendar' },
  { id: 'day', label: 'Day decision' },
  { id: 'forecast', label: 'Forecast explorer' },
  { id: 'shadow', label: 'Shadow prices & what-ifs' },
  { id: 'layers', label: 'Layers' },
  { id: 'how', label: 'How it works' },
] as const;
export type PageId = (typeof PAGES)[number]['id'];

type Theme = 'light' | 'dark';

interface AppState {
  page: PageId;
  go: (p: PageId) => void;
  period: PeriodKey;
  setPeriod: (p: PeriodKey) => void;
  day: string | null; // ISO date selected in the calendar / day page
  setDay: (d: string | null) => void;
  store: string;
  setStore: (s: string) => void;
  item: string | null; // null = let the page pick an interesting one
  setItem: (i: string | null) => void;
  theme: Theme;
  toggleTheme: () => void;
  holidayPolicy: boolean;
  setHolidayPolicy: (on: boolean | ((prev: boolean) => boolean)) => void;
}

const Ctx = createContext<AppState | null>(null);

function readHash(): PageId {
  const id = window.location.hash.replace(/^#\/?/, '').split('?')[0];
  return (PAGES.find((p) => p.id === id)?.id ?? 'overview') as PageId;
}

function systemTheme(): Theme {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function storedTheme(): Theme | null {
  try {
    const t = localStorage.getItem('shelfwise-theme') || localStorage.getItem('retailops-theme');
    return t === 'light' || t === 'dark' ? t : null;
  } catch {
    return null;
  }
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [page, setPage] = useState<PageId>(readHash);
  const [period, setPeriodRaw] = useState<PeriodKey>('test');
  const [day, setDay] = useState<string | null>(null);
  const [store, setStore] = useState('CA_1');
  const [item, setItem] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(() => storedTheme() ?? systemTheme());
  const [holidayPolicy, setHolidayPolicyRaw] = useState<boolean>(() => {
    try {
      const v = localStorage.getItem('shelfwise-holiday-policy');
      return v === null ? true : v === 'true';
    } catch {
      return true;
    }
  });

  const setHolidayPolicy = useCallback((on: boolean | ((prev: boolean) => boolean)) => {
    setHolidayPolicyRaw((prev) => {
      const next = typeof on === 'function' ? on(prev) : on;
      try {
        localStorage.setItem('shelfwise-holiday-policy', String(next));
      } catch {
        /* storage blocked */
      }
      return next;
    });
  }, []);

  useEffect(() => {
    const onHash = () => setPage(readHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  useEffect(() => {
    if (storedTheme()) return;
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)');
    const on = () => setTheme(mq.matches ? 'dark' : 'light');
    mq?.addEventListener('change', on);
    return () => mq?.removeEventListener('change', on);
  }, []);

  const go = useCallback((p: PageId) => {
    if (readHash() !== p) window.location.hash = `/${p}`;
    setPage(p);
    window.scrollTo({ top: 0 });
  }, []);

  const setPeriod = useCallback((p: PeriodKey) => {
    setPeriodRaw(p);
    setDay(null);
    setItem(null);
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((t) => {
      const next: Theme = t === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try {
        localStorage.setItem('shelfwise-theme', next);
      } catch {
        /* storage blocked: the toggle still works for this visit */
      }
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({ page, go, period, setPeriod, day, setDay, store, setStore, item, setItem, theme, toggleTheme, holidayPolicy, setHolidayPolicy }),
    [page, go, period, setPeriod, day, store, item, theme, toggleTheme, holidayPolicy, setHolidayPolicy],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp() {
  const v = useContext(Ctx);
  if (!v) throw new Error('useApp outside provider');
  return v;
}
