import { AnimatePresence, LayoutGroup, MotionConfig, motion, useReducedMotion } from 'motion/react';
import { Moon, Sun } from 'lucide-react';
import { AppStateProvider, PAGES, useApp, type PageId } from './lib/state';
import { PeriodSwitch } from './components/ui';
import { Overview } from './pages/Overview';
import { Workflow } from './pages/Workflow';
import { DayPage } from './pages/DayDetail';
import { ForecastExplorer } from './pages/Forecast';
import { ShadowPage } from './pages/Shadow';
import { LayersPage } from './pages/Layers';
import { HowItWorks } from './pages/HowItWorks';

const WITH_PERIOD: PageId[] = ['overview', 'workflow', 'day', 'forecast', 'shadow'];

function Header() {
  const { page, go, theme, toggleTheme } = useApp();
  const reduce = useReducedMotion();
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 pt-3 sm:px-6">
        <a href="#/overview" onClick={(e) => { e.preventDefault(); go('overview'); }} className="flex min-w-0 items-baseline gap-2 text-ink no-underline">
          <span className="display text-[1.35rem] font-semibold leading-none">Shelfwise</span>
          <span className="hidden truncate text-[12.5px] text-ink-3 md:inline">Stock ordering for 10 stores on real Walmart M5 sales · CS2105, IIT Patna</span>
        </a>
        <div className="ml-auto" />
        <button
          type="button"
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          className="grid size-9 place-items-center rounded-full bg-surface-2 text-ink-2 transition-colors hover:bg-surface-3 hover:text-ink"
        >
          {theme === 'dark' ? <Sun size={17} aria-hidden /> : <Moon size={17} aria-hidden />}
        </button>
      </div>
      <nav aria-label="Pages" className="no-scrollbar mx-auto max-w-7xl overflow-x-auto px-2 sm:px-4">
        <LayoutGroup id="tabs">
          <ul className="flex min-w-max gap-1 pb-0 pt-2">
            {PAGES.map((p) => {
              const active = p.id === page;
              return (
                <li key={p.id}>
                  <a
                    href={`#/${p.id}`}
                    aria-current={active ? 'page' : undefined}
                    onClick={(e) => { e.preventDefault(); go(p.id); }}
                    className={`relative block whitespace-nowrap px-3 pb-2.5 pt-1.5 text-[13.5px] font-medium no-underline transition-colors ${active ? 'text-ink' : 'text-ink-3 hover:text-ink'}`}
                  >
                    {p.label}
                    {active && (
                      <motion.span
                        layoutId="tab-underline"
                        className="absolute inset-x-2 -bottom-px h-[2px] rounded-full bg-accent"
                        transition={reduce ? { duration: 0 } : { type: 'spring', bounce: 0.15, duration: 0.45 }}
                      />
                    )}
                  </a>
                </li>
              );
            })}
          </ul>
        </LayoutGroup>
      </nav>
    </header>
  );
}

function PeriodBar() {
  const { page } = useApp();
  if (!WITH_PERIOD.includes(page)) return null;
  return (
    <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-3 gap-y-2 px-4 pt-5 sm:px-6">
      <span className="text-[12.5px] font-medium text-ink-3">Period</span>
      <PeriodSwitch />
    </div>
  );
}

function Pages() {
  const { page } = useApp();
  const reduce = useReducedMotion();
  const view = {
    overview: <Overview />,
    workflow: <Workflow />,
    day: <DayPage />,
    forecast: <ForecastExplorer />,
    shadow: <ShadowPage />,
    layers: <LayersPage />,
    how: <HowItWorks />,
  }[page];
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.main
        key={page}
        id="main"
        tabIndex={-1}
        className="mx-auto max-w-7xl px-4 pb-20 pt-5 sm:px-6"
        initial={reduce ? { opacity: 0 } : { opacity: 0, y: 10, filter: 'blur(6px)' }}
        animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
        exit={reduce ? { opacity: 0 } : { opacity: 0, y: -6, filter: 'blur(4px)' }}
        transition={{ duration: reduce ? 0.01 : 0.3, ease: [0.16, 1, 0.3, 1] }}
      >
        {view}
      </motion.main>
    </AnimatePresence>
  );
}

export function App() {
  return (
    <MotionConfig reducedMotion="user">
      <AppStateProvider>
        <button
          type="button"
          onClick={() => document.getElementById('main')?.focus()}
          className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2"
        >
          Skip to content
        </button>
        <Header />
        <PeriodBar />
        <Pages />
        <footer className="mx-auto max-w-7xl px-4 pb-10 text-[12px] text-ink-3 sm:px-6">
          Every number on this site is read from the saved Shelfwise outputs (see dashboard/README.md for the file behind each one). Money is in M5 shelf-price units (US dollars).
        </footer>
      </AppStateProvider>
    </MotionConfig>
  );
}
