import { useEffect, useRef, useState } from 'react';
import { useReducedMotion } from 'motion/react';
import * as echarts from 'echarts/core';
import { BarChart, CustomChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts';
import {
  AriaComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components';
import { LegacyGridContainLabel } from 'echarts/features';
import { SVGRenderer } from 'echarts/renderers';
import type { EChartsCoreOption } from 'echarts/core';
import { useApp } from '../lib/state';

echarts.use([
  BarChart, LineChart, HeatmapChart, ScatterChart, CustomChart,
  AriaComponent, DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent,
  MarkPointComponent, TitleComponent, TooltipComponent, VisualMapComponent, SVGRenderer, LegacyGridContainLabel,
]);

/** Reads the current CSS tokens so charts follow the light/dark theme. */
export function tokens() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string) => cs.getPropertyValue(n).trim();
  return {
    surface: v('--surface'),
    surface2: v('--surface-2'),
    line: v('--line'),
    lineStrong: v('--line-strong'),
    ink: v('--ink'),
    ink2: v('--ink-2'),
    ink3: v('--ink-3'),
    accent: v('--accent'),
    s1: v('--s1'),
    s1soft: v('--s1-soft'),
    s2: v('--s2'),
    s3: v('--s3'),
    s4: v('--s4'),
    s5: v('--s5'),
    neutral: v('--neutral-mark'),
    seq: [1, 2, 3, 4, 5, 6].map((i) => v(`--seq-${i}`)),
    heat: [0, 1, 2, 3, 4].map((i) => v(`--heat-${i}`)),
    good: v('--good'),
    critical: v('--critical'),
  };
}
export type Tokens = ReturnType<typeof tokens>;

/** Shared axis / tooltip styling (recessive grid, text in ink tokens). */
export function base(t: Tokens) {
  return {
    textStyle: { fontFamily: 'IBM Plex Sans, system-ui, sans-serif', color: t.ink2 },
    tooltip: {
      backgroundColor: t.surface,
      borderColor: t.line,
      borderWidth: 1,
      textStyle: { color: t.ink, fontSize: 12 },
      extraCssText: 'box-shadow: var(--shadow); border-radius: 10px; padding: 8px 10px;',
    },
    axis: {
      axisLine: { lineStyle: { color: t.lineStrong } },
      axisTick: { show: false },
      axisLabel: { color: t.ink3, fontSize: 11 },
      splitLine: { lineStyle: { color: t.line, type: [3, 3] as number[] } },
    },
  };
}

export function Chart({
  option,
  height = 280,
  label,
  className = '',
  onEvents,
}: {
  option: (t: Tokens, reduce: boolean) => EChartsCoreOption;
  height?: number | string;
  label: string;
  className?: string;
  onEvents?: Record<string, (p: unknown) => void>;
}) {
  const el = useRef<HTMLDivElement>(null);
  const inst = useRef<echarts.ECharts | null>(null);
  const [ready, setReady] = useState(false);
  const { theme } = useApp();
  const reduce = !!useReducedMotion();

  // Create the chart only once its box has a size (pages animate in, so the first frame can be 0 wide).
  useEffect(() => {
    const node = el.current;
    if (!node) return;
    const ro = new ResizeObserver(() => {
      if (!node.clientWidth || !node.clientHeight) return;
      if (!inst.current) {
        inst.current = echarts.init(node, undefined, { renderer: 'svg' });
        setReady(true);
      } else inst.current.resize();
    });
    ro.observe(node);
    return () => {
      ro.disconnect();
      inst.current?.dispose();
      inst.current = null;
    };
  }, []);

  useEffect(() => {
    const c = inst.current;
    if (!c || !ready) return;
    const t = tokens();
    const o = option(t, reduce) as Record<string, unknown>;
    c.setOption(
      {
        animation: !reduce,
        animationDuration: 700,
        animationDurationUpdate: 650,
        animationEasing: 'cubicOut',
        animationEasingUpdate: 'cubicInOut',
        aria: { enabled: true, label: { description: label } },
        ...o,
      },
      { notMerge: false, replaceMerge: ['series', 'xAxis', 'yAxis', 'visualMap', 'dataZoom', 'legend', 'grid'] },
    );
  }, [option, theme, reduce, label, ready]);

  useEffect(() => {
    const c = inst.current;
    if (!c || !onEvents || !ready) return;
    for (const [k, fn] of Object.entries(onEvents)) c.on(k, fn);
    return () => {
      for (const k of Object.keys(onEvents)) c.off(k);
    };
  }, [onEvents, ready]);

  return <div ref={el} role="img" aria-label={label} className={className} style={{ height, width: '100%' }} />;
}
