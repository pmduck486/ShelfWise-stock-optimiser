import { useCallback, useMemo } from 'react';
import { useApp } from '../lib/state';
import { PERIOD_SHORT, useJSON, type ForecastFile, type ItemsFile, type SeriesFile } from '../lib/data';
import { fmtDay, fmtNum } from '../lib/format';
import { Card, ErrorState, Loading, Note, SectionHead, Select, Skeleton, Source } from '../components/ui';
import { Chart, base, type Tokens } from '../components/Chart';
import { InView } from '../components/motion/InView';

function FanChart({ s, item }: { s: SeriesFile; item: string }) {
  const option = useCallback(
    (t: Tokens) => {
      const b = base(t);
      const x = s.dates.map(fmtDay);
      const v = s.items[item];
      const band = (name: string, lo: number[], hi: number[], color: string) => [
        { name, type: 'line', data: lo, stack: name, lineStyle: { opacity: 0 }, itemStyle: { color }, symbol: 'none', silent: true, tooltip: { show: false } },
        { name, type: 'line', data: hi.map((h, i) => +(h - lo[i]).toFixed(2)), stack: name, lineStyle: { opacity: 0 }, itemStyle: { color }, symbol: 'none', areaStyle: { color, opacity: 1 }, silent: true, tooltip: { show: false } },
      ];
      return {
        textStyle: b.textStyle,
        legend: { top: 0, left: 0, itemWidth: 14, itemHeight: 10, textStyle: { color: t.ink2, fontSize: 11.5 }, data: ['q50 to q80', 'q80 to q90', 'q90 to q95', 'q50 (median)', 'Actual sales'] },
        grid: { left: 8, right: 16, top: 40, bottom: 64, containLabel: true },
        tooltip: {
          ...b.tooltip, trigger: 'axis',
          formatter: (ps: { dataIndex: number }[]) => {
            const i = ps[0].dataIndex;
            return `<b>${x[i]}</b><br/>q50 ${fmtNum(v.q50[i], 1)} · q80 ${fmtNum(v.q80[i], 1)}<br/>q90 ${fmtNum(v.q90[i], 1)} · q95 ${fmtNum(v.q95[i], 1)}<br/>actual <b>${fmtNum(v.act[i], 0)}</b>${v.act[i] > v.q95[i] ? ' (above q95)' : ''}`;
          },
        },
        dataZoom: [
          { type: 'slider', bottom: 8, height: 22, borderColor: t.line, fillerColor: 'rgba(120,120,120,0.15)', handleStyle: { color: t.surface, borderColor: t.lineStrong }, textStyle: { color: t.ink3 }, dataBackground: { lineStyle: { color: t.lineStrong }, areaStyle: { color: t.surface2 } } },
          { type: 'inside' },
        ],
        xAxis: { type: 'category', data: x, boundaryGap: false, ...b.axis },
        yAxis: { type: 'value', ...b.axis, axisLine: { show: false } },
        series: [
          ...band('q90 to q95', v.q90, v.q95, t.seq[0]),
          ...band('q80 to q90', v.q80, v.q90, t.seq[1]),
          ...band('q50 to q80', v.q50, v.q80, t.seq[2]),
          { name: 'q50 (median)', type: 'line', data: v.q50, symbol: 'none', lineStyle: { color: t.seq[5], width: 2 }, itemStyle: { color: t.seq[5] } },
          { name: 'Actual sales', type: 'scatter', data: v.act, symbolSize: 7, itemStyle: { color: t.s2, borderColor: t.surface, borderWidth: 1.5 } },
        ],
      };
    },
    [s, item],
  );
  return <Chart option={option} height={360} label={`Same-day forecast bands q50 to q95 and actual sales for ${s.store} ${item}`} />;
}

function CoverageBars({ f }: { f: ForecastFile }) {
  const option = useCallback(
    (t: Tokens) => {
      const b = base(t);
      const lines = ['q50', 'q80', 'q90', 'q95'] as const;
      const groups: { name: string; vals: (number | null)[] }[] = [
        { name: 'Validation fold 1 (Nov–Dec 2014)', vals: lines.map((l) => f.folds.fold1_festive.model[`cover_${l}`]) },
        { name: 'Validation fold 2 (Jul–Oct 2015)', vals: lines.map((l) => f.folds.fold2_main.model[`cover_${l}`]) },
        { name: 'Test window (Nov–Dec 2015)', vals: lines.map((l) => f.test.all[`cover_${l}`]) },
        { name: 'Holdout 1', vals: [null, f.holdout.b1.cover_q80, f.holdout.b1.cover_q90, f.holdout.b1.cover_q95] },
        { name: 'Holdout 2', vals: [null, f.holdout.b2.cover_q80, f.holdout.b2.cover_q90, f.holdout.b2.cover_q95] },
      ];
      const colors = [t.s1, t.s2, t.s3, t.s4, t.s5];
      return {
        textStyle: b.textStyle,
        legend: { top: 0, left: 0, textStyle: { color: t.ink2, fontSize: 11.5 } },
        grid: { left: 8, right: 16, top: 64, bottom: 8, containLabel: true },
        tooltip: { ...b.tooltip, trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v: number | null) => (v == null ? 'not scored' : `${fmtNum(v, 1)}%`) },
        xAxis: { type: 'category', data: lines.map((l) => `${l} (target ${l.slice(1)}%)`), ...b.axis, axisLabel: { color: t.ink2, fontSize: 12 } },
        yAxis: { type: 'value', min: 40, max: 100, ...b.axis, axisLine: { show: false }, axisLabel: { ...b.axis.axisLabel, formatter: '{value}%' } },
        series: [
          ...groups.map((g, i) => ({ name: g.name, type: 'bar', barGap: '12%', barMaxWidth: 16, data: g.vals, itemStyle: { color: colors[i], borderRadius: [3, 3, 0, 0] } })),
          { name: 'Target', type: 'scatter', symbol: 'rect', symbolSize: [56, 2], data: [50, 80, 90, 95], itemStyle: { color: t.ink }, z: 10, tooltip: { show: false } },
        ],
      };
    },
    [f],
  );
  return <Chart option={option} height={300} label="Coverage of each forecast line against its target, by period" />;
}

export function ForecastExplorer() {
  const { period, store, setStore, item, setItem } = useApp();
  const items = useJSON<ItemsFile>('items.json');
  const fc = useJSON<ForecastFile>('forecast.json');
  const ser = useJSON<SeriesFile>(`series/${period}/${store}.json`);
  const itemId = item ?? 'FOODS_3_090';

  const own = useMemo(() => {
    if (!ser.data) return null;
    const v = ser.data.items[itemId];
    const open = v.act.map((_, i) => !ser.data!.dates[i].endsWith('12-25'));
    const n = open.filter(Boolean).length;
    const cov = (k: 'q50' | 'q80' | 'q90' | 'q95') => (v.act.filter((a, i) => open[i] && a <= v[k][i]).length / n) * 100;
    const mae = v.act.reduce((s, a, i) => s + (open[i] ? Math.abs(a - v.q50[i]) : 0), 0) / n;
    return { n, mae, q50: cov('q50'), q80: cov('q80'), q90: cov('q90'), q95: cov('q95') };
  }, [ser.data, itemId]);

  if (items.error || fc.error) return <ErrorState message={(items.error || fc.error)!} retry={() => { items.retry(); fc.retry(); }} />;
  if (!items.data || !fc.data) return <Loading what="forecasts" />;
  const F = fc.data;

  return (
    <div className="space-y-5">
      <Card>
        <SectionHead
          title="The forecaster's ranges against what sold"
          right={
            <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-2 sm:w-[420px]">
              <Select label="Store" value={store} onChange={setStore} options={items.data.stores.map((s) => ({ value: s, label: s }))} />
              <Select label="Product" value={itemId} onChange={(v) => setItem(v)} options={items.data.items.map((x) => ({ value: x.id, label: `${x.id} · ${x.dept}`, group: x.perishable ? 'Perishable' : 'Long-life' }))} />
            </div>
          }
        >
          {PERIOD_SHORT[period]}: the forecast made on the morning of each day for that same day (data at least 2 days old). Units sold per day. Drag the slider under the chart to zoom into a date range.
        </SectionHead>
        {ser.error ? <ErrorState message={ser.error} retry={ser.retry} /> : ser.data ? <FanChart s={ser.data} item={itemId} /> : <Skeleton h={360} />}
        {own && (
          <div className="tabular mt-3 flex flex-wrap gap-x-6 gap-y-1 text-[13px] text-ink-2">
            <span>This store-item, {own.n} open days:</span>
            <span>q50 MAE <b className="text-ink">{fmtNum(own.mae, 2)}</b> units</span>
            <span>at or below q50 <b className="text-ink">{fmtNum(own.q50, 0)}%</b></span>
            <span>q80 <b className="text-ink">{fmtNum(own.q80, 0)}%</b></span>
            <span>q90 <b className="text-ink">{fmtNum(own.q90, 0)}%</b></span>
            <span>q95 <b className="text-ink">{fmtNum(own.q95, 0)}%</b></span>
          </div>
        )}
        <Note className="mt-2">One store-item is noisy: over a few weeks its coverage can be well off target even when the chain-wide numbers below are on target.</Note>
        <Source>series/{period}/{store}.json, from the replay's forecasts file and inventory ledger</Source>
      </Card>

      <InView>
        <Card>
          <SectionHead title="Are the lines honest? Coverage by period">
            Coverage = share of store-item-days with sales at or below the line. A q90 line should cover about 90%. Holdout files store only q80 to q95.
            The q50 line covers a little more than 50% because many days have zero sales and a zero forecast (a tie counts as covered).
          </SectionHead>
          <CoverageBars f={F} />
          <Note className="mt-2">Test window: {F.test.note}. Calibration on the validation folds needed no stretching (k = {Object.values(F.calibration.shifts).slice(1).join(', ')} for q80, q90, q95; tolerance {F.calibration.tolerance_pts} points).</Note>
          <Source>{F.sources.join('; ')}</Source>
        </Card>
      </InView>

      <InView>
        <Card>
          <SectionHead title="Better than the two baselines">
            Error of the median line (MAE, units per store-item-day) and pinball loss of each line on the two validation folds. Lower is better.
            Naive = same day last week; stub = average of the same weekday over the last 4 weeks.
          </SectionHead>
          <div className="overflow-x-auto">
            <table className="tabular w-full min-w-[560px] text-[13.5px]">
              <thead>
                <tr className="text-left text-[12px] text-ink-3">
                  <th className="py-2 pr-3 font-medium">Fold</th><th className="py-2 pr-3 font-medium">Method</th>
                  <th className="py-2 pr-3 text-right font-medium">MAE</th>
                  {['q50', 'q80', 'q90', 'q95'].map((l) => <th key={l} className="py-2 pr-3 text-right font-medium">pinball {l}</th>)}
                </tr>
              </thead>
              <tbody>
                {(['fold1_festive', 'fold2_main'] as const).flatMap((fold) =>
                  (['naive', 'stub', 'model'] as const).map((m) => {
                    const s = F.folds[fold][m];
                    return (
                      <tr key={fold + m} className={`border-t border-line ${m === 'model' ? 'font-semibold text-ink' : 'text-ink-2'}`}>
                        <td className="py-1.5 pr-3">{m === 'naive' ? `${fold === 'fold1_festive' ? 'Fold 1' : 'Fold 2'} (${F.fold_info[fold].start} to ${F.fold_info[fold].end})` : ''}</td>
                        <td className="py-1.5 pr-3">{m === 'model' ? 'Our forecaster' : m === 'naive' ? 'Naive baseline' : 'Stub baseline'}</td>
                        <td className="py-1.5 pr-3 text-right">{fmtNum(s.mae, 3)}</td>
                        {(['pinball_q50', 'pinball_q80', 'pinball_q90', 'pinball_q95'] as const).map((k) => <td key={k} className="py-1.5 pr-3 text-right">{fmtNum(s[k] ?? null, 3)}</td>)}
                      </tr>
                    );
                  }),
                )}
              </tbody>
            </table>
          </div>
          <div className="tabular mt-4 grid gap-2 text-[13px] sm:grid-cols-3">
            <div className="rounded-xl bg-surface-2 p-3"><div className="text-ink-3">Test window q50 MAE (computed)</div><div className="text-[18px] font-semibold text-ink">{fmtNum(F.test.all.mae, 2)}</div><div className="text-ink-3">perishable {fmtNum(F.test.perishable.mae, 2)} · long-life {fmtNum(F.test.long_life.mae, 2)}</div></div>
            <div className="rounded-xl bg-surface-2 p-3"><div className="text-ink-3">Holdout 1 q50 MAE</div><div className="text-[18px] font-semibold text-ink">{fmtNum(F.holdout.b1.mae, 2)}</div><div className="text-ink-3">{F.holdout.b1.days} days · {F.holdout.b1.rows.toLocaleString('en-US')} rows</div></div>
            <div className="rounded-xl bg-surface-2 p-3"><div className="text-ink-3">Holdout 2 q50 MAE</div><div className="text-[18px] font-semibold text-ink">{fmtNum(F.holdout.b2.mae, 2)}</div><div className="text-ink-3">{F.holdout.b2.days} days · {F.holdout.b2.rows.toLocaleString('en-US')} rows</div></div>
          </div>
          <Source>outputs/forecast/scores/*.json, data/processed_holdout_b*/forecast_scores.csv</Source>
        </Card>
      </InView>
    </div>
  );
}
