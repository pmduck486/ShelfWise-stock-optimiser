import { useEffect, useState } from 'react';

/* ---- types for the JSON written by export_data.py (schema in README.md) ---- */

export type PeriodKey = 'test' | 'b1' | 'b2';
export type MethodId = 'last_week' | 'topup_avg' | 'topup_forecast' | 'store_model' | 'store_ilp';

export interface MethodRow {
  id: MethodId;
  label: string;
  fill: number;
  lost: number;
  waste: number;
  stock: number;
  net_cost: number;
  penalty: number;
  total: number;
  stockout_days: number;
  orders: number;
  trucks: number;
  purchase: number;
  turnover: number;
  days_of_stock: number;
  n_vars: number | null;
  n_int: number | null;
  solve_seconds: number | null;
  audit_passed: boolean | null;
  rules_passed: boolean | null;
  service_met_share: number;
  vs_best_simple: number;
  source: string;
}

export interface LayerRow {
  id: string;
  label: string;
  floor: boolean;
  spoil: boolean;
  stockcap: boolean;
  holiday?: boolean;
  fill: number;
  lost: number;
  waste: number;
  stock: number;
  total: number;
  verdict: string;
}

export interface Summary {
  periods: Record<PeriodKey, {
    label: string;
    range: string;
    weeks: number;
    note: string;
    best_simple: MethodId;
    methods: MethodRow[];
    store_ilp_layers: string;
  }>;
  period_order: PeriodKey[];
  layers: { source: string; rows: LayerRow[]; test: string };
  money: { unit: string; note: string };
  totals_definition: string;
}

export interface CalendarDay {
  date: string;
  dow: number;
  closed: boolean;
  can_order: boolean;
  cover_days: number;
  demand: number;
  served: number;
  lost: number;
  fill: number | null;
  waste: number;
  cartons: number;
  units: number;
  trucks: number;
  floor_n: number;
  floor_mean: number | null;
  floor_max: number | null;
  spoil_n: number;
  spoil_mean: number | null;
  binding: number | null;
  n_vars: number | null;
  n_int: number | null;
  ilp_seconds: number | null;
  events: { name: string; type: string }[];
  snap: Record<string, number>;
  methods: Record<MethodId, [number | null, number, number]>;
  limits: Partial<Record<'truck' | 'space' | 'budget', [number, number, number]>>;
  holiday?: {
    served: number;
    lost: number;
    waste: number;
    fill: number | null;
    cartons: number;
    units: number;
    trucks: number;
  } | null;
}

export interface CalendarFile {
  periods: Record<PeriodKey, {
    label: string;
    range: string;
    start: string;
    end: string;
    layers: string;
    days: CalendarDay[];
    self_check?: { matched: number; total: number; what: string };
  }>;
  period_order: PeriodKey[];
  stores: string[];
  method_order: MethodId[];
}

export interface ItemInfo {
  id: string;
  dept: string;
  rank: number;
  avg_daily: number;
  perishable: boolean;
  shelf_life: number | null;
  upc: number;
}

export interface ItemParams {
  price: number;
  buy: number;
  hold: number;
  service_level: number;
  level: number;
}

export interface PeriodParams {
  items: ItemParams[];
  lost_sale_penalty_share: number;
  waste_disposal_share: number;
  order_line_cost: number;
  truck_trip_cost: number;
  truck_capacity_cases: number;
  store_min_order_cases: number;
  shelf_days: number;
  shelf_days_perishable: number;
  shelf_min_cases: number;
  space_factor: number | null;
  budget_factor: number | null;
  time_limit_s: number | null;
  service_floor: { enabled: boolean; share: number | null; gap_penalty_factor: number | null };
  spoilage_cap_in_ilp: boolean;
  stock_value_cap: boolean;
  closed_days: string[];
}

export interface ItemsFile {
  items: ItemInfo[];
  params: Record<PeriodKey, PeriodParams>;
  stores: string[];
  source: string;
}

/** One store-item row of a day file, in the column order of `cols`. */
export type DayRow = [
  number, number, number, number, // q50a q80a q90a q95a (today)
  number | null, number | null, number | null, number | null, // q50b..q95b (tomorrow)
  number, number, // act, pos
  number | null, number | null, // line, cap (null: no order possible / long-life cap)
  number, number, number, // cases, lost, waste
  number, number, number, number, // alt orders (cartons): last_week, topup_avg, topup_forecast, store_model
  (number | null)?, (number | null)?, (number | null)?, // optional holiday metrics: h_line, h_cap, h_cases
];

export interface DayFile {
  date: string;
  period: PeriodKey;
  can_order: boolean;
  closed: boolean;
  cover: string[];
  fc_dates: string[];
  stores: Record<string, {
    truck?: [number, number, number];
    space?: [number, number, number];
    budget?: [number, number, number];
    trucks: number;
    cartons: number;
  }>;
  layers: [number, 0 | 1, number, number, number][];
  floor_on: boolean;
  floor_share: number | null;
  spoil_in_ilp: boolean;
  cols: string[];
  rows: DayRow[];
}

export interface SeriesFile {
  store: string;
  period: PeriodKey;
  dates: string[];
  items: Record<string, { q50: number[]; q80: number[]; q90: number[]; q95: number[]; act: number[]; stock: number[]; order: number[] }>;
}

export interface ShadowPeriod {
  limits: Record<string, { store_days: number; binding: number; value_max: number; use_mean: number; use_max: number }>;
  layers: Record<string, { rows: number; mean: number; max: number; median: number; store_items: number }>;
  floor_matrix: { stores: string[]; dates: string[]; cells: [number, number][][] };
  top_floor_items: { store: string; item: string; days: number }[];
  source: string;
}

export interface WhatIfRow {
  param: string;
  value: number;
  fill: number;
  lost: number;
  waste: number;
  stock: number;
  total: number;
  net_cost: number;
  penalty: number;
  shadow: Record<'truck' | 'space' | 'budget', [number, number]>;
}

export interface ShadowFile {
  periods: Record<PeriodKey, ShadowPeriod>;
  whatif: { rows: WhatIfRow[]; source: string; note: string };
}

type Score = { mae: number; cover_q50: number; cover_q80: number; cover_q90: number; cover_q95: number; n?: number; pinball_q50?: number; pinball_q80?: number; pinball_q90?: number; pinball_q95?: number };

export interface ForecastFile {
  folds: Record<string, { model: Score; naive: Score; stub: Score; model_perishable: Score }>;
  fold_info: Record<string, { start: string; end: string }>;
  holdout: Record<'b1' | 'b2', { days: number; rows: number; mae: number; cover_q80: number; cover_q90: number; cover_q95: number; source: string }>;
  test: { all: Score; perishable: Score; long_life: Score; note: string };
  calibration: { shifts: Record<string, number>; tolerance_pts: number };
  sources: string[];
}

/* ---- loading ---- */

const cache = new Map<string, Promise<unknown>>();

export function loadJSON<T>(path: string): Promise<T> {
  const url = `${import.meta.env.BASE_URL}data/${path}`;
  if (!cache.has(url)) {
    const p = fetch(url).then((res) => {
      if (!res.ok) throw new Error(`Could not load ${path} (HTTP ${res.status}).`);
      return res.json();
    });
    p.catch(() => cache.delete(url));
    cache.set(url, p);
  }
  return cache.get(url) as Promise<T>;
}

export type Loaded<T> = { data: T | null; error: string | null; loading: boolean; retry: () => void };

export function useJSON<T>(path: string | null): Loaded<T> {
  const [state, setState] = useState<{ data: T | null; error: string | null; path: string | null }>({ data: null, error: null, path: null });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (!path) return;
    let alive = true;
    loadJSON<T>(path).then(
      (data) => alive && setState({ data, error: null, path }),
      (err: Error) => alive && setState({ data: null, error: err.message, path }),
    );
    return () => {
      alive = false;
    };
  }, [path, nonce]);
  const current = state.path === path;
  return {
    data: current ? state.data : null,
    error: current ? state.error : null,
    loading: !!path && (!current || (!state.data && !state.error)),
    retry: () => setNonce((n) => n + 1),
  };
}

export const METHOD_SHORT: Record<MethodId, string> = {
  last_week: 'Habit',
  topup_avg: 'Top-up avg',
  topup_forecast: 'Top-up forecast',
  store_model: 'Store model',
  store_ilp: 'Store ILP',
};

export const PERIOD_SHORT: Record<PeriodKey, string> = { test: 'Test weeks', b1: 'Holdout 1', b2: 'Holdout 2' };
