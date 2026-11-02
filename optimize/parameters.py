"""Frozen pre-test calibration and common initial-state preparation for L2."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model import State


@dataclass
class ParameterBundle:
    config: dict
    item_parameters: pd.DataFrame
    store_parameters: pd.DataFrame
    manifest: dict


@dataclass
class PlanningCase:
    state: State
    forecasts: pd.DataFrame
    parameters: ParameterBundle
    initialization_log: pd.DataFrame
    source_files: dict


# Preserve the earlier internal name for checkpoint-2 callers.
SmallCase = PlanningCase


def _positive_integer(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{label} must be a positive integer')
    return value


def _finite(value, label, *, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{label} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def compile_parameters(sales, items, cfg, store_ids, item_ids):
    """Use only frozen pre-test records; never mutate input frames or cfg.

    Store demand means use calendar days including configured zero-imputed
    absences. Item prices are medians over active-store/day records. Pack bands
    and service categories use the supplied pre-test global item metadata.
    """
    config = deepcopy(cfg)
    om = config['order_model']
    calibration = om['calibration']
    if calibration['freeze_at'] != 'test_start':
        raise ValueError('This adapter freezes calibration at test_start')
    if calibration['price_aggregation'] != 'median_active_store_days':
        raise ValueError('Unsupported price aggregation')
    if calibration['missing_sales_policy'] != 'zero':
        raise ValueError('Missing sales require an explicit supported policy')
    average_days = _positive_integer(calibration['average_window_days'], 'average window')
    price_days = _positive_integer(calibration['price_window_days'], 'price window')
    lag = _positive_integer(calibration['observation_lag_days'], 'observation lag')
    reference = pd.Timestamp(config['data']['windows']['test_start']).normalize()
    end = reference-pd.Timedelta(days=lag)
    average_start = end-pd.Timedelta(days=average_days-1)
    price_start = end-pd.Timedelta(days=price_days-1)
    if min(average_start, price_start) < pd.Timestamp(config['data']['history_start']):
        raise ValueError('Calibration window precedes declared history')
    stores, active_items = list(store_ids), list(item_ids)
    if not stores or not active_items or len(set(stores)) != len(stores) or len(set(active_items)) != len(active_items):
        raise ValueError('Active locations/items must be nonempty and unique')
    homes = config['network']['home_dc']
    if any(s not in homes for s in stores):
        raise ValueError('Unknown store in calibration')
    active_dcs = [d for d in config['network']['dcs'] if d in {homes[s] for s in stores}]
    if set(active_dcs) != {homes[s] for s in stores}:
        raise ValueError('Home DC must be declared in network.dcs')

    item_columns = ['item_id', 'rank', 'avg_daily_units', 'is_perishable', 'shelf_life_days']
    if not set(item_columns) <= set(items):
        raise ValueError('Item metadata columns are incomplete')
    metadata = items.loc[items.item_id.isin(active_items), item_columns].copy()
    if metadata.item_id.duplicated().any() or set(metadata.item_id) != set(active_items):
        raise ValueError('Item metadata must be complete and unique')
    metadata = metadata.set_index('item_id').loc[active_items]
    required = ['date', 'store_id', 'item_id', 'units', 'price']
    if not set(required) <= set(sales):
        raise ValueError('Sales calibration columns are incomplete')
    history = sales.loc[sales.store_id.isin(stores) & sales.item_id.isin(active_items), required].copy()
    history['date'] = pd.to_datetime(history.date).dt.normalize()
    history = history.loc[(history.date >= min(average_start, price_start)) & (history.date <= end)]
    if history.duplicated(['date', 'store_id', 'item_id']).any():
        raise ValueError('Duplicate pre-test sales keys')
    if not np.isfinite(history.units.to_numpy(dtype=float)).all() or (history.units < 0).any():
        raise ValueError('Calibration sales must be finite and nonnegative')
    demand_history = history[history.date >= average_start]
    price_history = history[history.date >= price_start]
    if price_history.empty or not np.isfinite(price_history.price.to_numpy(dtype=float)).all() or (price_history.price <= 0).any():
        raise ValueError('Reference price must be available, finite and positive in the fixed window')
    prices = price_history.groupby('item_id').price.median()
    if set(prices.index) != set(active_items):
        raise ValueError('Missing item reference price; future prices are not a fallback')
    grid = pd.MultiIndex.from_product([stores, active_items], names=['store_id', 'item_id'])
    totals = demand_history.groupby(['store_id', 'item_id']).units.sum().reindex(grid, fill_value=0)
    counts = demand_history.groupby(['store_id', 'item_id']).size().reindex(grid, fill_value=0)
    averages = totals/average_days

    bands = om['case_size_bands']
    previous = -math.inf
    for upper, units in bands:
        if float(upper) <= previous:
            raise ValueError('Case-size bands must be strictly increasing')
        _positive_integer(units, 'units per case')
        previous = float(upper)
    if not bands or not math.isinf(float(bands[-1][0])) or float(bands[-1][0]) < 0:
        raise ValueError('Case-size bands must end with a positive infinity bound')
    pallet_cases = _positive_integer(om['cases_per_pallet'], 'cases per pallet')
    levels = om['service_levels']
    buy_share = _finite(om['buy_cost_share_of_price'], 'purchase share', positive=True)
    holding_rate = _finite(om['holding_rate_per_year'], 'holding rate')
    days_year = _positive_integer(om['holding_days_per_year'], 'holding days per year')
    horizon = _positive_integer(om['horizon_days'], 'horizon')
    budget_factor = _finite(om['budget_factor'], 'budget factor')
    compiled_items, item_records, store_records = {}, [], []
    for p in active_items:
        row = metadata.loc[p]
        if not isinstance(row.is_perishable, (bool, np.bool_)):
            raise ValueError('is_perishable must be boolean, not a truthy string')
        perishable = bool(row.is_perishable)
        item_average = _finite(row.avg_daily_units, 'global item average')
        rank = _finite(row['rank'], 'item rank', positive=True)
        if rank != int(rank):
            raise ValueError('Item rank must be integer')
        case = next(units for upper, units in bands if item_average <= float(upper))
        if perishable:
            raw_life = _finite(row.shelf_life_days, 'shelf life', positive=True)
            if raw_life != int(raw_life):
                raise ValueError('Shelf life must be a whole number of days')
            life = int(raw_life)
            level = levels['perishable']
        else:
            life = None
            level = (levels['top_n_staples']['level'] if rank <= levels['top_n_staples']['n'] else
                     levels['slow_movers']['level'] if item_average < levels['slow_movers']['below_avg_daily_units'] else
                     levels['default'])
        if level not in (0.5, 0.8, 0.9, 0.95):
            raise ValueError('Service level must match a supplied quantile')
        price = float(prices[p])
        cost = price*buy_share
        compiled_items[p] = dict(units_per_case=int(case), cases_per_pallet=pallet_cases,
                                 price=price, buy_cost=cost,
                                 holding_cost_per_unit_day=cost*holding_rate/days_year,
                                 service_level=float(level), shelf_life_days=life)
        item_records.append(dict(item_id=p, rank=int(rank), global_avg_daily_units=item_average,
                                 is_perishable=perishable, **compiled_items[p]))
        shelf_days = om['shelf_days_perishable'] if perishable else om['shelf_days']
        for s in stores:
            average = float(averages.loc[(s, p)])
            capacity = max(_finite(shelf_days, 'shelf days')*average,
                           _finite(om['shelf_min_cases'], 'shelf minimum cartons')*case)
            store_records.append(dict(store_id=s, item_id=p, avg_daily_units=average,
                                      shelf_capacity_units=capacity, units_per_case=int(case),
                                      observed_calendar_days=int(counts.loc[(s, p)]),
                                      imputed_zero_days=int(average_days-counts.loc[(s, p)])))
    parameters = dict(
        items=compiled_items,
        store_averages={s: {p: float(averages.loc[(s, p)]) for p in active_items} for s in stores},
        dc_budgets={d: budget_factor*horizon*sum(float(averages.loc[(s, p)])*compiled_items[p]['buy_cost']
                    for s in stores if homes[s] == d for p in active_items) for d in active_dcs},
    )
    canonical = history.sort_values(['date', 'store_id', 'item_id']).copy()
    canonical['date'] = canonical.date.dt.strftime('%Y-%m-%d')
    canonical['units'] = canonical.units.astype(float)
    canonical['price'] = canonical.price.astype(float)
    payload = dict(history=canonical.to_dict('records'), parameters=parameters)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    manifest = dict(reference_origin=reference.date().isoformat(), average_start=average_start.date().isoformat(),
                    average_end=end.date().isoformat(), price_start=price_start.date().isoformat(),
                    price_end=end.date().isoformat(), average_window_days=average_days,
                    price_window_days=price_days, observation_lag_days=lag,
                    price_aggregation=calibration['price_aggregation'],
                    missing_calendar_rows=int((average_days-counts).sum()),
                    observed_calibration_rows=len(history), stores=stores, items=active_items,
                    calibration_sha256=digest,
                    assumptions='Frozen calendar-day local averages; missing rows zero; median active-store-day item price.')
    om['parameters'] = parameters
    om['calibration_manifest'] = manifest
    return ParameterBundle(config, pd.DataFrame(item_records), pd.DataFrame(store_records), manifest)


def initialize_state(store_stock, dc_stock, pipeline, bundle):
    """Prepare the common opening state once; do not call during daily replanning."""
    reference = pd.Timestamp(bundle.manifest['reference_origin']).date()
    stores = store_stock.copy(deep=True)
    dcs = dc_stock.copy(deep=True)
    for frame, location_column in [(stores, 'store_id'), (dcs, 'dc_id')]:
        if not {'date', location_column, 'item_id', 'units'} <= set(frame):
            raise ValueError('Initial stock columns are incomplete')
        if set(pd.to_datetime(frame.date).dt.date) != {reference}:
            raise ValueError('Initial stock date must match the frozen reference origin')
        if frame.duplicated([location_column, 'item_id']).any():
            raise ValueError('Duplicate opening-stock keys')
        if not np.isfinite(frame.units.to_numpy(dtype=float)).all() or (frame.units < 0).any():
            raise ValueError('Invalid initial stock')
    caps = bundle.store_parameters[['store_id', 'item_id', 'shelf_capacity_units']]
    prepared = stores.merge(caps, on=['store_id', 'item_id'], how='left', validate='one_to_one')
    if prepared.shelf_capacity_units.isna().any() or len(prepared) != len(caps):
        raise ValueError('Opening store-stock grid must match calibrated active store-items')
    clipped = np.minimum(prepared.units, prepared.shelf_capacity_units)
    # Preserve physical whole units when the supplied initial state is integral;
    # the model itself still permits continuous forecast inventory trajectories.
    integral = np.equal(prepared.units, np.floor(prepared.units))
    rounding = bundle.config['order_model']['initialization']['integral_clip_rounding']
    if rounding != 'floor':
        raise ValueError('Unsupported integral initial-stock clipping rule')
    clipped = np.where(integral, np.floor(clipped), clipped)
    log = prepared[['date', 'store_id', 'item_id', 'units', 'shelf_capacity_units']].rename(columns={'units': 'supplied_units'})
    log['initialized_units'] = clipped
    log['clipped_units'] = log.supplied_units-log.initialized_units
    log['accounting_category'] = 'initial_state_adjustment'
    prepared['units'] = clipped
    clean_store = prepared[['store_id', 'item_id', 'units']].copy()
    clean_dc = dcs[['dc_id', 'item_id', 'units']].copy()
    return State(reference, clean_store, clean_dc, pipeline.copy(deep=True)), log


def load_case(input_dir, config_path, profile='small', forecast_path=None):
    """Load one origin for the small or full main-product profile, not a replay."""
    input_dir = Path(input_dir)
    cfg = yaml.safe_load(Path(config_path).read_text())
    if profile not in {'small', 'full'}:
        raise ValueError('Unknown planning profile')
    if cfg['data']['product_set'] != 'main':
        raise NotImplementedError('Random product profiles are a later robustness checkpoint')
    suffix = '_small' if profile == 'small' else ''
    filenames = dict(sales='sales.parquet', items='items.csv',
                     forecasts=f'forecasts_stub{suffix}.parquet',
                     store_stock=f'state_start_store_stock{suffix}.csv',
                     dc_stock=f'state_start_dc_stock{suffix}.csv',
                     pipeline=f'state_start_pipeline{suffix}.csv')
    paths = {key: input_dir/value for key, value in filenames.items()}
    if forecast_path is not None:
        paths['forecasts'] = Path(forecast_path)
    store_stock = pd.read_csv(paths['store_stock'])
    dc_stock = pd.read_csv(paths['dc_stock'])
    pipeline = pd.read_csv(paths['pipeline'])
    stores = store_stock.store_id.drop_duplicates().tolist()
    active_items = store_stock.item_id.drop_duplicates().tolist()
    expected_stores = cfg['stubs']['small']['stores'] if profile == 'small' else cfg['data']['stores']
    expected_count = cfg['stubs']['small']['n_items'] if profile == 'small' else cfg['data']['main']['n_items']
    if set(stores) != set(expected_stores) or len(active_items) != expected_count:
        raise ValueError(f'{profile} initial-state dimensions disagree with declared profile')
    items = pd.read_csv(paths['items'])
    expected_items = set(items.sort_values('rank').head(expected_count).item_id)
    if set(active_items) != expected_items:
        raise ValueError(f'{profile} profile must use the declared top-ranked pre-test items')
    sales = pd.read_parquet(paths['sales'], columns=['date', 'store_id', 'item_id', 'units', 'price'])
    bundle = compile_parameters(sales, items, cfg, stores, active_items)
    state, log = initialize_state(store_stock, dc_stock, pipeline, bundle)
    forecasts = pd.read_parquet(paths['forecasts'])
    forecasts = forecasts[pd.to_datetime(forecasts.origin).dt.date == state.date].copy()
    return PlanningCase(state, forecasts, bundle, log, {key: str(path.resolve()) for key, path in paths.items()})


def load_small_case(input_dir, config_path):
    return load_case(input_dir, config_path, profile='small')


def load_full_case(input_dir, config_path):
    return load_case(input_dir, config_path, profile='full')
