from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml


def load_tiny(path=None):
    """Build the artificial case solely from its YAML fixture."""
    from optimize import State

    path = path or Path(__file__).with_name('tiny.yaml')
    cfg = yaml.safe_load(Path(path).read_text())
    fixture = cfg['example']
    origin = date.fromisoformat(fixture['date'])
    store = next(iter(cfg['network']['home_dc']))
    dc = cfg['network']['home_dc'][store]
    records = []
    for item, demands in fixture['demand'].items():
        for offset, demand in enumerate(demands):
            records.append(dict(origin=origin, date=origin + timedelta(days=offset),
                                store_id=store, item_id=item,
                                **{q: demand for q in ('q50', 'q80', 'q90', 'q95')}))
    state = State(
        date=origin,
        store_stock=pd.DataFrame([dict(store_id=store, item_id=p, units=n)
                                  for p, n in fixture['store_stock'].items()]),
        dc_stock=pd.DataFrame([dict(dc_id=dc, item_id=p, units=n)
                               for p, n in fixture['dc_stock'].items()]),
        pipeline=pd.DataFrame(columns=['order_date', 'arrival_date', 'location_type',
                                       'location_id', 'item_id', 'units']),
    )
    return state, pd.DataFrame(records), cfg
