"""Past sales a simple rule may read on day d: only days d-history_days .. d-1.

The engine builds one PastSales per day from rows strictly before that day and
hands only this object to the rule, so a rule cannot see today's or later sales
(the future-poison tests check it). Asking for a day >= d raises.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd


class PastSales:
    def __init__(self, sales: pd.DataFrame, before: date, days: int, missing_units: float):
        self.before = before
        self.first = before - timedelta(days=int(days))
        d = pd.to_datetime(sales.date).dt.date
        rows = sales.loc[(d >= self.first) & (d < before), ['store_id', 'item_id', 'units']].copy()
        rows['date'] = d[rows.index]
        self.missing_units = float(missing_units)
        # wide table: one row per date, one column per (store, item); a missing row = missing_units
        self._table = rows.pivot_table(index='date', columns=['store_id', 'item_id'], values='units',
                                       aggfunc='sum') if len(rows) else pd.DataFrame()

    def matrix(self, days, keys) -> np.ndarray:
        """Units for each day (rows) and (store, item) key (columns)."""
        days = list(days)
        for day in days:
            if day >= self.before:
                raise ValueError(f'Sales for {day} requested on {self.before}: not before the decision day')
            if day < self.first:
                raise ValueError(f'Sales for {day} are older than the history handed over (from {self.first})')
        if self._table.empty:
            return np.full((len(days), len(keys)), self.missing_units)
        index = pd.MultiIndex.from_tuples(list(keys), names=['store_id', 'item_id'])
        out = self._table.reindex(index=days, columns=index)
        return out.fillna(self.missing_units).to_numpy(dtype=float)
