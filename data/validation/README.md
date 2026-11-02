# Validation input snapshot

Exact source schemas and values, restricted to this L2 validator's inputs:

- sales.parquet: 100,800 rows = 33,600 frozen calibration rows for
  28 September–25 October 2015 plus all 67,200 test rows for
  1 November–26 December 2015.
- forecasts_stub.parquet: all 470,400 supplied stub rows, 56 origins.
- forecasts_stub_small.parquet: all 980 supplied small-case rows, seven origins.
- Original full/small starting-state CSVs and 120-product item metadata.

No holdout sales are included. Forecasts extend to 1 January 2016; those are
predictions, not holdout actuals. The shared stub is unchanged. This snapshot
does not contain the complete training history or forecaster code.
