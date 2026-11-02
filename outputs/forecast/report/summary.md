# Forecaster scores for the slides

All numbers are on the two validation folds (the test window is never scored here). Scores leave out 25 Dec: stores are closed, every line is 0 and there are no sales.

## Error (MAE and pinball, lower is better)

| fold | method | MAE | pinball q50 | pinball q80 | pinball q90 | pinball q95 |
|---|---|---|---|---|---|---|
| fold1_festive | naive baseline | 4.883 | 2.442 | 1.971 | 1.398 | 0.964 |
| fold1_festive | stub (4-week average) | 4.389 | 2.195 | 1.805 | 1.284 | 0.883 |
| fold1_festive | model | 3.550 | 1.775 | 1.554 | 1.102 | 0.727 |
| fold2_main | naive baseline | 5.210 | 2.605 | 2.077 | 1.464 | 1.003 |
| fold2_main | stub (4-week average) | 4.725 | 2.362 | 1.939 | 1.374 | 0.943 |
| fold2_main | model | 3.817 | 1.908 | 1.572 | 1.094 | 0.712 |

Model MAE against the stub (the order model's placeholder):

- fold1_festive: 3.550 vs 4.389, 19.1% lower
- fold2_main: 3.817 vs 4.725, 19.2% lower

## Coverage (share of days with sales at or below each line)

| fold | q50 (target 50%) | q80 (target 80%) | q90 (target 90%) | q95 (target 95%) |
|---|---|---|---|---|
| fold1_festive | 56.7% | 82.0% | 90.4% | 94.5% |
| fold2_main | 54.2% | 80.9% | 90.2% | 94.7% |

Calibration numbers (Task 7, tolerance 3 points): q50 = 0, q80 = 1, q90 = 1, q95 = 1. All lines were already within tolerance, so nothing was stretched.

## Notes

- 25 Dec: left out of every score, for the model and both baselines alike. The forecasts file still has 25 Dec rows, with all four lines set to 0.
- q50 coverage above 50% on the festive fold comes from 0 = 0 ties: on a day with no sales and a q50 of 0 the day counts as covered. Counting ties as half gives about 51%.
- MAE = 2 x the q50 pinball, so the two q50 columns rank methods the same way.

Example store-item chart: example_CA_1_FOODS_3_090.png

Charts: mae_vs_baselines.png, coverage.png
