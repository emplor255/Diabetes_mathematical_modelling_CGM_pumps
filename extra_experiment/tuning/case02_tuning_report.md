# Case 02 fitting retune and conditional sensitivity

Retuning first used 16 soft-L1 starts (8 near nominal, 8 log-uniform over the original bounds), then 8 ordinary least-squares starts (4 near nominal, 4 log-uniform), up to 500 function evaluations per start. Same 15 observations, model, bounds, fixed K/reference/absorption assumptions and control settings. Ordinary least squares directly minimizes in-sample RMSE but is less robust to outliers; robust-loss costs must not be compared numerically against ordinary-least-squares costs. No insulin responsiveness constraint or control-performance-based parameter selection.

| Fit | RMSE | MAE | R² | beta1 |
|---|---:|---:|---:|---:|
| Before (4 local starts, soft-L1) | 7.22987 | 5.69247 | 0.84401 | 1e-06 |
| 16 mixed starts, soft-L1 | 7.22993 | 5.69233 | 0.84401 | 1e-06 |
| Main: 8 mixed starts, linear | 7.16448 | 5.76103 | 0.84682 | 1e-06 |

The main result is selected for glucose-fitting error, not for a visually stronger control response. A lower in-sample RMSE is not evidence of better out-of-sample prediction or identification of physiological parameters. Previous results are retained in `case02_before/` and `case02_robust_multistart/`.

RMSE changes by 0.905% relative to the original fit; MAE changes from 5.69247 to 5.76103 mg/dL. This is a tradeoff between fitting losses, not uniform improvement across metrics.
The main beta1 remains at its lower bound. Retuning has not resolved the weak modeled response to additional control or the parameter-identification problem.

## Conditional beta1 profile (not confidence intervals)

At each fixed beta1, the other six parameters are refitted from three starts: the original fitted vector, the nominal vector, and the preceding grid solution. The smallest robust-loss cost is retained, not the best control response. This is a bounded numerical search, not a proof of the global profile minimum.

| Fixed beta1 | Robust cost | RMSE | R² | Max BG change under constant u=10 |
|---|---:|---:|---:|---:|
| 1e-06 | 9.18261 | 7.22991 | 0.84401 | 0.07186 |
| 1e-05 | 9.19564 | 7.23274 | 0.84389 | 0.12630 |
| 0.0001 | 9.27309 | 7.26230 | 0.84261 | 0.68958 |
| 0.001 | 9.60927 | 7.42030 | 0.83568 | 5.33519 |
| 0.01 | 10.70272 | 7.96298 | 0.81077 | 33.11913 |
| 0.1 | 11.86591 | 8.48595 | 0.78510 | 100.34241 |

The forced u=10 simulations are a sensitivity diagnostic, not optimized controls or insulin doses. Profile solutions do not replace the main fit. Similar glucose errors with different control responses indicate that a good in-sample fit alone does not establish a reliable response to intervention. Conversely, a worse fit at a selected beta1 is not justification to force that value for a prettier control plot.

Only 15 points are available, with no premeal baseline. Reference BG=65 and the dimensionless insulin scale remain strong assumptions. No clinical insulin sensitivity or dose recommendation can be inferred from these fits.
