# Additional full-record experiments

Research simulations only. No insulin dosing recommendation or clinical efficacy claim.

## Input audit

| Case | Rows / unique times | Meals / boluses | Max gap (min) | Reference BG |
|---|---:|---:|---:|---:|
| case_01 | 202 / 201 | 5 / 26 | 60 | 79 |
| case_02 | 15 / 15 | 3 / 3 | 60 | 65 |

## In-sample fit (all recorded times, one parameter vector per case)

| Case | N | RMSE | MAE | R² | Optimizer success |
|---|---:|---:|---:|---:|---|
| case_01 | 201 | 31.897 | 22.135 | -0.336 | True |
| case_02 | 15 | 7.230 | 5.692 | 0.844 | True |

## Paired simulations with the same recorded meals and boluses

| Case / scenario | Objective | Peak BG | Min BG | Time 70–110 (%) | Below 70 (%) |
|---|---:|---:|---:|---:|---:|
| case_01 / Recorded boluses only | 7659.72 | 147.23 | 84.48 | 42.51 | 0.00 |
| case_01 / Recorded boluses + optimized unitless control | -2703.98 | 107.17 | 68.76 | 99.43 | 0.57 |
| case_02 / Recorded boluses only | -267.90 | 122.22 | 65.00 | 59.41 | 1.89 |
| case_02 / Recorded boluses + optimized unitless control | -267.91 | 122.22 | 65.00 | 59.41 | 1.89 |

## Case-specific findings

- case_01: R²=-0.336 is negative: the fitted trajectory has greater squared error than the observed-mean constant predictor. Control results from this poorly fitting model must not be presented as evidence of effectiveness.
- case_01: data-only log-parameter Jacobian condition number 3.68e+05; inspect parameter boundaries and multistart variability before interpreting individual coefficients.
- case_01: optimized time below 70 increases by 0.567157 percentage points despite a lower objective. The paper objective does not enforce a safety constraint.
- case_02: data-only log-parameter Jacobian condition number 1.3e+08; inspect parameter boundaries and multistart variability before interpreting individual coefficients.
- case_02: optimized time below 70 increases by 0.000319 percentage points despite a lower objective. The paper objective does not enforce a safety constraint.
- case_02: objective improvement is less than 0.1 in its native units; the two simulated glucose curves are essentially coincident. No meaningful clinical benefit can be inferred.

## Interpretation and limitations

- BG is provisionally interpreted as mg/dL; source CSVs do not declare units. Eating and bolus amounts are used in native units, with fitted effective gains. Do not interpret gains as physiological dose-conversion factors.
- The seven freely fitted parameters are five shared kinetic coefficients and two input gains. K=120 mg/dL and basal insulin=1 are fixed; beta2 is derived at the assumed reference.
- A fourth, first-order bolus depot is an explicit extension of the paper model, not part of the Hall2018 fit. Its absorption rate is assumed to be 1/h, not estimated or calibrated for a drug. Inputs are impulses, not interpolated dose rates.
- No measured insulin concentration, basal-dose history, insulin-on-board history, or diagnostic group is supplied. Initial digestive/bolus depots are assumed zero. Initial quick insulin is the reference equilibrium. These are strong assumptions.
- case_02 begins with BG=65 and a meal/bolus, without a premeal baseline; its first BG is used as reference by assumption. Fifteen points cannot establish reliable identification of seven parameters. See bound flags, multistart results, and Jacobian diagnostics.
- Noise labels are preserved, not assigned undocumented reliability weights. Duplicate-time BG is averaged, meal/bolus amounts are summed. No observations are discarded for a poor fit; no interpolated observations enter the fit.
- All scores are in-sample. There is no holdout validation, clinical validation, uncertainty propagation, or between-patient statistical inference from these two records.
- The comparator retains recorded boluses. Optimization adds nonnegative dimensionless control directly to the model insulin equation; it does not recommend replacing recorded boluses. All future recorded inputs are known: this is offline open-loop optimization.
- The original paper objective has no hard hypoglycaemia constraint and may be negative. A smaller objective does not guarantee safer glucose. Inspect minimum BG and time below 70, including any worsening. Duration metrics use piecewise-linear trajectories, not sample counts.
- A zero-control fallback is always a candidate. Optimizer status is exported even when a nonconverged candidate has the lowest finite objective.
- Integration is rechecked with tighter tolerances; numerical_check.csv records the maximum glucose difference for each scenario.
