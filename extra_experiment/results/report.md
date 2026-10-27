# Additional recorded-input experiments

Research simulations only. No insulin dosing recommendation or clinical efficacy claim.

## Input audit

| Case | Rows / unique times | Meals / boluses | Max gap (min) | Reference BG |
|---|---:|---:|---:|---:|
| case_01 | 30 / 29 | 1 / 3 | 40 | 79 |
| case_02 | 15 / 15 | 3 / 3 | 60 | 65 |

## Analysis windows

- case_01: first_meal, 06:29–09:24; 30 of 202 source rows. First meal: 06:58.
  Fit loss: soft_l1; 4 starts (local); max evaluations/start: 200.
- case_02: full_record, 07:30–13:15; 15 of 15 source rows. First meal: 07:30.
  Fit loss: linear; 8 starts (mixed); max evaluations/start: 500.
- The first-meal window is selected by logged Eating times, not glucose values or fit quality. No endpoints or missing observations are interpolated. Its fit metrics do not evaluate the rest of the day and are not directly comparable with full-day scores.

## In-sample fit (all times within each analysis window, one parameter vector per case)

| Case | N | RMSE | MAE | R² | Optimizer success |
|---|---:|---:|---:|---:|---|
| case_01 | 29 | 6.490 | 3.993 | 0.979 | True |
| case_02 | 15 | 7.164 | 5.761 | 0.847 | True |

## Paired simulations with the same recorded meals and boluses

| Case / scenario | Objective | Peak BG | Min BG | Time 70–110 (%) | Below 70 (%) |
|---|---:|---:|---:|---:|---:|
| case_01 / Recorded boluses only | 7866.11 | 209.00 | 79.00 | 19.71 | 0.00 |
| case_01 / Recorded boluses + optimized unitless control | 5381.42 | 197.56 | 76.94 | 20.02 | 0.00 |
| case_02 / Recorded boluses only | -335.51 | 121.05 | 65.00 | 62.91 | 0.87 |
| case_02 / Recorded boluses + optimized unitless control | -335.53 | 121.05 | 65.00 | 62.91 | 0.87 |

## Case-specific findings

- case_01: data-only log-parameter Jacobian condition number 2.82e+08; inspect parameter boundaries and multistart variability before interpreting individual coefficients.
- case_02: data-only log-parameter Jacobian condition number 4.21e+08; inspect parameter boundaries and multistart variability before interpreting individual coefficients.
- case_02: optimized time below 70 increases by 0.000083 percentage points despite a lower objective. The paper objective does not enforce a safety constraint.
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
