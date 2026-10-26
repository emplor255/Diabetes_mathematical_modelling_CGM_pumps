# Reproduction report

## Data used

- Hall2018: 105,417 CGM rows from 57 participants (38 healthy, 14 prediabetes, 5 type 2 diabetes).
- Standardized meals: 6,152 input rows, 176 event windows, and 147 retained full-span windows (internal gaps may remain).
- QC: 87 duplicate timestamp rows collapsed by their mean; 6 `Low` readings approximated by 40 mg/dL with weight 0.8 (not exact measured concentrations).

Dataset coverage: [overview figure](figures/figure_6_dataset_overview.png), [cohort counts](tables/dataset_overview.csv), and [pointwise Figure 2 sample sizes](tables/meal_profile_sample_sizes.csv). The meal cohort is a nested subset of Hall2018, not an independent validation set. Figure 2 shows observed participant-averaged glucose and pointwise mean confidence intervals, not model fits, individual-response bands, or controller performance.

## Parameter estimation and ablation

- Fitted participants: 30.
- Proposed-model median participant RMSE: 9.14 mg/dL.
- Reduced two-state median participant RMSE: 9.71 mg/dL.
- The proposed model had lower BIC for 46.7% of participants.
- BIC therefore does not support a general advantage of the proposed model over the reduced ablation.
- 28/30 fits had at least one kinetic parameter at a bound.
- Specifically, beta0 was near its lower bound in 24/30 fits and gamma was near its upper bound in 21/30 fits.
- 30/30 fits had normalized kinetic-Jacobian condition number >100 or infinite.

The last two diagnostics are essential: CGM alone does not identify hidden insulin dynamics reliably. Group-level parameter differences in the CSV tables are exploratory, not mechanistic proof.

### Cohort-wide fit evaluation

| Outcome | Proposed 3-state | Reduced 2-state |
| --- | --- | --- |
| Participants | 30 | 30 |
| Meal windows | 147 | 147 |
| CGM observations | 4,452 | 4,452 |
| Pooled RMSE (mg/dL) | 10.83 | 11.26 |
| Pooled MAE (mg/dL) | 7.00 | 7.33 |
| Participant RMSE, median [Q1, Q3] | 9.14 [7.72, 10.97] | 9.71 [7.88, 11.38] |
| Participant MAE, median [Q1, Q3] | 6.53 [5.17, 7.65] | 7.35 [5.21, 7.85] |
| Within-event R2 | 0.665 | 0.638 |
| Event R2, median [Q1, Q3] | 0.615 [0.287, 0.853] | 0.612 [0.280, 0.819] |
| Events with R2 < 0 | 22/147 | 24/147 |
| Lower participant RMSE | 22/30 | 8/30 |
| Lower nominal AIC | 17/30 | 13/30 |
| Lower nominal BIC | 14/30 | 16/30 |

| Parameter | Healthy | Prediabetes | Type 2 diabetes | T2D - Healthy median |
| --- | --- | --- | --- | --- |
| alpha | 2.79 [0.899, 4.65] | 1.3 [0.654, 1.75] | 1.49 [1.25, 3.31] | -1.3 |
| beta0 | 0.0001 [0.0001, 0.016] | 0.0001 [0.0001, 0.000101] | 0.000105 [0.000103, 0.0117] | +4.54e-06 |
| beta1 | 0.526 [0.43, 0.742] | 0.228 [0.152, 0.364] | 0.195 [0.132, 0.345] | -0.331 |
| beta2_derived | 470 [260, 608] | 386 [298, 556] | 704 [485, 751] | +234 |
| gamma | 9.99 [9.82, 10] | 10 [9.8, 10] | 10 [9.64, 10] | +0.00493 |
| delta | 2.91 [1.38, 6.82] | 1.79 [0.968, 1.94] | 1.4 [0.936, 3.2] | -1.52 |

All fitted participants and meals are included. These are unweighted in-sample errors, not held-out validation. Pooled RMSE/MAE weight observations equally; participant summaries weight people equally. Within-event R² uses the sum of within-meal centered glucose sums of squares as its denominator. Quartiles are descriptive, not confidence intervals. Meal-time observations inform initialization. AIC/BIC are nominal criteria without a serial-correlation correction.

Parameter entries are median [Q1, Q3]; the final column is the difference of group medians. Prediabetes is not combined with T2D. beta2 is derived, insulin is uncalibrated, and boundary estimates / weak identifiability preclude strong physiological claims. No significance tests are claimed for these descriptive comparisons.

The [selected-fit figure](figures/figure_3_selected_model_fits.png) shows the highest event R² within each group, not typical cases. All observed points in each selected event are shown; curves connect the saved model predictions without smoothing the data. See [all-fit diagnostics](figures/figure_7_fit_diagnostics.png) for unselected cohort coverage.

Exports: [Excel workbook](tables/parameter_estimation_evaluation.xlsx), [overall fit table](tables/fit_overall.tex), [parameter comparison](tables/parameter_group_comparison.tex), [boundary diagnostics](tables/parameter_bound_diagnostics.tex).

## Optimal control

- Median reduction in the paper's objective: 99.68.
- Median integrated control intensity: 2.169 (unitless).
- 8/30 optimized profiles reached the imposed upper control bound for at least one integration step.

These are mathematical control intensities only. The source files contain neither insulin concentrations nor administered doses, so the result cannot be converted to units of insulin or used for treatment.

### Cohort-wide paired control comparison

Table 1. All 30 fitted participants; no outcome-based exclusions.

| Outcome (unit) | No control, mean (SD) | Optimized, mean (SD) | Paired change [95% CI] | Improved / unchanged / worsened |
| --- | --- | --- | --- | --- |
| Time in 70-110 mg/dL (%) | 66.75 (38.02) | 94.13 (12.48) | +27.37 [+16.56, +39.45] | 18/11/1 |
| Time above 110 mg/dL (%) | 32.83 (38.33) | 4.79 (12.51) | -28.04 [-39.87, -17.28] | 19/11/0 |
| Time below 70 mg/dL (%) | 0.42 (2.30) | 1.08 (3.16) | +0.66 [+0.00, +1.59] | 0/27/3 |
| Peak glucose (mg/dL) | 123.70 (33.74) | 102.60 (17.56) | -21.09 [-27.93, -14.77] | 26/4/0 |
| Paper objective J (model objective units) | 902.88 (2909.14) | -322.50 (451.05) | -1225.39 [-2216.09, -465.13] | 26/4/0 |
| Integrated control intensity (model intensity x h) | 0.00 (0.00) | 4.30 (4.83) | +4.30 [+2.71, +6.09] | -- |

Participants with any simulated glucose below 70 mg/dL: 1/30 without control versus 4/30 with control (3 newly affected). Minimum simulated glucose across participants: 67.50 versus 66.60 mg/dL. Optimizers reported success for 30/30; 8/30 reached the upper control bound.

### Breakdown by participant status

Healthy (n=20).

| Outcome (unit) | No control, mean (SD) | Optimized, mean (SD) | Paired change [95% CI] | Improved / unchanged / worsened |
| --- | --- | --- | --- | --- |
| Time in 70-110 mg/dL (%) | 83.01 (26.95) | 97.95 (5.05) | +14.93 [+5.60, +26.39] | 9/10/1 |
| Time above 110 mg/dL (%) | 16.36 (27.21) | 0.93 (4.15) | -15.43 [-26.79, -6.13] | 10/10/0 |
| Time below 70 mg/dL (%) | 0.63 (2.81) | 1.13 (3.25) | +0.50 [+0.00, +1.39] | 0/18/2 |
| Peak glucose (mg/dL) | 111.58 (22.76) | 96.21 (10.05) | -15.36 [-22.05, -9.41] | 16/4/0 |
| Paper objective J (model objective units) | -138.38 (689.63) | -417.23 (86.37) | -278.85 [-616.70, -68.71] | 16/4/0 |
| Integrated control intensity (model intensity x h) | 0.00 (0.00) | 2.31 (3.12) | +2.31 [+1.18, +3.79] | -- |

Prediabetes (n=7).

| Outcome (unit) | No control, mean (SD) | Optimized, mean (SD) | Paired change [95% CI] | Improved / unchanged / worsened |
| --- | --- | --- | --- | --- |
| Time in 70-110 mg/dL (%) | 38.22 (40.66) | 93.00 (7.97) | +54.78 [+26.30, +79.85] | 6/1/0 |
| Time above 110 mg/dL (%) | 61.78 (40.66) | 5.58 (8.24) | -56.20 [-79.85, -29.33] | 6/1/0 |
| Time below 70 mg/dL (%) | 0.00 (0.00) | 1.42 (3.75) | +1.42 [+0.00, +4.26] | 0/6/1 |
| Peak glucose (mg/dL) | 135.76 (28.66) | 106.85 (10.10) | -28.91 [-43.06, -15.87] | 7/0/0 |
| Paper objective J (model objective units) | 1400.59 (1860.45) | -372.24 (101.11) | -1772.83 [-3070.29, -620.85] | 7/0/0 |
| Integrated control intensity (model intensity x h) | 0.00 (0.00) | 6.61 (4.55) | +6.61 [+3.35, +9.62] | -- |

Type 2 diabetes (n=3).

| Outcome (unit) | No control, mean (SD) | Optimized, mean (SD) | Paired change [95% CI] | Improved / unchanged / worsened |
| --- | --- | --- | --- | --- |
| Time in 70-110 mg/dL (%) | 24.94 (31.44) | 71.30 (29.86) | +46.36 [+25.83, +73.51] | 3/0/0 |
| Time above 110 mg/dL (%) | 75.06 (31.44) | 28.70 (29.86) | -46.36 [-73.51, -25.83] | 3/0/0 |
| Time below 70 mg/dL (%) | 0.00 (0.00) | 0.00 (0.00) | +0.00 [+0.00, +0.00] | 0/3/0 |
| Peak glucose (mg/dL) | 176.34 (54.65) | 135.28 (33.68) | -41.07 [-59.14, -11.43] | 3/0/0 |
| Paper objective J (model objective units) | 6683.28 (6781.16) | 425.03 (1382.71) | -6258.26 [-11540.35, -479.03] | 3/0/0 |
| Integrated control intensity (model intensity x h) | 0.00 (0.00) | 12.11 (5.82) | +12.11 [+5.57, +16.73] | -- |

Table definitions and scope:

- Coverage: all 30 fitted participants, with one 2.5-hour simulated meal scenario per person. Each scenario uses that person's median fitted meal amplitude and delay, starting at reference glucose. The 147 fitted meal windows are used for parameter estimation; this table does not evaluate each observed meal separately or include Hall2018 participants without fitted meal data.
- Scenario values are participant means (sample SD), with equal weight per person. Change is optimized minus no control, computed within each person before aggregation. Time-percentage changes are percentage points, not relative percentages. Time percentages reuse the saved fractions of simulated samples on the 1-minute grid, including both endpoints, to match the existing control results.
- 95% CIs are percentile bootstrap intervals for the mean paired change, resampling participant pairs 10,000 times (seed 20260815). They describe between-participant sampling uncertainty conditional on fitted parameters; they do not propagate parameter, model, or optimizer uncertainty. Small subgroups are descriptive. No p-values are used to claim superiority.
- Improved/unchanged/worsened counts follow the favorable direction for each outcome: higher time in target; lower time above/below target, peak glucose, and objective. The numerical tie tolerance is 1e-06 in the metric's units. Peak reduction must be interpreted together with time below target. Control intensity is a resource metric with no improvement classification.
- The paper objective may be negative; its signed paired change in original units is reported. It equals a time-weighted squared distance from the target midpoint plus effort, up to a constant. Reducing it need not improve time in target or prevent glucose below 70 mg/dL. Because this is the quantity optimized, its reduction alone is not independent validation.
- The comparator is no external control. These are in-model, open-loop simulations with the same fitted dynamics used for optimization, not held-out predictions or observed treatment outcomes. The table does not establish superiority to another controller or clinical efficacy. Control intensity has model units, with no calibration to administered insulin dose.

Exports: [Excel tables](tables/control_evaluation.xlsx), [numeric cohort statistics](tables/control_cohort_statistics.csv), [paired participant changes](tables/control_paired_changes.csv), [LaTeX table](tables/control_evaluation.tex).

## Reproducibility notes

- Meal time is defined as 30 minutes after the beginning of a full-span 180-minute S6 Data window (supplement file identifier .s015).
- Only full-span windows with at least the configured minimum number of unique CGM points are fitted; internal gaps may remain.
- beta2 is derived from the participant's pre-meal equilibrium; K=120 mg/dL and basal insulin=1 are fixed because insulin is unobserved.
- The printed model's required absorption delay is implemented as an event-level fitted nuisance parameter.
- Random seeds, bounds, optimizer settings, event QC, predictions, and control trajectories are exported under `tables/`.

<!-- BEGIN ID_FIT_AND_DYNAMICS -->

## ID-ordered fitting performance and parameter exploration

Figures: [participant RMSE by ascending ID](figures/figure_8_fit_by_id.png) and [one-at-a-time parameter sweep](figures/figure_9_parameter_sweep.png). Both have vector PDF versions. The ID plot includes all fitted participants, not only favorable fits.

The dynamical reference is 2133-020 (Healthy); selection: Nearest participant RMSE to cohort median; ties by ascending natural ID. Each coefficient is multiplied by 0.5, 0.75, 1, 1.25, 1.5, with a 12-hour horizon and no control.

This is a strict one-coefficient-at-a-time experiment: the other coefficients, full initial state, meal amplitude and delay stay fixed. Nominal beta2 is derived once, then held fixed outside its own panel. Varying beta2 independently is an exploratory override only; it does not alter parameter fitting. Because the starting state is not re-equilibrated, transient adjustment before absorption can occur.

All 30 scenario summaries are exported; 4 perturbed values lie outside their original fitting bounds (no clipping). The no-meal equilibrium Jacobian has strictly negative eigenvalue real parts in 30/30 scenarios. This is local equilibrium stability, not proof of global stability or sustained oscillations. The factors are not uncertainty intervals; conclusions concern this reference scenario. Behavior beyond 150 minutes is unvalidated extrapolation.

Exports: [ID-ordered fit data](tables/fit_performance_by_id.csv), [all sweep trajectories](tables/parameter_sweep_trajectories.csv), [scenario metrics and equilibrium eigenvalues](tables/parameter_sweep_summary.csv), [complete configuration](tables/parameter_sweep_config.json), [manuscript captions](manuscript/fit_by_id_and_dynamics.tex).

<!-- END ID_FIT_AND_DYNAMICS -->
