# CGM mathematical-model paper: reproducible analysis

This repository implements the data analysis, parameter estimation, model ablation,
and optimal-control simulation described in `Blank.pdf` using:

- `hall2018_full_another_version.xlsx`: full CGM series and glycaemic-status labels;
- `pbio.2005143.s015.xlsx`: standardized meal windows from 30 minutes before to
  150 minutes after the meal.

The outputs are research simulations. They are **not insulin-dose recommendations**.
Neither workbook contains measured insulin concentration, insulin-pump delivery, or
administered insulin dose, so the control variable is dimensionless and cannot be
converted to insulin units.

## Local inputs (not included in Git)

The Excel workbooks and `Blank.pdf` are intentionally excluded from this repository.
Place your authorized copies of the two workbooks listed above in the project root
before running the analysis. Generated `outputs/` and `outputs_*` directories are
versioned at the project owner's request, including result spreadsheets and
manuscript artifacts. Recreate results with the commands below.

## Additional individual-record experiments

`extra_experiment/` contains a separate recorded-input fitting and research-control
experiment for the two CSV records from `emplor255/Diabetes-Control`: the first-meal
phase for case_01 and the full record for case_02. It preserves
recorded meal and bolus events, and does not alter the Hall2018 results or core
model. See `extra_experiment/README.md` for the explicit model extension, input
assumptions, limitations, and reproduction commands.

## Run

Python 3.9 or newer is required.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python run_analysis.py --workers 4
```

The default full run fits all 30 participants with usable standardized meals. A fast
smoke run is:

```bash
.venv/bin/python run_analysis.py \
  --max-participants 3 \
  --stage1-iterations 3 \
  --stage2-max-nfev 40 \
  --skip-control \
  --output outputs_smoke
```

Useful controls:

```text
--seed INT                 deterministic optimizer seed
--workers INT              participant-level process parallelism
--min-event-points INT     minimum unique points in a complete meal window
--stage1-iterations INT    differential-evolution generations
--stage2-max-nfev INT      robust least-squares evaluation limit
--skip-control             skip the unitless control experiment
```

## Outputs

`outputs/analysis_report.md` is the concise result summary. Detailed artifacts are:

- `outputs/tables/data_audit.json`: schema/QC counts and complete run configuration;
- `outputs/tables/meal_event_qc.csv`: inclusion and exclusion reason for every meal;
- `outputs/tables/participant_parameters.csv`: patient-level kinetic estimates and
  identifiability diagnostics;
- `outputs/tables/event_fit_metrics.csv`: per-meal RMSE/MAE/R² and nuisance estimates;
- `outputs/tables/model_comparison.csv`: proposed vs reduced-model RMSE, AIC, and BIC;
- `outputs/tables/control_summary.csv`: research-only control comparison;
- `outputs/tables/control_trajectories.csv`: complete simulated trajectories;
- `outputs/tables/control_evaluation.xlsx`: overall and subgroup control comparison
  (mean/SD, paired changes with bootstrap 95% CIs, and counts improved/unchanged/worsened);
- `outputs/tables/control_cohort_statistics.csv`: numeric statistics, including median/IQR;
- `outputs/tables/control_paired_changes.csv`: matched changes for every fitted participant;
- `outputs/tables/control_cohort_diagnostics.csv`: below-target and control-saturation counts;
- `outputs/tables/control_evaluation.tex`: the overall comparison table for a manuscript;
- `outputs/figures/figure_*.png`: cohort, meal, fit, parameter, and control figures.

The paired control table is also included directly in `outputs/analysis_report.md`
and generated automatically by the main pipeline. To add or refresh it from the
saved results without rerunning model fitting or control optimization:

```bash
.venv/bin/python summarize_control.py --output outputs
```

This table covers every fitted participant (30 in the full run: 20 healthy,
7 prediabetes, 3 type 2 diabetes), with **one simulated median-meal scenario per
person**. The 147 usable meals inform parameter estimation; they are not 147 separate
control tests. The 57-person Hall2018 cohort is not the control-analysis denominator.
Each person has equal weight; changes are optimized minus no control, and differences
of percentages are percentage points. The bootstrap resamples paired participants
10,000 times with the saved seed. It is conditional on the fitted models and does not
propagate parameter uncertainty. Small-subgroup intervals are descriptive.

The table includes time below 70 mg/dL and individual worsening as well as target
time and peak reduction. The paper objective is an optimization target and can be
negative; a relative percentage improvement in it would be misleading. Lower objective
alone does not establish better target time or freedom from below-target glucose.
Only the no-external-control comparator is evaluated; no clinical or held-out
performance claim follows from these simulations.

## Dataset section and overview

The English manuscript section is in `outputs/manuscript/dataset_section.tex`, with
references in `outputs/manuscript/dataset_references.bib`. It includes the generated
cohort table and `figure_6_dataset_overview.pdf` (a vector figure; PNG is also available).
LaTeX paths assume compilation from the repository root; adapt them if transferring
the section to another project. Required packages are `booktabs`, `graphicx`, and `url`.
Merge the two citation entries into the manuscript's bibliography. An optional,
more explicit caption for the unchanged Figure 2 is in
`outputs/manuscript/figure_2_caption.tex`.

Recreate the descriptive artifacts directly from the workbooks, without refitting:

```bash
.venv/bin/python describe_dataset.py --output outputs
```

The main pipeline also generates these artifacts before selecting any fitting subset.
`dataset_overview.csv` contains full-cohort and nested-meal counts and ages;
`dataset_meal_availability.csv` distinguishes absent, excluded, and retained slots;
`meal_profile_sample_sizes.csv` gives the actual pointwise denominators of Figure 2.
The manuscript prose describes the supplied files under the default 24-point QC
threshold; it is not automatically rewritten when the data or QC options change.

The `.s015` file identifier refers to **S6 Data**, not S15 Data, in
[Hall et al. (2018)](https://doi.org/10.1371/journal.pbio.2005143).
Figure 2 is descriptive: replicate readings are averaged within participant/time,
then across participants. Its shading is a pointwise bootstrap CI for the group
mean, not individual-response variability. Absolute glucose curves include baseline
differences; the type 2 diabetes meal subgroup has only three people. No efficacy
or independent validation claim follows from this plot. Its original seaborn
bootstrap was not explicitly seeded, so bands can vary slightly on a full rerun;
the existing image has not been changed by the dataset-description script.

## Parameter-estimation figures and tables

Regenerate presentation artifacts from saved predictions, without refitting or
changing control results:

```bash
.venv/bin/python summarize_fits.py --output outputs
```

The main pipeline generates these artifacts automatically. Outputs include:

- `outputs/figures/figure_3_selected_model_fits.png` and `.pdf`: one highest-event-R²
  example per status group, with every fitted observation, both model curves, and
  proposed-model residuals. This is explicitly a favorable selection, not typical
  cohort performance. The original Figure 3 is preserved.
- `outputs/figures/figure_7_fit_diagnostics.png` and `.pdf`: paired RMSE for all
  participants and cumulative RMSE distributions for all meal events.
- `outputs/tables/fit_overall.tex`: pooled and participant-level fit metrics for
  proposed and reduced models, including negative event R² and nominal AIC/BIC wins.
- `outputs/tables/parameter_group_comparison.tex`: medians [Q1, Q3] for Healthy,
  prediabetes, and T2D separately; the final column is T2D minus Healthy medians.
- `outputs/tables/parameter_bound_diagnostics.tex`: lower/upper boundary counts
  under the original 0.5%-of-search-range rule, to accompany the parameter table.
- `outputs/tables/parameter_estimation_evaluation.xlsx`: formatted tables, numeric
  overall/subgroup statistics, individual participants/events, contrasts, selection
  records, and interpretation notes. Numeric CSV versions are also exported.
- `outputs/manuscript/parameter_estimation_results.tex`: optional manuscript text,
  table inputs, and figure captions for the current full run. It requires `booktabs`
  and `graphicx`; paths assume compilation from the repository root. Numerical prose
  and selected-event identifiers must be rechecked if the fits change.

All scores are **in-sample**, recomputed from saved predictions with no fit-quality
exclusions. Pooled errors weight observations equally; participant summaries weight
people equally. Within-event R² uses event-centered observed sums of squares so that
between-event baseline differences do not inflate the denominator. Negative R² is
retained; constant-glucose events have undefined R² and their count is exported.
Because the meal-time value informs initialization, positive-time-only errors are
also exported numerically. The optimizer used robust, weighted residuals, whereas
these descriptive metrics are unweighted. Nominal AIC/BIC do not correct for serially
correlated CGM errors. Group comparisons are descriptive, without p-values: they do
not account for parameter uncertainty or confounding, and T2D has only three people.
Do not treat boundary-concentrated estimates as well-identified physiological markers.

## ID-ordered fitting performance and dynamical parameter sweeps

Create the two additional figures from saved fits (no refitting):

```bash
.venv/bin/python explore_dynamics.py --output outputs
```

The main analysis pipeline also generates them automatically:

- `outputs/figures/figure_8_fit_by_id.png` / `.pdf`: RMSE of each fitted participant
  in ascending natural-ID order. All meals/observations for each person are included.
  IDs are equally spaced categories, not numeric predictors. The dashed line is the
  median participant RMSE, not pooled RMSE or the robust fitting objective.
- `outputs/figures/figure_9_parameter_sweep.png` / `.pdf`: glucose trajectories in
  six panels for alpha, beta0, beta1, beta2, gamma, and delta. Each coefficient is
  varied over 0.5, 0.75, 1, 1.25, and 1.5 times its nominal value, keeping all other
  coefficients fixed. Initial glucose, initial quick insulin, meal amplitude, delay,
  K, and basal insulin are identical across scenarios; external control is zero.
- `outputs/tables/fit_performance_by_id.csv`, `parameter_sweep_trajectories.csv`,
  `parameter_sweep_summary.csv`, and `parameter_sweep_config.json`: plot data,
  all three simulated states, scalar summaries, exact no-meal equilibrium Jacobian
  eigenvalues, and complete scenario configuration.
- `outputs/manuscript/fit_by_id_and_dynamics.tex`: generated captions and actual
  reference coefficients/settings. The existing analysis report is also updated.

The default reference is an actual fitted participant nearest the cohort median
RMSE (ties resolved by ascending natural ID), **not** a synthetic vector of marginal
parameter medians. In the current full run this is `2133-020`. The scenario uses this
person's reference glucose and median fitted meal amplitude/delay; it is not a
reconstruction of one specific observed meal. The default horizon is 12 hours with
1-minute output spacing. Customize the reference, horizon, or multipliers with:

```bash
.venv/bin/python explore_dynamics.py --output outputs \
  --reference-id 2133-018 --horizon-hours 6 --factors 0.5 0.75 1 1.25 1.5
```

This command replaces only the new exploration artifacts with that scenario; it
does not refit anyone or change control results. Rerun the default command to restore
the default scenario. It loads the original ModelConfig from `data_audit.json`.

Important: beta2 is derived **once** at the nominal reference equilibrium. It is then
fixed in the other five panels and varied independently in its own panel. This is
strict one-coefficient-at-a-time analysis of the ODE, not a sweep constrained to the
fitting-stage equilibrium manifold. The initial insulin state is also frozen, so a
perturbed trajectory may start away from its new equilibrium and move before meal
absorption. Keyword-only simulation overrides implement this experiment; default
fitting and control behavior are unchanged. Some factors cross optimization bounds;
they are explicitly flagged rather than clipped. The beta2 panel has no independently
fitted bounds. No confidence intervals or physiological identifiability are implied.

The fitting windows end at 150 minutes; later simulated behavior is **unvalidated
extrapolation**. The reference beta0 is very small, so its multiplier curves nearly
overlap; this is retained rather than visually exaggerated. Complex equilibrium
eigenvalues allow damped modes, not necessarily sustained oscillations. Negative real
parts establish local stability of that no-meal equilibrium only, not global behavior.

## How the paper was made executable

The draft has no empirical result tables and contains unresolved notation. The code
makes the smallest explicit choices needed for a reproducible run:

1. The printed glucose equation contains `beta0`, but the parameter list omits it;
   the code fits `beta0` as insulin-independent clearance.
2. `K` and basal insulin are not specified and cannot be inferred from CGM alone;
   they are fixed at 120 mg/dL and 1 dimensionless unit.
3. The draft writes an ODE for total insulin even after decomposing it into basal and
   quick components. The code evolves only the quick component and keeps basal fixed.
4. Free `beta2` is practically non-identifiable from CGM alone. The code derives it
   by enforcing a no-meal equilibrium at each participant's median pre-meal glucose.
5. The paper requires a digestive delay although Equation (1) has no delay term. A
   bounded event-level absorption delay is fitted as a nuisance parameter.
6. Only post-meal observations are used in the dynamical loss. The preceding 30
   minutes define the participant baseline and the glucose value at meal time.
7. The cited Bergman/Bertuzzi equations are not given in the draft, and the available
   files lack their required insulin measurements. The executable ablation is therefore
   labeled honestly as a reduced two-state model rather than as either cited model.

Stage 1 smooths the participant's median response with a cubic spline and uses global
differential evolution. Stage 2 uses the resulting point to initialize robust bounded
nonlinear least squares on every usable meal event. Participant kinetic parameters are
shared across events; meal amplitude and delay are event-specific nuisance parameters.

## Data QC choices

- Duplicate participant/meal/timestamp rows are averaged and down-weighted to 0.8.
- The six literal `Low` readings are approximated by 40 mg/dL (not exact measured
  concentrations) and down-weighted to 0.8.
- A meal is fitted only when its timestamp span is exactly 180 minutes (within a small
  numerical tolerance) and it has at least 24 unique CGM points. This keeps meal time
  unambiguous: window start plus 30 minutes.
- Every excluded window and reason remains in `meal_event_qc.csv`.
- Full timestamp span does not imply uninterrupted data: internal missing points
  remain in some retained events (largest observed gap: 55 minutes).

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The primary scientific limitation is structural identifiability: only glucose is
observed, while digestive glucose and insulin are latent. Check
`jacobian_condition_number` and `kinetic_bound_hits` before interpreting any fitted
parameter. Group tests are exploratory and should not be treated as causal or clinical
evidence.
