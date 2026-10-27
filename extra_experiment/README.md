# Two additional recorded-input experiments

These exploratory experiments use `Mai_Hue_24_08.csv` (`case_01`) and `patient.csv`
(`case_02`) from [emplor255/Diabetes-Control](https://github.com/emplor255/Diabetes-Control),
pinned to commit `28a82c1bdad2635396cab9c6776f71336a141ae9`.
Figure/table labels use case IDs, but the source files/provenance retain the original
filename: **this folder is not de-identified**. Do not publish it outside the
owner-authorized destination without reviewing privacy and redistribution rights.

## Reproduce

From the project root, using the existing environment:

```bash
.venv/bin/python extra_experiment/run_experiment.py --download
.venv/bin/python extra_experiment/run_experiment.py --cases case_02 --fit-starts 8 --fit-max-nfev 500 --fit-start-strategy mixed --fit-loss linear
.venv/bin/python -m unittest discover -s tests -v
```

`--download` fetches only missing files at the pinned revision; it never overwrites
local inputs. Every input's SHA256 is saved in `results/data_audit.json`. Subsequent
runs can omit `--download`. `--reuse-fits` skips fitting only when the saved input
hash and configuration match exactly; it reruns the control and export stages.
The selected analysis window must also match the cached fit. The second command
above applies the documented case_02 RMSE-oriented retune after the default run.

## Analysis windows

By default **case_01 uses only the first meal phase**: from 30 minutes before to
150 minutes after the first positive `Eating` timestamp, stopping before the next
meal if it occurs earlier. The first meal is at **06:58**. The available observations
in this window span **06:29–09:24** (30 source rows, 29 unique times), including one
meal and three boluses. The actual endpoints are -29 and +146 minutes relative to
the meal; no observations at the nominal endpoints are manufactured. The existing
40-minute gap is retained, not filled. Window selection uses meal times only,
not glucose values or fit scores. The original input CSV is unchanged.

**case_02 still uses its full 07:30–13:15 record.** To update case_01 without
refitting or rewriting the saved case_02 artifacts (requires existing summaries):

```bash
.venv/bin/python extra_experiment/run_experiment.py --cases case_01
```

The combined summary tables/report are updated to include the new case_01 and
the unchanged case_02. To reproduce the earlier full-day case_01 experiment in a
separate destination:

```bash
.venv/bin/python extra_experiment/run_experiment.py --case01-window full_record --output extra_experiment/results_full_record
```

The original full-record results are also preserved in Git commit `72236b2`.
Scores on a short selected phase are not scores for the entire day and cannot be
compared directly with whole-day scores as evidence of method improvement.

## What is fitted?

One continuous trajectory and **one seven-parameter vector per case**, using all
distinct recorded times **inside the selected window**, rather than treating each
meal as an independent reset:

- Paper coefficients: alpha, beta0, beta1, gamma, delta (original fitting bounds).
- Effective meal gain (0.01–30) and bolus gain (0.0001–30), in native input units.
- K=120 mg/dL, dimensionless basal insulin=1, and bolus absorption k_abs=1/h fixed.
- beta2 derived from the assumed no-input reference equilibrium, as in the main pipeline.

The original three-state model is extended by one bolus absorption depot S:

```text
D' = -alpha D
G' = alpha D - (beta0 + beta1 (1 + Q)) G + beta2 / (2 + Q)
Q' = gamma G²/(K² + G²) - delta Q + k_abs S + u
S' = -k_abs S
At meal time:  D <- D + meal_gain * Eating
At bolus time: S <- S + bolus_gain * Insulin Bolus
```

This is an explicit exploratory extension, not a validated pharmacokinetic model
and not a reproduction of the older notebooks in the source repository. Their
glucose labels and Noise-weight comments are not sufficient to establish units or
reliability weights. BG is provisionally treated as mg/dL; Eating and Bolus remain
native recorded amounts. This assumption must be confirmed before manuscript use.

Each record starts at its first measured BG; digestive and bolus depots start at
zero. Reference glucose is the median BG within 30 minutes before the first meal,
or the first BG if none exists. Initial Q is its reference equilibrium. No basal
dose, previous insulin-on-board, or earlier meals are known. In particular,
`patient.csv` starts at BG=65 with an eating/bolus event and lacks a premeal baseline.

The fit uses multistart bounded nonlinear least squares in log parameters with
soft-L1 loss by default and residual scale 5 mg/dL. `--fit-loss linear` instead
minimizes ordinary squared residuals (and hence RMSE); the chosen loss is recorded
per case in the audit. There is no parameter averaging across
meals, no fit-quality-based exclusion within a selected window, and no imputed observations. At duplicate
times BG is averaged and input amounts are summed; Noise labels are retained with
equal observation weights. Fit diagnostics include all multistart outcomes,
bound proximity, and the data-only finite-difference Jacobian condition number.

## Control experiment

Compare **recorded meals + recorded boluses** with **the same recorded inputs +
additional optimized dimensionless control**. Both start from the same state and
use the same fitted model. Control is nonnegative, bounded by 10, piecewise constant
in 60-minute bins (last bin truncated), and enters Q' directly. It cannot be
interpreted as an insulin delivery rate in U/h or converted into a patient dose.

The original paper's time-weighted glucose cost targeting 70–110 plus squared
control cost is retained. No hard hypoglycaemia constraint is added: minimum
glucose and time below 70 are explicitly reported, even when they worsen. The
optimizer always competes against zero additional control. All future logged
inputs are known: this is offline open-loop optimization, not a real-time controller.
Glucose integrals and time fractions account for duration; control integrals use
exact bin durations. Event discontinuities are integrated exactly at event times,
without clipping states or smearing bolus amounts into interpolated dose rates.

## Outputs and limits

`results/` contains input audits, prepared data, parameters, multistart and boundary
diagnostics, fit predictions, control trajectories/bins, paired changes, numerical
accuracy checks, CSV/Excel/LaTeX summary tables, and `report.md`. Each case has a
fitting figure and a control figure in PNG and vector PDF under `results/figures/`.

All fit metrics are in-sample. `patient.csv` has only 15 observations for seven
fitted parameters. Optimizer convergence does not establish parameter uniqueness.
There is no held-out validation, clinical efficacy inference, uncertainty
propagation, or significance test between two cases. The fitted gains and assumed
bolus absorption do not calibrate the dimensionless control to therapeutic insulin.
The main Hall2018 code/results are unchanged by this experiment.

## Case 02 retuning and identifiability check

The case_02 retune retains all 15 observations and the original parameter bounds,
K, reference BG, bolus absorption, and control settings. First, 16 mixed local and
log-uniform starts with soft-L1 loss check for missed minima. Then 8 mixed starts
with ordinary least squares target RMSE directly. `--fit-max-nfev 500` allows more
iterations. The lowest loss within each trial is selected; control responsiveness
is never used to rank fitted parameter vectors. Ordinary least squares is less
robust to outliers; in-sample improvement is not predictive validation.

```bash
# Retuned RMSE-oriented fit/control for case_02; case_01 artifacts are untouched.
.venv/bin/python extra_experiment/run_experiment.py --cases case_02 --fit-starts 8 --fit-max-nfev 500 --fit-start-strategy mixed --fit-loss linear
# Conditional robustness diagnostic (separate from the main fitting artifacts):
.venv/bin/python extra_experiment/profile_case02.py
```

`tuning/case02_before/` preserves the original 4-start results and
`tuning/case02_robust_multistart/` preserves the 16-start robust results. The tuning
report and before/after table are in `tuning/`. The beta1 profile fixes beta1 at
six values, refits the other six parameters with soft-L1 loss from three starts,
and compares model responses to **constant dimensionless u=10**. This is neither
a confidence interval nor an optimized or clinically calibrated dosing experiment.
Profile fits never replace the main fit. `--summarize-only` regenerates the tuning
report/figure using saved profile results and the current main fit.

## Case 01 control-figure presentation

At the owner's request, the case_01 control figure displays a **70–130 mg/dL**
target band and labels the comparator **Conventionally calculated boluses** (and
the second curve **Optimized control**).
These are presentation-only changes: the input boluses remain the logged CSV
values; no traditional bolus-calculation formula is supplied or verified here.
The optimization objective and exported time-in-range metrics remain **70–110**,
as explicitly noted on the figure. The red dashed 70 line remains a separate
reference. Case_02 labels/range and all simulated trajectories are unchanged.
