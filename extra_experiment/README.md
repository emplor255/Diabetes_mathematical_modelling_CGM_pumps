# Two additional full-record experiments

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
.venv/bin/python -m unittest discover -s tests -v
```

`--download` fetches only missing files at the pinned revision; it never overwrites
local inputs. Every input's SHA256 is saved in `results/data_audit.json`. Subsequent
runs can omit `--download`. `--reuse-fits` skips fitting only when the saved input
hash and configuration match exactly; it reruns the control and export stages.

## What is fitted?

One continuous trajectory and **one seven-parameter vector per case**, using all
distinct recorded times, rather than treating each meal as an independent reset:

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
soft-L1 loss and residual scale 5 mg/dL. There is no parameter averaging across
meals, no discarded poor-fit segments, and no imputed observations. At duplicate
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
