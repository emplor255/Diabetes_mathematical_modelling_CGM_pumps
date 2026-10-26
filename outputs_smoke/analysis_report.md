# Reproduction report

## Data used

- Hall2018: 105,417 CGM rows from 57 participants (38 healthy, 14 prediabetes, 5 type 2 diabetes).
- Standardized meals: 6,152 input rows, 176 event windows, and 147 complete usable windows.
- QC: 87 duplicate timestamp rows collapsed by their mean; 6 `Low` readings represented at the device floor (40 mg/dL) with weight 0.8.

## Parameter estimation and ablation

- Fitted participants: 3.
- Proposed-model median participant RMSE: 13.99 mg/dL.
- Reduced two-state median participant RMSE: 15.32 mg/dL.
- The proposed model had lower BIC for 66.7% of participants.
- 3/3 fits had at least one kinetic parameter at a bound.
- 0/3 fits had normalized kinetic-Jacobian condition number >1,000 or infinite.

The last two diagnostics are essential: CGM alone does not identify hidden insulin dynamics reliably. Group-level parameter differences in the CSV tables are exploratory, not mechanistic proof.

## Optimal control

Optimal-control simulation was skipped.

## Reproducibility notes

- Meal time is defined as 30 minutes after the beginning of a complete 180-minute S15 window.
- Only complete windows with at least the configured minimum number of unique CGM points are fitted.
- beta2 is derived from the participant's pre-meal equilibrium; K=120 mg/dL and basal insulin=1 are fixed because insulin is unobserved.
- The printed model's required absorption delay is implemented as an event-level fitted nuisance parameter.
- Random seeds, bounds, optimizer settings, event QC, predictions, and control trajectories are exported under `tables/`.
