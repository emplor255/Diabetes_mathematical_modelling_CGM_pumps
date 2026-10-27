#!/usr/bin/env python3
"""Conditional beta1 sensitivity fits; diagnostic only, never selects main controls."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from extra_experiment.experiment import BOUNDS, NAMES, ExperimentConfig, grid_for, prepare_record, simulate


def fit_at_beta1(record, reference, beta1, starts, config):
    """Fix beta1, re-fit the other six coefficients under the original robust loss."""
    if not BOUNDS[2, 0] <= beta1 <= BOUNDS[2, 1]:
        raise ValueError("Fixed beta1 must be within the original bounds")
    if not starts:
        raise ValueError("At least one start is required")
    free = np.array([0, 1, 3, 4, 5, 6])
    lo, hi = np.log(BOUNDS[free, 0]), np.log(BOUNDS[free, 1])
    times = record.time_min.to_numpy() / 60.
    observed = record.glucose.to_numpy()

    def expand(log_values):
        p = np.empty(7)
        p[free] = np.exp(log_values)
        p[2] = beta1
        return p

    def residual(log_values):
        return (simulate(record, reference, expand(log_values), times, config)[:, 1] - observed) / 5.

    candidates, logs = [], []
    for number, start in enumerate(starts):
        x0 = np.clip(np.log(np.asarray(start)[free]), lo + 1e-8, hi - 1e-8)
        fit = least_squares(residual, x0, bounds=(lo, hi), loss="soft_l1", f_scale=1.,
                            max_nfev=config.fit_max_nfev, diff_step=1e-4,
                            xtol=1e-7, ftol=1e-7, gtol=1e-7)
        candidates.append(fit)
        logs.append({"fixed_beta1": beta1, "start": number + 1, "cost": float(fit.cost),
                     "success": bool(fit.success), "nfev": int(fit.nfev), "message": str(fit.message),
                     **dict(zip(NAMES, map(float, expand(fit.x))))})
    best = min(candidates, key=lambda fit: fit.cost)
    p = expand(best.x)
    errors = simulate(record, reference, p, times, config)[:, 1] - observed
    dense = grid_for(record)
    n_bins = int(np.ceil(dense[-1] * 60 / config.control_bin_minutes))
    baseline = simulate(record, reference, p, dense, config)[:, 1]
    forced = simulate(record, reference, p, dense, config, np.full(n_bins, config.control_max))[:, 1]
    row = {"fixed_beta1": beta1, "robust_cost": float(best.cost),
           "rmse": float(np.sqrt(np.mean(errors**2))), "mae": float(np.mean(abs(errors))),
           "r2": float(1 - np.sum(errors**2) / np.sum((observed - observed.mean())**2)),
           "optimizer_success": bool(best.success), "nfev": int(best.nfev),
           "max_BG_change_at_u10": float(np.max(abs(forced - baseline))),
           "baseline_peak_BG": float(baseline.max()), "forced_u10_peak_BG": float(forced.max()),
           "forced_u10_min_BG": float(forced.min()), **dict(zip(NAMES, map(float, p)))}
    return p, row, logs


def write_diagnostic(output, profile, main_fit, baseline_fit):
    """Rebuild comparison after the independent multistart fit has completed."""
    comparison = []
    candidates = [("Before (4 local starts, soft-L1)", baseline_fit)]
    robust_path = output / "case02_robust_multistart/case_02_fit.json"
    if robust_path.exists():
        candidates.append(("16 mixed starts, soft-L1", json.loads(robust_path.read_text())))
    config = main_fit["config"]
    candidates.append((f"Main: {config['fit_starts']} {config.get('fit_start_strategy', 'local')} starts, "
                       f"{config.get('fit_loss', 'soft_l1')}", main_fit))
    for name, saved in candidates:
        row = saved["summary"]
        comparison.append({"fit": name, **{key: row[key] for key in ["rmse", "mae", "r2", "beta0", "beta1",
                            "gamma", "delta", "data_log_jacobian_condition"]}})
    pd.DataFrame(comparison).to_csv(output / "case02_before_after.csv", index=False)
    lines = ["# Case 02 fitting retune and conditional sensitivity", "",
             "Retuning first used 16 soft-L1 starts (8 near nominal, 8 log-uniform over the original "
             "bounds), then 8 ordinary least-squares starts (4 near nominal, 4 log-uniform), "
             "up to 500 function evaluations per start. Same 15 observations, model, bounds, fixed "
             "K/reference/absorption assumptions and control settings. Ordinary least squares directly "
             "minimizes in-sample RMSE but is less robust to outliers; robust-loss costs must not be "
             "compared numerically against ordinary-least-squares costs. No insulin "
             "responsiveness constraint or control-performance-based parameter selection.", "",
             "| Fit | RMSE | MAE | R² | beta1 |", "|---|---:|---:|---:|---:|"]
    for row in comparison:
        lines.append(f"| {row['fit']} | {row['rmse']:.5f} | {row['mae']:.5f} | {row['r2']:.5f} | {row['beta1']:.5g} |")
    lines += ["", "The main result is selected for glucose-fitting error, not for a visually stronger "
              "control response. A lower in-sample RMSE is not evidence of better out-of-sample prediction "
              "or identification of physiological parameters. Previous results are retained in "
              "`case02_before/` and `case02_robust_multistart/`."]
    before, after = baseline_fit["summary"], main_fit["summary"]
    improvement = 100 * (before["rmse"] - after["rmse"]) / before["rmse"]
    lines += ["", f"RMSE changes by {improvement:.3f}% relative to the original fit; "
              f"MAE changes from {before['mae']:.5f} to {after['mae']:.5f} mg/dL. "
              "This is a tradeoff between fitting losses, not uniform improvement across metrics."]
    if after["beta1"] < 1.01 * BOUNDS[2, 0]:
        lines += ["The main beta1 remains at its lower bound. Retuning has not resolved the "
                  "weak modeled response to additional control or the parameter-identification problem."]
    lines += ["", "## Conditional beta1 profile (not confidence intervals)", "",
              "At each fixed beta1, the other six parameters are refitted from three starts: "
              "the original fitted vector, the nominal vector, and the preceding grid solution. "
              "The smallest robust-loss cost is retained, not the best control response. This is "
              "a bounded numerical search, not a proof of the global profile minimum.", "",
              "| Fixed beta1 | Robust cost | RMSE | R² | Max BG change under constant u=10 |",
              "|---|---:|---:|---:|---:|"]
    for row in profile.itertuples():
        lines.append(f"| {row.fixed_beta1:.5g} | {row.robust_cost:.5f} | {row.rmse:.5f} | {row.r2:.5f} | "
                     f"{row.max_BG_change_at_u10:.5f} |")
    lines += ["", "The forced u=10 simulations are a sensitivity diagnostic, not optimized controls "
              "or insulin doses. Profile solutions do not replace the main fit. Similar glucose "
              "errors with different control responses indicate that a good in-sample fit alone "
              "does not establish a reliable response to intervention. Conversely, a worse "
              "fit at a selected beta1 is not justification to force that value for a prettier control plot.", "",
              "Only 15 points are available, with no premeal baseline. Reference BG=65 and the "
              "dimensionless insulin scale remain strong assumptions. No clinical insulin sensitivity "
              "or dose recommendation can be inferred from these fits.", ""]
    (output / "case02_tuning_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "extra_experiment/tuning"
    output.mkdir(exist_ok=True)
    results = ROOT / "extra_experiment/results"
    baseline = json.loads((output / "case02_before/case_02_fit.json").read_text())
    source = ROOT / "extra_experiment/data/patient.csv"
    record, audit = prepare_record(pd.read_csv(source))
    cfg = ExperimentConfig(fit_max_nfev=500)
    if not args.summarize_only:
        original = np.asarray(baseline["parameters"])
        nominal = np.array([1.5, .05, .03, 1., 1., 4., .3])
        previous = original.copy()
        rows, logs = [], []
        for beta1 in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1]:
            previous, row, runs = fit_at_beta1(record, audit["reference_glucose"], beta1,
                                              [original, nominal, previous], cfg)
            rows.append(row)
            logs.extend(runs)
            pd.DataFrame(rows).to_csv(output / "case02_beta1_profile.csv", index=False)
            pd.DataFrame(logs).to_csv(output / "case02_beta1_profile_multistart.csv", index=False)
            print(f"beta1={beta1:g}: cost={row['robust_cost']:.4f}, RMSE={row['rmse']:.4f}, "
                  f"max u=10 BG change={row['max_BG_change_at_u10']:.4f}", flush=True)
        (output / "case02_profile_audit.json").write_text(json.dumps({
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "reference_glucose": audit["reference_glucose"], "config": asdict(cfg),
            "beta1_grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1], "starts_per_point": 3,
            "free_parameters_per_point": 6, "loss": "soft_l1", "residual_scale": 5.,
            "interpretation": "Conditional sensitivity only; not confidence intervals or clinical doses",
        }, indent=2))
    profile = pd.read_csv(output / "case02_beta1_profile.csv")
    main_fit = json.loads((results / "case_02_fit.json").read_text())
    write_diagnostic(output, profile, main_fit, baseline)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].semilogx(profile.fixed_beta1, profile.rmse, "o-", color="#2471A3")
    axes[0].axhline(main_fit["summary"]["rmse"], color="gray", ls="--",
                   label=f"Main fit ({main_fit['config'].get('fit_loss', 'soft_l1')})")
    axes[0].set(xlabel="Fixed beta1 (other parameters refitted)", ylabel="In-sample RMSE (mg/dL)")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(profile.fixed_beta1, profile.max_BG_change_at_u10, "o-", color="#C0392B")
    axes[1].set(xlabel="Fixed beta1 (other parameters refitted)", ylabel="Max |BG(u=10) − BG(u=0)| (mg/dL)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Case 02: fixed-beta1 robust refits and control sensitivity (not clinical dosing)", fontsize=11)
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"case02_beta1_profile.{ext}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
