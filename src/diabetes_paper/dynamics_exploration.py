"""ID-ordered fitting errors and controlled one-at-a-time parameter experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .data import STATUS_ORDER
from .estimation import KINETIC_BOUNDS
from .fit_evaluation import COLORS, evaluate_fits
from .model import (
    ModelConfig, PARAMETER_NAMES, derive_beta2, proposed_rhs,
    quick_insulin_equilibrium, simulate_meal,
)


SWEEP_PARAMETERS = ("alpha", "beta0", "beta1", "beta2", "gamma", "delta")
SYMBOLS = {"alpha": r"$\alpha$", "beta0": r"$\beta_0$", "beta1": r"$\beta_1$",
           "beta2": r"$\beta_2$", "gamma": r"$\gamma$", "delta": r"$\delta$"}
UNITS = {"alpha": r"h$^{-1}$", "beta0": r"h$^{-1}$", "beta1": r"(model I · h)$^{-1}$",
         "beta2": "mg/dL/h", "gamma": "model I/h", "delta": r"h$^{-1}$"}


@dataclass(frozen=True)
class SweepConfig:
    factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
    horizon_hours: float = 12.0
    step_minutes: float = 1.0


def _natural_id_key(value: str) -> tuple:
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", str(value)))


def fit_performance_by_id(participant_statistics: pd.DataFrame) -> pd.DataFrame:
    proposed = participant_statistics.loc[participant_statistics.model.eq("Proposed 3-state")].copy()
    reduced = participant_statistics.loc[participant_statistics.model.eq("Reduced 2-state"), ["userID", "rmse"]]
    proposed = proposed[["userID", "status", "n_events", "n_points", "rmse", "mae"]].rename(
        columns={"rmse": "rmse_proposed", "mae": "mae_proposed", "n_points": "n_observations"}
    )
    result = proposed.merge(reduced.rename(columns={"rmse": "rmse_reduced"}),
                            on="userID", how="left", validate="one_to_one")
    if result.empty or result.rmse_reduced.isna().any():
        raise ValueError("Every participant needs both saved model errors")
    order = sorted(result.userID, key=_natural_id_key)
    result = result.set_index("userID").loc[order].reset_index()
    result.insert(0, "id_order", np.arange(1, len(result) + 1))
    return result


def select_reference(participants: pd.DataFrame, reference_id: str | None = None) -> pd.Series:
    if participants.empty or participants.userID.duplicated().any():
        raise ValueError("A nonempty table with unique participant IDs is required")
    if reference_id is not None:
        chosen = participants.loc[participants.userID.eq(reference_id)]
        if chosen.empty:
            raise ValueError(f"Reference ID {reference_id!r} is not in the fitted cohort")
        return chosen.iloc[0].copy()
    if not np.isfinite(participants.rmse_proposed.to_numpy(float)).all():
        raise ValueError("Reference selection requires finite RMSE values")
    distance = (participants.rmse_proposed - participants.rmse_proposed.median()).abs()
    # For an even cohort the two central RMSEs can be equally near the median.
    # Treat numerical round-off as a tie and use ascending natural ID.
    tied = participants.loc[np.isclose(distance, distance.min(), rtol=0, atol=1e-12)]
    selected_id = min(tied.userID, key=_natural_id_key)
    return tied.loc[tied.userID.eq(selected_id)].iloc[0].copy()


def no_meal_equilibrium(parameters: np.ndarray, beta2: float, model: ModelConfig) -> dict:
    """Positive no-input equilibrium and exact Jacobian of this implemented ODE."""
    alpha, beta0, beta1, gamma, delta = np.asarray(parameters, dtype=float)

    def balance(glucose: float) -> float:
        quick = quick_insulin_equilibrium(glucose, parameters, model)
        total = model.basal_insulin + quick
        return beta2 / (1 + total) - (beta0 + beta1 * total) * glucose

    upper = 100.0
    for _ in range(80):
        if balance(upper) < 0:
            break
        upper *= 2
    else:
        raise RuntimeError("Could not bracket the positive equilibrium")
    glucose = float(brentq(balance, 0.0, upper, xtol=1e-11))
    quick = quick_insulin_equilibrium(glucose, parameters, model)
    total = model.basal_insulin + quick
    saturation_slope = 2 * glucose * model.k_half_mg_dl**2 / (model.k_half_mg_dl**2 + glucose**2)**2
    jacobian = np.array([
        [-alpha, 0, 0],
        [alpha, -beta0 - beta1 * total, -beta1 * glucose - beta2 / (1 + total)**2],
        [0, gamma * saturation_slope, -delta],
    ])
    eigenvalues = np.linalg.eigvals(jacobian)
    max_real = float(eigenvalues.real.max())
    imaginary = float(np.abs(eigenvalues.imag).max())
    residual = proposed_rhs(0, np.array([0.0, glucose, quick]), parameters, beta2, model)
    result = {
        "equilibrium_glucose_mg_dl": glucose,
        "equilibrium_quick_insulin": quick,
        "equilibrium_rhs_max_abs": float(np.abs(residual).max()),
        "max_eigenvalue_real_per_hour": max_real,
        "max_eigenvalue_imag_abs_per_hour": imaginary,
        "locally_asymptotically_stable": bool(max_real < -1e-9),
        "complex_eigenvalue_pair": bool(imaginary > 1e-9),
    }
    for index, value in enumerate(sorted(eigenvalues, key=lambda item: (item.real, item.imag)), start=1):
        result[f"eigenvalue_{index}_real_per_hour"] = float(value.real)
        result[f"eigenvalue_{index}_imag_per_hour"] = float(value.imag)
    return result


def run_parameter_sweep(
    participants: pd.DataFrame,
    reference_id: str | None = None,
    sweep: SweepConfig | None = None,
    model: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = sweep or SweepConfig()
    model = model or ModelConfig()
    factors = np.asarray(cfg.factors, dtype=float)
    if factors.ndim != 1 or len(factors) < 2 or not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("Supply at least two finite positive parameter factors")
    if len(np.unique(factors)) != len(factors) or not np.any(factors == 1.0):
        raise ValueError("Factors must be unique and include the exact nominal factor 1.0")
    if not np.isfinite([cfg.horizon_hours, cfg.step_minutes]).all() or cfg.horizon_hours <= 0 or cfg.step_minutes <= 0:
        raise ValueError("Simulation horizon and output step must be positive and finite")
    if model.k_half_mg_dl <= 0 or model.basal_insulin < 0 or not np.isfinite([model.k_half_mg_dl, model.basal_insulin]).all():
        raise ValueError("Model constants must be finite with K > 0 and basal insulin >= 0")
    reference = select_reference(participants, reference_id)
    baseline = reference[list(PARAMETER_NAMES)].to_numpy(dtype=float)
    glucose0 = float(reference.reference_glucose)
    amplitude = float(reference.median_meal_amplitude)
    delay_minutes = float(reference.median_absorption_delay_min)
    if not np.isfinite(np.r_[baseline, glucose0, amplitude, delay_minutes]).all() or np.any(baseline <= 0) or glucose0 <= 0 or amplitude < 0 or delay_minutes < 0:
        raise ValueError("Invalid nominal coefficients or meal scenario")
    if cfg.horizon_hours * 60 <= delay_minutes:
        raise ValueError("Simulation horizon must extend beyond the absorption delay")
    beta2_nominal = derive_beta2(baseline, glucose0, model)
    if "beta2_derived" in reference and not np.isclose(float(reference.beta2_derived), beta2_nominal, rtol=1e-8, atol=1e-8):
        raise ValueError("Saved beta2 and model configuration disagree; use the original run's ModelConfig")
    quick0 = quick_insulin_equilibrium(glucose0, baseline, model)
    nominal = dict(zip(PARAMETER_NAMES, map(float, baseline)))
    nominal["beta2"] = float(beta2_nominal)
    bounds = dict(zip(PARAMETER_NAMES, KINETIC_BOUNDS))
    # Exact endpoint, plus the absorption instant, are always included for audit.
    times = np.unique(np.r_[np.arange(0, cfg.horizon_hours * 60, cfg.step_minutes),
                             delay_minutes, cfg.horizon_hours * 60]) / 60
    frames, summaries = [], []
    for parameter in SWEEP_PARAMETERS:
        for factor in sorted(factors):
            current = nominal.copy()
            current[parameter] = nominal[parameter] * float(factor)
            coefficients = np.array([current[name] for name in PARAMETER_NAMES])
            states = simulate_meal(
                times, coefficients, amplitude, delay_minutes / 60, glucose0, glucose0, model,
                beta2_override=current["beta2"], initial_quick_insulin_override=quick0,
            )
            if not np.isfinite(states).all() or np.any(states < -1e-7):
                raise RuntimeError(f"Nonfinite or negative state in sweep {parameter} x {factor}")
            outside = parameter in bounds and not (bounds[parameter][0] <= current[parameter] <= bounds[parameter][1])
            glucose = states[:, 1]
            prefix = {"reference_id": str(reference.userID), "varied_parameter": parameter,
                      "factor": float(factor), "parameter_value": current[parameter],
                      "outside_original_fit_bounds": bool(outside)}
            frames.append(pd.DataFrame({**prefix, "time_min": times * 60,
                                        "digestive_glucose": states[:, 0], "glucose": glucose,
                                        "quick_insulin": states[:, 2]}))
            summaries.append({
                **prefix, **{f"value_{name}": current[name] for name in SWEEP_PARAMETERS},
                "initial_glucose": glucose0, "initial_quick_insulin": quick0,
                "meal_amplitude": amplitude, "absorption_delay_min": delay_minutes,
                "peak_glucose_mg_dl": float(glucose.max()),
                "time_to_peak_min": float(times[int(np.argmax(glucose))] * 60),
                "minimum_glucose_mg_dl": float(glucose.min()), "end_glucose_mg_dl": float(glucose[-1]),
                "glucose_change_from_initial_at_end": float(glucose[-1] - glucose0),
                **no_meal_equilibrium(coefficients, current["beta2"], model),
            })
    metadata = {
        "reference_id": str(reference.userID), "reference_status": str(reference.status),
        "reference_selection": "Explicit user-selected ID" if reference_id else "Nearest participant RMSE to cohort median; ties by ascending natural ID",
        "reference_participant_rmse_mg_dl": float(reference.rmse_proposed),
        "nominal_coefficients": nominal, "model_config": asdict(model), "sweep_config": asdict(cfg),
        "initial_state_before_absorption": [0.0, glucose0, quick0],
        "meal_amplitude": amplitude, "absorption_delay_min": delay_minutes,
        "control_intensity": 0.0,
        "beta2_policy": "Derived once at the nominal state, then held fixed except in the beta2 panel; not recomputed for other perturbations",
        "initial_state_policy": "Same full initial state for every scenario; not re-equilibrated after changing a coefficient",
        "interpretation": "Deterministic one-at-a-time model exploration, not uncertainty intervals, a global sensitivity study, or an intervention trial",
        "bounds_policy": "No clipping to fitting bounds; outside-bound cases flagged. beta2 has no independently fitted bounds",
        "time_scope": "Fitting windows extend to 150 min; simulations beyond 150 min are unvalidated extrapolations",
        "stability_scope": "Jacobian eigenvalues refer to the positive no-meal equilibrium; complex eigenvalues do not prove sustained oscillations",
        "solver": {"rtol": 3e-5, "atol": 1e-7, "max_step_minutes": 2.5},
    }
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries), metadata


def plot_fit_by_id(table: pd.DataFrame, destination: Path) -> None:
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axis = plt.subplots(figsize=(13.5, 5.4))
        x = np.arange(len(table))
        markers = dict(zip(STATUS_ORDER, ["o", "^", "s"]))
        for status in STATUS_ORDER:
            mask = table.status.eq(status)
            if mask.any():
                axis.scatter(x[mask], table.loc[mask, "rmse_proposed"], color=COLORS[status],
                             marker=markers[status], s=57, label=f"{status} (n={int(mask.sum())})", zorder=3)
        for xx, row in zip(x, table.itertuples()):
            axis.annotate(f"{row.rmse_proposed:.2f}", (xx, row.rmse_proposed), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=8, color="#334155")
        median = float(table.rmse_proposed.median())
        axis.axhline(median, color="#59636e", linestyle="--", linewidth=1.1,
                     label=f"Participant median = {median:.2f}")
        ids = table.userID.astype(str)
        common_prefix = "2133-" if ids.str.startswith("2133-").all() else ""
        labels = ids.str.removeprefix(common_prefix) if common_prefix else ids
        axis.set_xticks(x, labels, rotation=60, ha="right", fontsize=9)
        axis.set(xlim=(-0.7, len(table) - 0.3), ylim=(0, max(table.rmse_proposed.max() * 1.22, 1)),
                 ylabel="Participant RMSE (mg/dL)",
                 xlabel=f"Participant ID in ascending order{(' (prefix ' + common_prefix + ')') if common_prefix else ''}")
        axis.yaxis.grid(True, color="#e5e7eb", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
        fig.suptitle("Fitting performance by participant ID", fontsize=16, y=0.98)
        fig.text(0.5, 0.915,
                 f"Proposed model · {len(table)} participants · {int(table.n_events.sum())} meal windows · "
                 f"{int(table.n_observations.sum()):,} CGM observations · in-sample",
                 ha="center", fontsize=10, color="#55616f")
        fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.20)
        for suffix in [".png", ".pdf"]:
            fig.savefig(destination.with_suffix(suffix), dpi=250, bbox_inches="tight")
        plt.close(fig)


def plot_parameter_sweep(trajectories: pd.DataFrame, summary: pd.DataFrame, metadata: dict, destination: Path) -> None:
    factors = sorted(trajectories.factor.unique())
    palette = plt.get_cmap("viridis")(np.linspace(0.06, 0.93, len(factors)))
    colors = {factor: color for factor, color in zip(factors, palette)}
    colors[1.0] = "#202020"
    horizon = float(metadata["sweep_config"]["horizon_hours"]) * 60
    initial = metadata["initial_state_before_absorption"][1]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(14.3, 8.5), sharex=True, sharey=True)
        low, high = trajectories.glucose.min(), trajectories.glucose.max()
        span = max(high - low, 10)
        for index, (axis, parameter) in enumerate(zip(axes.flat, SWEEP_PARAMETERS)):
            panel = trajectories.loc[trajectories.varied_parameter.eq(parameter)]
            axis.axhline(initial, color="#9ca3af", linewidth=0.8, linestyle=":")
            if horizon > 150:
                axis.axvline(150, color="#c5cbd3", linestyle="--", linewidth=0.8)
            for factor in factors:
                curve = panel.loc[panel.factor.eq(factor)].sort_values("time_min")
                # A dashed baseline remains visible when small beta0 changes overlap.
                axis.plot(curve.time_min, curve.glucose, color=colors[factor],
                          linewidth=2.0 if factor == 1 else 1.5,
                          linestyle="--" if factor == 1 else "-", zorder=4 if factor == 1 else 2)
            nominal = metadata["nominal_coefficients"][parameter]
            axis.set_title(f"{chr(65 + index)}  |  {SYMBOLS[parameter]} = {nominal:.4g} {UNITS[parameter]}",
                           loc="left", fontsize=11, pad=10)
            values = summary.loc[summary.varied_parameter.eq(parameter)].sort_values("factor").parameter_value
            axis.text(0.98, 0.96, f"Range: {values.min():.3g}–{values.max():.3g}", transform=axis.transAxes,
                      ha="right", va="top", fontsize=8.5, color="#55616f",
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
            axis.set(xlim=(0, horizon), ylim=(low - 0.10 * span, high + 0.17 * span))
            axis.grid(alpha=0.16)
        for axis in axes[1]:
            axis.set_xlabel("Time after meal (min)")
        for axis in axes[:, 0]:
            axis.set_ylabel("Simulated glucose (mg/dL)")
        handles = [Line2D([0], [0], color=colors[f], linewidth=2, linestyle="--" if f == 1 else "-",
                          label=f"{f:g}× nominal" + (" (reference)" if f == 1 else "")) for f in factors]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(factors), 5), frameon=False,
                   bbox_to_anchor=(0.5, 0.06), fontsize=10)
        fig.suptitle("One-at-a-time parameter exploration of the uncontrolled model", fontsize=16, y=0.985)
        fig.text(0.5, 0.94,
                 f"Reference: {metadata['reference_id']} ({metadata['reference_status']}) · "
                 f"G(0) = {initial:g} mg/dL · fixed meal and full initial state · no refitting",
                 ha="center", fontsize=10, color="#55616f")
        fig.text(0.5, 0.025,
                 "Other coefficients stay fixed, including β₂ outside its own panel. "
                 "Beyond 150 min: unvalidated extrapolation. Overlapping curves indicate small effects.",
                 ha="center", fontsize=9, color="#55616f")
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.86, hspace=0.28, wspace=0.17)
        for suffix in [".png", ".pdf"]:
            fig.savefig(destination.with_suffix(suffix), dpi=250, bbox_inches="tight")
        plt.close(fig)


def _write_caption(output: Path, table: pd.DataFrame, metadata: dict) -> None:
    nominal = metadata["nominal_coefficients"]
    factors = ", ".join(f"{factor:g}" for factor in sorted(metadata["sweep_config"]["factors"]))
    coefficients = ", ".join(f"{SYMBOLS[name]}={nominal[name]:.6g}" for name in SWEEP_PARAMETERS)
    initial = metadata["initial_state_before_absorption"]
    try:
        figure_root = (output.resolve().relative_to(Path.cwd()) / "figures").as_posix()
    except ValueError:
        figure_root = (output.resolve() / "figures").as_posix()
    lines = [
        "% Requires graphicx. Paths assume compilation from repository root.",
        r"\begin{figure*}[t]", r"\centering",
        r"\includegraphics[width=\textwidth]{" + figure_root + "/figure_8_fit_by_id.pdf}",
        r"\caption{Participant-level fitting performance in ascending participant-ID order. "
        f"All {len(table)} fitted participants, {int(table.n_events.sum())} meal windows and "
        f"{int(table.n_observations.sum()):,} CGM observations are included. "
        r"For each participant, RMSE is computed from all their fitted meal observations as "
        r"$\sqrt{N_i^{-1}\sum_{e,t}(\widehat G_{iet}-G_{iet})^2}$, without observation weights. "
        r"Points are colored and shaped by clinical status; numeric labels show RMSE in mg/dL. "
        r"The dashed horizontal line is the median of participant RMSE values, not pooled RMSE. "
        r"IDs are equally spaced in sorted order, with no implication of a continuous covariate. "
        r"These are in-sample errors, including meal-time observations used for initialization; "
        r"no participant is removed based on fitting quality.}",
        r"\label{fig:fit-by-id}", r"\end{figure*}", "",
        r"\begin{figure*}[t]", r"\centering",
        r"\includegraphics[width=\textwidth]{" + figure_root + "/figure_9_parameter_sweep.pdf}",
        r"\caption{One-at-a-time numerical exploration of the implemented uncontrolled model. "
        f"The nominal coefficient vector is taken from fitted participant {metadata['reference_id']} "
        f"({metadata['reference_status']}); the reference-selection rule is recorded in the experiment metadata. "
        f"Each coefficient is multiplied by {factors}, while all other coefficients remain fixed. "
        r"The dashed black curve is the nominal response. "
        r"The same initial digestive glucose, modeled glucose and quick-insulin state "
        r"are used for every scenario; initial insulin is not re-equilibrated after perturbation. "
        r"Nominal $\beta_2$ is derived from the fitted equilibrium constraint and then frozen, except "
        r"in its own panel, where it is perturbed independently for this dynamical experiment only. "
        r"Thus the experiment does not impose the fitting-stage equilibrium constraint on every "
        r"perturbed coefficient vector. Horizontal dotted lines indicate nominal initial glucose; "
        r"vertical dashed lines, when shown, mark the 150-minute endpoint of the fitting windows. "
        r"Later behavior is unvalidated extrapolation. Multipliers are not clipped to optimization "
        r"bounds; out-of-bound scenarios are flagged in the exported tables. The curves are not "
        r"confidence bands, calibrated physiological interventions, or evidence of clinical efficacy.}",
        r"\label{fig:parameter-sweep}", r"\end{figure*}", "",
        r"\paragraph{Reproducible simulation settings.}",
        f"Nominal coefficients: {coefficients}. "
        f"Initial states are $D(0)=0$, $G(0)={initial[1]:.6g}$ mg/dL and "
        f"$I_q(0)={initial[2]:.6g}$ model units. "
        f"Digestive-glucose amplitude is {metadata['meal_amplitude']:.6g}, injected after an "
        f"absorption delay of {metadata['absorption_delay_min']:.6g} minutes; "
        f"these are the reference participant's median fitted event amplitude and delay. "
        f"The horizon is {metadata['sweep_config']['horizon_hours']:g} hours, with nominal output spacing "
        f"of {metadata['sweep_config']['step_minutes']:g} minute(s), plus the absorption instant and endpoint. "
        f"Fixed constants are $K={metadata['model_config']['k_half_mg_dl']:g}$ mg/dL and "
        f"$I_b={metadata['model_config']['basal_insulin']:g}$ model units; external control is zero. "
        r"Time in the equations is in hours; insulin is dimensionless and is not a clinical dose.", "",
        r"\paragraph{Scope of the dynamical analysis.}",
        r"The export includes the positive no-meal equilibrium and the eigenvalues of its exact "
        r"three-state Jacobian for each coefficient vector. Negative real parts indicate local "
        r"asymptotic stability of that equilibrium, while a complex pair allows damped oscillatory "
        r"linearized modes. Neither a complex pair nor a finite simulated trajectory establishes "
        r"a sustained oscillation, a limit cycle, or a global stability result. The analysis concerns "
        r"one fitted reference scenario, not global parameter sensitivity or parameter identifiability.", "",
    ]
    manuscript = output / "manuscript"
    manuscript.mkdir(parents=True, exist_ok=True)
    (manuscript / "fit_by_id_and_dynamics.tex").write_text("\n".join(lines), encoding="utf-8")


def exploration_report_lines(summary: pd.DataFrame, metadata: dict) -> list[str]:
    return [
        "<!-- BEGIN ID_FIT_AND_DYNAMICS -->", "", "## ID-ordered fitting performance and parameter exploration", "",
        "Figures: [participant RMSE by ascending ID](figures/figure_8_fit_by_id.png) and "
        "[one-at-a-time parameter sweep](figures/figure_9_parameter_sweep.png). "
        "Both have vector PDF versions. The ID plot includes all fitted participants, not only favorable fits.", "",
        f"The dynamical reference is {metadata['reference_id']} ({metadata['reference_status']}); "
        f"selection: {metadata['reference_selection']}. "
        f"Each coefficient is multiplied by {', '.join(str(f) for f in metadata['sweep_config']['factors'])}, "
        f"with a {metadata['sweep_config']['horizon_hours']:g}-hour horizon and no control.", "",
        "This is a strict one-coefficient-at-a-time experiment: the other coefficients, full initial state, "
        "meal amplitude and delay stay fixed. Nominal beta2 is derived once, then held fixed outside its own panel. "
        "Varying beta2 independently is an exploratory override only; it does not alter parameter fitting. "
        "Because the starting state is not re-equilibrated, transient adjustment before absorption can occur.", "",
        f"All {len(summary)} scenario summaries are exported; "
        f"{int(summary.outside_original_fit_bounds.sum())} perturbed values lie outside their original fitting bounds "
        "(no clipping). The no-meal equilibrium Jacobian has strictly negative eigenvalue real parts in "
        f"{int(summary.locally_asymptotically_stable.sum())}/{len(summary)} scenarios. "
        "This is local equilibrium stability, not proof of global stability or sustained oscillations. "
        "The factors are not uncertainty intervals; conclusions concern this reference scenario. "
        "Behavior beyond 150 minutes is unvalidated extrapolation.", "",
        "Exports: [ID-ordered fit data](tables/fit_performance_by_id.csv), "
        "[all sweep trajectories](tables/parameter_sweep_trajectories.csv), "
        "[scenario metrics and equilibrium eigenvalues](tables/parameter_sweep_summary.csv), "
        "[complete configuration](tables/parameter_sweep_config.json), "
        "[manuscript captions](manuscript/fit_by_id_and_dynamics.tex).", "",
        "<!-- END ID_FIT_AND_DYNAMICS -->", "",
    ]


def update_exploration_report(output: Path, summary: pd.DataFrame, metadata: dict) -> None:
    report = output / "analysis_report.md"
    if not report.exists():
        return
    text = report.read_text(encoding="utf-8")
    start, end = "<!-- BEGIN ID_FIT_AND_DYNAMICS -->", "<!-- END ID_FIT_AND_DYNAMICS -->"
    block = "\n".join(exploration_report_lines(summary, metadata))
    if start in text:
        before, rest = text.split(start, 1)
        if end not in rest:
            raise ValueError("Existing exploration report section has an unmatched marker")
        _, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block
    report.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_exploration_artifacts(
    output: Path, participants: pd.DataFrame, predictions: pd.DataFrame, comparisons: pd.DataFrame,
    reference_id: str | None = None, sweep: SweepConfig | None = None, model: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    evaluation = evaluate_fits(participants, predictions, comparisons)
    table = fit_performance_by_id(evaluation["participants"])
    trajectories, summary, metadata = run_parameter_sweep(participants, reference_id, sweep, model)
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "fit_performance_by_id.csv", index=False)
    trajectories.to_csv(tables / "parameter_sweep_trajectories.csv", index=False)
    summary.to_csv(tables / "parameter_sweep_summary.csv", index=False)
    (tables / "parameter_sweep_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot_fit_by_id(table, figures / "figure_8_fit_by_id")
    plot_parameter_sweep(trajectories, summary, metadata, figures / "figure_9_parameter_sweep")
    _write_caption(output, table, metadata)
    update_exploration_report(output, summary, metadata)
    return table, summary, metadata
