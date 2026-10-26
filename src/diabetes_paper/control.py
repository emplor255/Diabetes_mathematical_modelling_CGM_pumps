from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .model import ModelConfig, derive_beta2, proposed_rhs, quick_insulin_equilibrium


@dataclass(frozen=True)
class ControlConfig:
    horizon_hours: float = 2.5
    integration_step_minutes: float = 1.0
    control_bin_minutes: float = 15.0
    max_control_intensity: float = 10.0
    control_penalty: float = 1.0
    optimizer_maxiter: int = 60


def _rk4_step(function, time: float, state: np.ndarray, step: float) -> np.ndarray:
    k1 = function(time, state)
    k2 = function(time + step / 2.0, state + step * k1 / 2.0)
    k3 = function(time + step / 2.0, state + step * k2 / 2.0)
    k4 = function(time + step, state + step * k3)
    updated = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    # Numerical enforcement of the paper's positivity criterion.
    updated[0] = max(updated[0], 0.0)
    updated[1] = max(updated[1], 1.0)
    updated[2] = max(updated[2], 0.0)
    return updated


def simulate_piecewise_control(
    parameters: Sequence[float],
    reference_glucose: float,
    meal_amplitude: float,
    absorption_delay_hours: float,
    controls: Sequence[float],
    model_config: ModelConfig | None = None,
    control_config: ControlConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_cfg = model_config or ModelConfig()
    control_cfg = control_config or ControlConfig()
    step = control_cfg.integration_step_minutes / 60.0
    times = np.arange(0.0, control_cfg.horizon_hours + step / 2.0, step)
    controls = np.asarray(controls, dtype=float)
    expected_bins = int(
        np.ceil(control_cfg.horizon_hours * 60.0 / control_cfg.control_bin_minutes)
    )
    if len(controls) != expected_bins:
        raise ValueError(f"Expected {expected_bins} control bins, received {len(controls)}")

    beta2 = derive_beta2(parameters, reference_glucose, model_cfg)
    quick0 = quick_insulin_equilibrium(reference_glucose, parameters, model_cfg)
    state = np.array([0.0, reference_glucose, quick0], dtype=float)
    states = np.empty((len(times), 3), dtype=float)
    states[0] = state
    applied = np.empty(len(times), dtype=float)
    meal_added = False

    def control_at(time: float) -> float:
        index = min(
            int(time * 60.0 // control_cfg.control_bin_minutes), len(controls) - 1
        )
        return float(np.clip(controls[index], 0.0, control_cfg.max_control_intensity))

    if absorption_delay_hours <= 1e-12:
        state[0] += meal_amplitude
        meal_added = True

    for index, time in enumerate(times[:-1]):
        if not meal_added and time + step / 2.0 >= absorption_delay_hours:
            state[0] += meal_amplitude
            meal_added = True
        u = control_at(time)
        applied[index] = u

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return proposed_rhs(t, y, parameters, beta2, model_cfg, control_at(t))

        state = _rk4_step(rhs, time, state, step)
        states[index + 1] = state
    applied[-1] = control_at(times[-1])
    return times, states, applied


def paper_objective(
    times: np.ndarray,
    glucose: np.ndarray,
    control: np.ndarray,
    model_config: ModelConfig,
    control_config: ControlConfig,
) -> float:
    horizon = max(float(times[-1]), np.finfo(float).eps)
    glucose_cost = (
        (times / horizon)
        * (glucose - model_config.target_lower_mg_dl)
        * (glucose - model_config.target_upper_mg_dl)
    )
    running = glucose_cost + control_config.control_penalty * control**2
    return float(np.trapz(running, times))


def _trajectory_metrics(
    times: np.ndarray,
    glucose: np.ndarray,
    control: np.ndarray,
    model_config: ModelConfig,
    control_config: ControlConfig,
) -> dict[str, float]:
    return {
        "paper_objective": paper_objective(
            times, glucose, control, model_config, control_config
        ),
        "total_control_intensity": float(np.trapz(control, times)),
        "max_control_intensity": float(np.max(control)),
        "time_at_control_upper_bound_pct": float(
            100.0
            * np.mean(
                control >= 0.995 * control_config.max_control_intensity
            )
        ),
        "mean_glucose": float(np.mean(glucose)),
        "min_glucose": float(np.min(glucose)),
        "peak_glucose": float(np.max(glucose)),
        "time_70_110_pct": float(
            100.0
            * np.mean(
                (glucose >= model_config.target_lower_mg_dl)
                & (glucose <= model_config.target_upper_mg_dl)
            )
        ),
        "time_below_70_pct": float(
            100.0 * np.mean(glucose < model_config.target_lower_mg_dl)
        ),
        "time_above_110_pct": float(
            100.0 * np.mean(glucose > model_config.target_upper_mg_dl)
        ),
    }


def optimize_participant_control(
    participant: pd.Series,
    model_config: ModelConfig | None = None,
    control_config: ControlConfig | None = None,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    model_cfg = model_config or ModelConfig()
    control_cfg = control_config or ControlConfig()
    parameters = np.array(
        [
            participant["alpha"],
            participant["beta0"],
            participant["beta1"],
            participant["gamma"],
            participant["delta"],
        ],
        dtype=float,
    )
    reference = float(participant["reference_glucose"])
    amplitude = float(participant["median_meal_amplitude"])
    delay = float(participant["median_absorption_delay_min"]) / 60.0
    n_bins = int(
        np.ceil(control_cfg.horizon_hours * 60.0 / control_cfg.control_bin_minutes)
    )

    def objective(values: np.ndarray) -> float:
        time, states, applied = simulate_piecewise_control(
            parameters,
            reference,
            amplitude,
            delay,
            values,
            model_cfg,
            control_cfg,
        )
        return paper_objective(time, states[:, 1], applied, model_cfg, control_cfg)

    bounds = [(0.0, control_cfg.max_control_intensity)] * n_bins
    starts = [np.zeros(n_bins), np.full(n_bins, 0.1 * control_cfg.max_control_intensity)]
    candidates = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": control_cfg.optimizer_maxiter, "ftol": 1e-9},
        )
        for start in starts
    ]
    best = min(candidates, key=lambda result: float(result.fun))
    optimized_controls = np.clip(best.x, 0.0, control_cfg.max_control_intensity)
    zero_controls = np.zeros(n_bins)

    summaries: list[dict[str, float | str]] = []
    trajectories: list[dict[str, float | str]] = []
    for scenario, controls in [
        ("No external control", zero_controls),
        ("Optimized unitless control", optimized_controls),
    ]:
        times, states, applied = simulate_piecewise_control(
            parameters,
            reference,
            amplitude,
            delay,
            controls,
            model_cfg,
            control_cfg,
        )
        summary: dict[str, float | str] = {
            "userID": str(participant["userID"]),
            "status": str(participant["status"]),
            "scenario": scenario,
            "optimizer_success": bool(best.success),
        }
        summary.update(
            _trajectory_metrics(times, states[:, 1], applied, model_cfg, control_cfg)
        )
        summaries.append(summary)
        trajectories.extend(
            {
                "userID": str(participant["userID"]),
                "status": str(participant["status"]),
                "scenario": scenario,
                "time_min": float(60.0 * time),
                "digestive_glucose": float(state[0]),
                "glucose": float(state[1]),
                "quick_insulin_state": float(state[2]),
                "control_intensity": float(control_value),
            }
            for time, state, control_value in zip(times, states, applied)
        )
    return summaries, trajectories


def optimize_all_controls(
    participants: pd.DataFrame,
    model_config: ModelConfig | None = None,
    control_config: ControlConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, float | str]] = []
    trajectories: list[dict[str, float | str]] = []
    for index, (_, participant) in enumerate(participants.iterrows(), start=1):
        print(
            f"[{index}/{len(participants)}] optimizing unitless control for {participant['userID']}",
            flush=True,
        )
        participant_summary, participant_trajectory = optimize_participant_control(
            participant, model_config, control_config
        )
        summaries.extend(participant_summary)
        trajectories.extend(participant_trajectory)
    return pd.DataFrame(summaries), pd.DataFrame(trajectories)
