from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp


PARAMETER_NAMES = ("alpha", "beta0", "beta1", "gamma", "delta")


@dataclass(frozen=True)
class ModelConfig:
    """Fixed quantities that cannot be identified from CGM alone.

    Insulin is dimensionless because neither workbook contains insulin concentration
    or dose. ``quick_insulin`` denotes the non-basal component I-bar in the paper.
    """

    k_half_mg_dl: float = 120.0
    basal_insulin: float = 1.0
    target_lower_mg_dl: float = 70.0
    target_upper_mg_dl: float = 110.0


class ModelSimulationError(RuntimeError):
    pass


def quick_insulin_equilibrium(
    glucose: float, parameters: Sequence[float], config: ModelConfig
) -> float:
    _, _, _, gamma, delta = np.asarray(parameters, dtype=float)
    saturation = glucose**2 / (config.k_half_mg_dl**2 + glucose**2)
    return float(gamma * saturation / delta)


def derive_beta2(
    parameters: Sequence[float], reference_glucose: float, config: ModelConfig
) -> float:
    """Derive beta2 by enforcing a no-meal equilibrium at reference glucose.

    The printed paper estimates beta2 while also fitting only G2. That creates a
    severe cancellation/identifiability problem. The equilibrium constraint keeps
    the printed glucose equation intact while making beta2 a reported derived
    parameter rather than a sixth freely fitted kinetic parameter.
    """

    _, beta0, beta1, _, _ = np.asarray(parameters, dtype=float)
    quick = quick_insulin_equilibrium(reference_glucose, parameters, config)
    total_insulin = config.basal_insulin + quick
    return float(
        (beta0 + beta1 * total_insulin)
        * reference_glucose
        * (1.0 + total_insulin)
    )


def proposed_rhs(
    time: float,
    state: np.ndarray,
    parameters: Sequence[float],
    beta2: float,
    config: ModelConfig,
    control: float = 0.0,
) -> np.ndarray:
    del time
    alpha, beta0, beta1, gamma, delta = np.asarray(parameters, dtype=float)
    digestive, glucose, quick_insulin = np.asarray(state, dtype=float)
    total_insulin = config.basal_insulin + max(float(quick_insulin), 0.0)
    saturation = glucose**2 / (config.k_half_mg_dl**2 + glucose**2)
    return np.array(
        [
            -alpha * digestive,
            alpha * digestive
            - beta0 * glucose
            - beta1 * total_insulin * glucose
            + beta2 / (1.0 + total_insulin),
            gamma * saturation - delta * quick_insulin + control,
        ],
        dtype=float,
    )


def _integrate(
    start: float,
    end: float,
    initial: np.ndarray,
    evaluation_times: np.ndarray,
    rhs: Callable[[float, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if end <= start + 1e-12:
        return np.array([start]), initial.reshape(3, 1)
    requested = np.asarray(evaluation_times, dtype=float)
    requested = requested[(requested >= start - 1e-10) & (requested <= end + 1e-10)]
    requested = np.unique(np.r_[requested, end])
    solution = solve_ivp(
        rhs,
        (start, end),
        initial,
        t_eval=requested,
        rtol=3e-5,
        atol=1e-7,
        max_step=1.0 / 24.0,
    )
    if not solution.success or solution.y.shape[1] != len(requested):
        raise ModelSimulationError(solution.message)
    if not np.all(np.isfinite(solution.y)):
        raise ModelSimulationError("Non-finite model state")
    return solution.t, solution.y


def simulate_meal(
    times_hours: Sequence[float],
    parameters: Sequence[float],
    meal_amplitude: float,
    absorption_delay_hours: float,
    initial_glucose: float,
    reference_glucose: float,
    config: ModelConfig | None = None,
    control: Callable[[float], float] | None = None,
    *,
    beta2_override: float | None = None,
    initial_quick_insulin_override: float | None = None,
) -> np.ndarray:
    """Simulate states at non-negative times after the recorded meal time.

    A fitted absorption delay is included because the paper explicitly requires a
    food-to-blood delay but Equation (1) does not contain one. It is treated as an
    event-level nuisance parameter, not a patient kinetic parameter.

    The keyword-only overrides are for dynamical-system experiments. By default,
    beta2 and initial quick insulin retain their original equilibrium definitions.
    Explicit overrides allow a strict one-coefficient-at-a-time sweep with all
    other coefficients and the full initial state held fixed; they are not used
    in parameter fitting or control optimization.
    """

    cfg = config or ModelConfig()
    times = np.asarray(times_hours, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times_hours must be a non-empty one-dimensional array")
    if np.any(times < -1e-10) or np.any(np.diff(times) < 0):
        raise ValueError("times_hours must be sorted and non-negative")
    parameters = np.asarray(parameters, dtype=float)
    if len(parameters) != 5 or np.any(parameters <= 0):
        raise ValueError("parameters must contain five positive values")
    if meal_amplitude < 0 or absorption_delay_hours < 0:
        raise ValueError("meal amplitude and delay must be non-negative")

    if beta2_override is not None and (not np.isfinite(beta2_override) or beta2_override <= 0):
        raise ValueError("beta2_override must be finite and positive")
    if initial_quick_insulin_override is not None and (
        not np.isfinite(initial_quick_insulin_override) or initial_quick_insulin_override < 0
    ):
        raise ValueError("initial_quick_insulin_override must be finite and non-negative")
    beta2 = (derive_beta2(parameters, reference_glucose, cfg)
             if beta2_override is None else float(beta2_override))
    quick0 = (quick_insulin_equilibrium(initial_glucose, parameters, cfg)
              if initial_quick_insulin_override is None else float(initial_quick_insulin_override))
    state0 = np.array([0.0, initial_glucose, quick0], dtype=float)
    delay = min(float(absorption_delay_hours), float(times[-1]))
    output = np.empty((len(times), 3), dtype=float)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        u = 0.0 if control is None else max(float(control(t)), 0.0)
        return proposed_rhs(t, y, parameters, beta2, cfg, u)

    before = times <= delay + 1e-10
    if delay > 1e-12:
        pre_t, pre_y = _integrate(0.0, delay, state0, times[before], rhs)
        for column in range(3):
            output[before, column] = np.interp(times[before], pre_t, pre_y[column])
        state_at_delay = pre_y[:, -1].copy()
    else:
        output[before] = state0
        state_at_delay = state0.copy()

    state_at_delay[0] += float(meal_amplitude)
    after = times > delay + 1e-10
    if after.any():
        post_t, post_y = _integrate(
            delay, float(times[-1]), state_at_delay, times[after], rhs
        )
        for column in range(3):
            output[after, column] = np.interp(times[after], post_t, post_y[column])
    return output


def simulate_reduced_model(
    times_hours: Sequence[float],
    alpha: float,
    clearance: float,
    meal_amplitude: float,
    absorption_delay_hours: float,
    initial_glucose: float,
    reference_glucose: float,
) -> np.ndarray:
    """Analytic two-state ablation: digestive glucose plus blood glucose."""

    times = np.asarray(times_hours, dtype=float)
    delay = float(absorption_delay_hours)
    result = reference_glucose + (initial_glucose - reference_glucose) * np.exp(
        -clearance * times
    )
    after = times > delay
    elapsed = times[after] - delay
    if after.any():
        glucose_at_delay = reference_glucose + (
            initial_glucose - reference_glucose
        ) * np.exp(-clearance * delay)
        if abs(alpha - clearance) < 1e-7:
            meal_term = alpha * meal_amplitude * elapsed * np.exp(-clearance * elapsed)
        else:
            meal_term = (
                alpha
                * meal_amplitude
                * (np.exp(-clearance * elapsed) - np.exp(-alpha * elapsed))
                / (alpha - clearance)
            )
        result[after] = (
            reference_glucose
            + (glucose_at_delay - reference_glucose) * np.exp(-clearance * elapsed)
            + meal_term
        )
    return result
