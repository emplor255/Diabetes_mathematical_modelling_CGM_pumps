from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import differential_evolution, least_squares

from .model import (
    ModelConfig,
    derive_beta2,
    simulate_meal,
    simulate_reduced_model,
)


KINETIC_BOUNDS = np.array(
    [
        [0.05, 10.0],  # alpha, h^-1
        [1e-4, 2.0],  # beta0, h^-1
        [1e-6, 1.0],  # beta1, (dimensionless insulin * h)^-1
        [1e-3, 10.0],  # gamma, dimensionless insulin / h
        [0.05, 10.0],  # delta, h^-1
    ],
    dtype=float,
)
AMPLITUDE_BOUNDS = (1.0, 1000.0)
DELAY_BOUNDS = (0.0, 0.75)
REDUCED_KINETIC_BOUNDS = np.array([[0.05, 10.0], [0.01, 10.0]], dtype=float)


@dataclass(frozen=True)
class FitConfig:
    seed: int = 20260815
    stage1_iterations: int = 12
    stage1_popsize: int = 5
    stage2_max_nfev: int = 180
    smoothing_sd_mg_dl: float = 3.0
    observation_sd_mg_dl: float = 5.0
    log_prior_sd: float = 4.0
    model: ModelConfig = ModelConfig()


@dataclass(frozen=True)
class EventData:
    event_id: str
    meal: str
    meal_type: str
    times: np.ndarray
    glucose: np.ndarray
    weights: np.ndarray
    initial_glucose: float


def _stable_seed(base_seed: int, participant_id: str, suffix: str) -> int:
    token = f"{base_seed}:{participant_id}:{suffix}".encode()
    return (base_seed + int(sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def _events_from_frame(points: pd.DataFrame) -> tuple[list[EventData], float]:
    events: list[EventData] = []
    premeal_values: list[float] = []
    for event_id, event in points.groupby("event_id", sort=True):
        event = event.sort_values("relative_min")
        relative_hours = event["relative_min"].to_numpy(dtype=float) / 60.0
        glucose_all = event["glucose"].to_numpy(dtype=float)
        premeal_values.extend(glucose_all[relative_hours < 0].tolist())
        initial_glucose = float(np.interp(0.0, relative_hours, glucose_all))
        post = relative_hours >= 0
        if post.sum() < 8:
            continue
        events.append(
            EventData(
                event_id=str(event_id),
                meal=str(event["Meal"].iloc[0]),
                meal_type=str(event["meal_type"].iloc[0]),
                times=relative_hours[post],
                glucose=glucose_all[post],
                weights=event["weight"].to_numpy(dtype=float)[post],
                initial_glucose=initial_glucose,
            )
        )
    if not events or not premeal_values:
        raise ValueError("Participant has no usable post-meal event with pre-meal baseline")
    return events, float(np.median(premeal_values))


def _representative_curve(
    events: list[EventData], smoothing_sd: float
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0.0, 2.5 + 1e-9, 5.0 / 60.0)
    curves = np.vstack(
        [np.interp(grid, event.times, event.glucose) for event in events]
    )
    median_curve = np.median(curves, axis=0)
    spline = UnivariateSpline(
        grid,
        median_curve,
        s=len(grid) * float(smoothing_sd) ** 2,
        k=min(3, len(grid) - 1),
    )
    return grid, spline(grid)


def _proposed_stage1(
    events: list[EventData],
    reference_glucose: float,
    config: FitConfig,
    participant_id: str,
) -> tuple[np.ndarray, float]:
    grid, smooth = _representative_curve(events, config.smoothing_sd_mg_dl)
    initial = float(smooth[0])
    bounds = [tuple(row) for row in KINETIC_BOUNDS]
    bounds.extend([AMPLITUDE_BOUNDS, DELAY_BOUNDS])

    def objective(candidate: np.ndarray) -> float:
        try:
            predicted = simulate_meal(
                grid,
                candidate[:5],
                candidate[5],
                candidate[6],
                initial,
                reference_glucose,
                config.model,
            )[:, 1]
        except (ValueError, RuntimeError, FloatingPointError):
            return 1e12
        if not np.all(np.isfinite(predicted)):
            return 1e12
        return float(np.mean((predicted - smooth) ** 2))

    result = differential_evolution(
        objective,
        bounds,
        seed=_stable_seed(config.seed, participant_id, "proposed"),
        maxiter=config.stage1_iterations,
        popsize=config.stage1_popsize,
        tol=0.01,
        polish=True,
        updating="immediate",
        workers=1,
    )
    return result.x, float(result.fun)


def _reduced_stage1(
    events: list[EventData],
    reference_glucose: float,
    config: FitConfig,
    participant_id: str,
) -> tuple[np.ndarray, float]:
    grid, smooth = _representative_curve(events, config.smoothing_sd_mg_dl)
    initial = float(smooth[0])
    bounds = [tuple(row) for row in REDUCED_KINETIC_BOUNDS]
    bounds.extend([AMPLITUDE_BOUNDS, DELAY_BOUNDS])

    def objective(candidate: np.ndarray) -> float:
        predicted = simulate_reduced_model(
            grid,
            candidate[0],
            candidate[1],
            candidate[2],
            candidate[3],
            initial,
            reference_glucose,
        )
        return float(np.mean((predicted - smooth) ** 2))

    result = differential_evolution(
        objective,
        bounds,
        seed=_stable_seed(config.seed, participant_id, "reduced"),
        maxiter=max(6, config.stage1_iterations // 2),
        popsize=max(4, config.stage1_popsize - 1),
        tol=0.01,
        polish=True,
        updating="immediate",
        workers=1,
    )
    return result.x, float(result.fun)


def _information_criteria(sse: float, n_obs: int, n_parameters: int) -> tuple[float, float]:
    variance = max(float(sse) / max(n_obs, 1), np.finfo(float).tiny)
    aic = n_obs * np.log(variance) + 2 * n_parameters
    bic = n_obs * np.log(variance) + n_parameters * np.log(max(n_obs, 2))
    return float(aic), float(bic)


def _normalized_condition_number(jacobian: np.ndarray) -> float:
    if jacobian.size == 0:
        return float("nan")
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(norms <= np.finfo(float).eps):
        return float("inf")
    singular = np.linalg.svd(jacobian / norms, compute_uv=False)
    if singular[-1] <= np.finfo(float).eps:
        return float("inf")
    return float(singular[0] / singular[-1])


def fit_participant(
    participant_id: str,
    points: pd.DataFrame,
    config: FitConfig,
) -> dict[str, Any]:
    events, reference_glucose = _events_from_frame(points)
    n_events = len(events)
    n_observations = int(sum(len(event.times) for event in events))
    status = str(points["status"].iloc[0])
    type_code = float(points["type_code"].iloc[0])
    age = float(points["age"].iloc[0])

    warm, stage1_mse = _proposed_stage1(
        events, reference_glucose, config, participant_id
    )
    x0 = np.r_[warm[:5], np.repeat(warm[5], n_events), np.repeat(warm[6], n_events)]
    lower = np.r_[
        KINETIC_BOUNDS[:, 0],
        np.repeat(AMPLITUDE_BOUNDS[0], n_events),
        np.repeat(DELAY_BOUNDS[0], n_events),
    ]
    upper = np.r_[
        KINETIC_BOUNDS[:, 1],
        np.repeat(AMPLITUDE_BOUNDS[1], n_events),
        np.repeat(DELAY_BOUNDS[1], n_events),
    ]

    def proposed_residual(candidate: np.ndarray) -> np.ndarray:
        residuals: list[np.ndarray] = []
        for index, event in enumerate(events):
            try:
                predicted = simulate_meal(
                    event.times,
                    candidate[:5],
                    candidate[5 + index],
                    candidate[5 + n_events + index],
                    event.initial_glucose,
                    reference_glucose,
                    config.model,
                )[:, 1]
                residual = (
                    np.sqrt(event.weights)
                    * (predicted - event.glucose)
                    / config.observation_sd_mg_dl
                )
            except (ValueError, RuntimeError, FloatingPointError):
                residual = np.full(len(event.times), 1e6)
            residuals.append(residual)
        # Very weak stabilization; its contribution is reported and parameter
        # identifiability is still diagnosed from data-only Jacobian rows.
        residuals.append(
            (np.log(candidate[:5]) - np.log(warm[:5])) / config.log_prior_sd
        )
        return np.concatenate(residuals)

    proposed = least_squares(
        proposed_residual,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=config.stage2_max_nfev,
        x_scale="jac",
        diff_step=1e-4,
    )
    proposed_x = proposed.x

    reduced_warm, reduced_stage1_mse = _reduced_stage1(
        events, reference_glucose, config, participant_id
    )
    reduced_x0 = np.r_[
        reduced_warm[:2],
        np.repeat(reduced_warm[2], n_events),
        np.repeat(reduced_warm[3], n_events),
    ]
    reduced_lower = np.r_[
        REDUCED_KINETIC_BOUNDS[:, 0],
        np.repeat(AMPLITUDE_BOUNDS[0], n_events),
        np.repeat(DELAY_BOUNDS[0], n_events),
    ]
    reduced_upper = np.r_[
        REDUCED_KINETIC_BOUNDS[:, 1],
        np.repeat(AMPLITUDE_BOUNDS[1], n_events),
        np.repeat(DELAY_BOUNDS[1], n_events),
    ]

    def reduced_residual(candidate: np.ndarray) -> np.ndarray:
        residuals: list[np.ndarray] = []
        for index, event in enumerate(events):
            predicted = simulate_reduced_model(
                event.times,
                candidate[0],
                candidate[1],
                candidate[2 + index],
                candidate[2 + n_events + index],
                event.initial_glucose,
                reference_glucose,
            )
            residuals.append(
                np.sqrt(event.weights)
                * (predicted - event.glucose)
                / config.observation_sd_mg_dl
            )
        residuals.append(
            (np.log(candidate[:2]) - np.log(reduced_warm[:2])) / config.log_prior_sd
        )
        return np.concatenate(residuals)

    reduced = least_squares(
        reduced_residual,
        reduced_x0,
        bounds=(reduced_lower, reduced_upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=config.stage2_max_nfev,
        x_scale="jac",
        diff_step=1e-4,
    )
    reduced_x = reduced.x

    prediction_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    proposed_sse = 0.0
    reduced_sse = 0.0
    for index, event in enumerate(events):
        proposed_prediction = simulate_meal(
            event.times,
            proposed_x[:5],
            proposed_x[5 + index],
            proposed_x[5 + n_events + index],
            event.initial_glucose,
            reference_glucose,
            config.model,
        )[:, 1]
        reduced_prediction = simulate_reduced_model(
            event.times,
            reduced_x[0],
            reduced_x[1],
            reduced_x[2 + index],
            reduced_x[2 + n_events + index],
            event.initial_glucose,
            reference_glucose,
        )
        proposed_errors = proposed_prediction - event.glucose
        reduced_errors = reduced_prediction - event.glucose
        proposed_sse += float(np.sum(proposed_errors**2))
        reduced_sse += float(np.sum(reduced_errors**2))
        denominator = float(np.sum((event.glucose - event.glucose.mean()) ** 2))
        event_rows.append(
            {
                "userID": participant_id,
                "event_id": event.event_id,
                "Meal": event.meal,
                "meal_type": event.meal_type,
                "status": status,
                "n_points": int(len(event.times)),
                "meal_amplitude": float(proposed_x[5 + index]),
                "absorption_delay_min": float(
                    60.0 * proposed_x[5 + n_events + index]
                ),
                "rmse_proposed": float(np.sqrt(np.mean(proposed_errors**2))),
                "mae_proposed": float(np.mean(np.abs(proposed_errors))),
                "r2_proposed": float(1.0 - np.sum(proposed_errors**2) / denominator)
                if denominator > 0
                else np.nan,
                "rmse_reduced": float(np.sqrt(np.mean(reduced_errors**2))),
                "mae_reduced": float(np.mean(np.abs(reduced_errors))),
            }
        )
        for time, observed, weight, pred, reduced_pred in zip(
            event.times,
            event.glucose,
            event.weights,
            proposed_prediction,
            reduced_prediction,
        ):
            prediction_rows.append(
                {
                    "userID": participant_id,
                    "event_id": event.event_id,
                    "Meal": event.meal,
                    "meal_type": event.meal_type,
                    "status": status,
                    "time_min": float(60.0 * time),
                    "observed_glucose": float(observed),
                    "weight": float(weight),
                    "proposed_glucose": float(pred),
                    "reduced_glucose": float(reduced_pred),
                }
            )

    proposed_rmse = float(np.sqrt(proposed_sse / n_observations))
    reduced_rmse = float(np.sqrt(reduced_sse / n_observations))
    proposed_aic, proposed_bic = _information_criteria(
        proposed_sse, n_observations, 5 + 2 * n_events
    )
    reduced_aic, reduced_bic = _information_criteria(
        reduced_sse, n_observations, 2 + 2 * n_events
    )
    kinetic_jacobian = proposed.jac[:n_observations, :5]
    condition_number = _normalized_condition_number(kinetic_jacobian)
    relative_position = (proposed_x[:5] - KINETIC_BOUNDS[:, 0]) / (
        KINETIC_BOUNDS[:, 1] - KINETIC_BOUNDS[:, 0]
    )
    bound_hits = int(((relative_position < 0.005) | (relative_position > 0.995)).sum())
    beta2 = derive_beta2(proposed_x[:5], reference_glucose, config.model)

    participant_row = {
        "userID": participant_id,
        "type_code": type_code,
        "status": status,
        "age": age,
        "n_events": n_events,
        "n_observations": n_observations,
        "reference_glucose": reference_glucose,
        "alpha": float(proposed_x[0]),
        "beta0": float(proposed_x[1]),
        "beta1": float(proposed_x[2]),
        "beta2_derived": beta2,
        "gamma": float(proposed_x[3]),
        "delta": float(proposed_x[4]),
        "median_meal_amplitude": float(np.median(proposed_x[5 : 5 + n_events])),
        "median_absorption_delay_min": float(
            60.0 * np.median(proposed_x[5 + n_events :])
        ),
        "stage1_mse": stage1_mse,
        "rmse_proposed": proposed_rmse,
        "rmse_reduced": reduced_rmse,
        "jacobian_condition_number": condition_number,
        "kinetic_bound_hits": bound_hits,
        "optimizer_success": bool(proposed.success),
        "optimizer_status": int(proposed.status),
        "optimizer_nfev": int(proposed.nfev),
    }
    comparison_rows = [
        {
            "userID": participant_id,
            "status": status,
            "model": "Proposed 3-state",
            "states": 3,
            "shared_kinetic_parameters": 5,
            "event_nuisance_parameters": 2 * n_events,
            "n_observations": n_observations,
            "sse": proposed_sse,
            "rmse": proposed_rmse,
            "aic": proposed_aic,
            "bic": proposed_bic,
            "stage1_mse": stage1_mse,
            "optimizer_success": bool(proposed.success),
        },
        {
            "userID": participant_id,
            "status": status,
            "model": "Reduced 2-state",
            "states": 2,
            "shared_kinetic_parameters": 2,
            "event_nuisance_parameters": 2 * n_events,
            "n_observations": n_observations,
            "sse": reduced_sse,
            "rmse": reduced_rmse,
            "aic": reduced_aic,
            "bic": reduced_bic,
            "stage1_mse": reduced_stage1_mse,
            "optimizer_success": bool(reduced.success),
        },
    ]
    return {
        "participant": participant_row,
        "events": event_rows,
        "predictions": prediction_rows,
        "comparison": comparison_rows,
    }


def _fit_worker(payload: tuple[str, pd.DataFrame, dict[str, Any]]) -> dict[str, Any]:
    participant_id, points, config_values = payload
    model_values = config_values.pop("model")
    config = FitConfig(model=ModelConfig(**model_values), **config_values)
    return fit_participant(participant_id, points, config)


def fit_all_participants(
    points: pd.DataFrame,
    config: FitConfig,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = points.loc[points["usable"]].copy()
    grouped = [(str(user), frame.copy()) for user, frame in usable.groupby("userID")]
    config_values = asdict(config)
    payloads = [(user, frame, dict(config_values)) for user, frame in grouped]
    results: list[dict[str, Any]] = []

    if workers <= 1:
        for index, (user, frame, _) in enumerate(payloads, start=1):
            print(f"[{index}/{len(payloads)}] fitting {user}", flush=True)
            results.append(fit_participant(user, frame, config))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fit_worker, payload): payload[0] for payload in payloads}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                user = futures[future]
                result = future.result()
                print(f"[{completed}/{len(payloads)}] fitted {user}", flush=True)
                results.append(result)

    participants = pd.DataFrame([result["participant"] for result in results]).sort_values(
        "userID"
    )
    events = pd.DataFrame(
        [row for result in results for row in result["events"]]
    ).sort_values(["userID", "event_id"])
    predictions = pd.DataFrame(
        [row for result in results for row in result["predictions"]]
    ).sort_values(["userID", "event_id", "time_min"])
    comparisons = pd.DataFrame(
        [row for result in results for row in result["comparison"]]
    ).sort_values(["userID", "model"])
    return participants, events, predictions, comparisons
