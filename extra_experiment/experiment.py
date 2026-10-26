"""Recorded-input extension of the paper model, not a clinical dosing model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize

from diabetes_paper.estimation import KINETIC_BOUNDS
from diabetes_paper.model import ModelConfig, derive_beta2, quick_insulin_equilibrium

NAMES = ("alpha", "beta0", "beta1", "gamma", "delta", "meal_gain", "bolus_gain")
BOUNDS = np.vstack([KINETIC_BOUNDS, [0.01, 30.0], [0.0001, 30.0]])


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20261026
    fit_starts: int = 4
    fit_max_nfev: int = 200
    insulin_absorption_per_hour: float = 1.0
    control_bin_minutes: float = 60.0
    control_max: float = 10.0
    control_penalty: float = 1.0
    control_maxiter: int = 80
    rtol: float = 1e-6
    atol: float = 1e-8


def prepare_record(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """No interpolation, clipping, outlier removal, or undocumented Noise weighting."""
    required = ["Timestamp", "BG", "Insulin Bolus", "Noise", "Eating"]
    if not set(required).issubset(frame.columns) or len(frame) < 3:
        raise ValueError("Expected Timestamp, BG, Insulin Bolus, Noise, Eating and >=3 rows")
    frame = frame[required].copy()
    times = pd.to_datetime(frame.Timestamp.astype(str).str.strip(), format="%H:%M", errors="raise")
    minutes = (times.dt.hour * 60 + times.dt.minute).to_numpy(dtype=float)
    # These sources are single-day records. Never silently sort a possible rollover.
    if np.any(np.diff(minutes) < 0):
        raise ValueError("Nonmonotone clock times: supply unambiguous single-day data")
    for column in ["BG", "Insulin Bolus", "Eating"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"Missing/nonfinite {column}; do not silently replace by zero")
    if (frame.BG <= 0).any() or (frame[["Insulin Bolus", "Eating"]] < 0).any().any():
        raise ValueError("Glucose must be positive; recorded inputs must be nonnegative")
    frame["time_min"] = minutes - minutes[0]
    counts = frame.groupby("time_min").size()
    clean = frame.groupby("time_min", as_index=False, sort=True).agg(
        Timestamp=("Timestamp", "first"), glucose=("BG", "mean"),
        bolus=("Insulin Bolus", "sum"), eating=("Eating", "sum"),
        noise=("Noise", lambda x: "|".join(sorted(set(x.astype(str))))),
    )
    if len(clean) < 3 or clean.time_min.iloc[-1] <= 0:
        raise ValueError("At least three distinct times are needed")
    meals = clean.loc[clean.eating > 0, "time_min"]
    if meals.empty:
        raise ValueError("No recorded meal inputs")
    first_meal = float(meals.iloc[0])
    pre = clean.loc[(clean.time_min >= first_meal - 30) & (clean.time_min < first_meal), "glucose"]
    reference = float(pre.median()) if len(pre) else float(clean.glucose.iloc[0])
    audit = {
        "raw_rows": len(frame), "unique_times": len(clean),
        "duplicate_rows_collapsed": int((counts - 1).sum()),
        "duration_min": float(clean.time_min.iloc[-1]),
        "largest_gap_min": float(clean.time_min.diff().max()),
        "recorded_meal_events": int((clean.eating > 0).sum()),
        "recorded_bolus_events": int((clean.bolus > 0).sum()),
        "total_eating_native_units": float(clean.eating.sum()),
        "total_bolus_native_units": float(clean.bolus.sum()),
        "noise_counts_raw": {str(k): int(v) for k, v in frame.Noise.value_counts(dropna=False).items()},
        "reference_glucose": reference,
        "reference_rule": "median in 30 min before first meal" if len(pre) else "first BG; no premeal baseline available",
        "premeal_baseline_points": len(pre),
        "observations_below_70": int((clean.glucose < 70).sum()),
        "bg_unit_assumption": "mg/dL, pending source-owner confirmation",
        "input_units": "native CSV amounts; no grams/U conversion asserted",
        "noise_handling": "labels preserved; all unique-time glucose observations weighted equally",
        "duplicate_handling": "mean BG; sum meal and bolus amounts; preserve Noise labels",
    }
    return clean, audit


def simulate(record, reference, parameters, evaluation_hours, config=None, controls=None,
             *, bolus_multiplier=1.0):
    """Event-exact integration; D/S jump only once per input; G/Q remain continuous.

    Additional depot S has S'=-k_abs*S, and Q'=paper_Q'+k_abs*S+u.
    At recorded meal/bolus times: D+=meal_gain*Eating, S+=bolus_gain*Bolus.
    Evaluation at an event is right-continuous for D/S. No state clipping is used.
    """
    cfg = config or ExperimentConfig()
    model = ModelConfig()
    p = np.asarray(parameters, dtype=float)
    times = np.asarray(evaluation_hours, dtype=float)
    horizon = float(record.time_min.iloc[-1]) / 60.0
    if p.shape != (7,) or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Seven positive, finite parameters required")
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > horizon + 1e-9):
        raise ValueError("Evaluation times must increase inside the recorded horizon")
    if cfg.insulin_absorption_per_hour <= 0 or cfg.control_bin_minutes <= 0:
        raise ValueError("Absorption and control interval must be positive")
    if not np.isfinite(reference) or reference <= 0 or bolus_multiplier < 0:
        raise ValueError("Invalid reference or bolus multiplier")
    width = cfg.control_bin_minutes / 60.0
    n_bins = int(np.ceil(horizon / width))
    u = np.zeros(n_bins) if controls is None else np.asarray(controls, dtype=float)
    if u.shape != (n_bins,) or not np.isfinite(u).all() or (u < 0).any() or (u > cfg.control_max).any():
        raise ValueError("Invalid control vector")
    events = {float(row.time_min / 60): (float(row.eating), float(row.bolus))
              for row in record.itertuples() if row.eating > 0 or row.bolus > 0}
    knots = np.unique(np.r_[0., horizon, list(events), np.arange(1, n_bins) * width])
    beta2 = derive_beta2(p[:5], reference, model)
    a, b0, b1, gamma, delta, meal_gain, bolus_gain = p
    k_abs = cfg.insulin_absorption_per_hour
    state = np.array([0., float(record.glucose.iloc[0]),
                      quick_insulin_equilibrium(reference, p[:5], model), 0.])
    result = np.empty((len(times), 4))
    for index, left in enumerate(knots):
        food, bolus = events.get(float(left), (0., 0.))
        state[0] += meal_gain * food
        state[3] += bolus_gain * bolus * bolus_multiplier
        if index == len(knots) - 1:
            result[np.isclose(times, left, atol=1e-12, rtol=0)] = state
            break
        right = knots[index + 1]
        control = float(u[min(int((left + 1e-10) / width), n_bins - 1)])

        def rhs(t, y):
            digestive, glucose, quick, depot = y
            insulin = model.basal_insulin + max(quick, 0.)
            return [-a * digestive,
                    a * digestive - (b0 + b1 * insulin) * glucose + beta2 / (1. + insulin),
                    gamma * glucose**2 / (model.k_half_mg_dl**2 + glucose**2)
                    - delta * quick + k_abs * depot + control,
                    -k_abs * depot]

        mask = (times >= left) & (times < right)
        evaluation = np.r_[times[mask], right]
        solution = solve_ivp(rhs, (left, right), state, t_eval=evaluation,
                             rtol=cfg.rtol, atol=cfg.atol, max_step=0.1)
        if not solution.success or not np.isfinite(solution.y).all() or (solution.y < -1e-6).any():
            raise RuntimeError("Integration failed or produced a negative state")
        result[mask] = solution.y[:, :-1].T
        state = solution.y[:, -1]
    return result


def fit_record(record, reference, config=None):
    cfg = config or ExperimentConfig()
    if cfg.fit_starts < 1:
        raise ValueError("At least one fit start is required")
    times = record.time_min.to_numpy() / 60.0
    observed = record.glucose.to_numpy()
    lower, upper = np.log(BOUNDS[:, 0]), np.log(BOUNDS[:, 1])
    nominal = np.array([1.5, .05, .03, 1., 1., 4., .3])
    rng = np.random.default_rng(cfg.seed)
    starts = [np.log(nominal)] + [np.clip(np.log(nominal) + rng.normal(0, 1.4, 7),
                                         lower + 1e-8, upper - 1e-8)
                                    for _ in range(cfg.fit_starts - 1)]

    def residual(log_p):
        return (simulate(record, reference, np.exp(log_p), times, cfg)[:, 1] - observed) / 5.

    runs, candidates = [], []
    for number, start in enumerate(starts):
        fit = least_squares(residual, start, bounds=(lower, upper), loss="soft_l1",
                            f_scale=1., max_nfev=cfg.fit_max_nfev, diff_step=1e-4,
                            xtol=1e-7, ftol=1e-7, gtol=1e-7)
        candidates.append(fit)
        runs.append({"start": number + 1, "cost": float(fit.cost), "success": bool(fit.success),
                     "nfev": int(fit.nfev), "status": int(fit.status), "message": fit.message})
        print(f"  fit start {number + 1}: cost={fit.cost:.3f}, success={fit.success}, nfev={fit.nfev}", flush=True)
    selected = min(range(len(candidates)), key=lambda i: candidates[i].cost)
    best = candidates[selected]
    parameters = np.exp(best.x)
    prediction = simulate(record, reference, parameters, times, cfg)[:, 1]
    # Finite-difference data-only Jacobian in log parameter coordinates (not the
    # robust-loss-rescaled optimizer Jacobian). All seven fitted parameters count.
    epsilon = 1e-4
    jacobian = np.column_stack([(residual(best.x + np.eye(7)[j] * epsilon)
                                 - residual(best.x - np.eye(7)[j] * epsilon)) / (2 * epsilon)
                                for j in range(7)])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else float("inf")
    errors = prediction - observed
    variance = float(np.sum((observed - observed.mean())**2))
    row = dict(zip(NAMES, map(float, parameters)))
    row.update({"beta2_derived": derive_beta2(parameters[:5], reference, ModelConfig()),
                "K_fixed_mg_dl": 120., "reference_glucose": reference,
                "n_observations": len(record), "n_fitted_parameters": 7,
                "rmse": float(np.sqrt(np.mean(errors**2))), "mae": float(np.mean(abs(errors))),
                "r2": float(1 - np.sum(errors**2) / variance) if variance > 0 else np.nan,
                "constant_first_BG_rmse": float(np.sqrt(np.mean((observed - observed[0])**2))),
                "optimizer_success": bool(best.success), "selected_start": selected + 1,
                "data_log_jacobian_condition": condition})
    position = (parameters - BOUNDS[:, 0]) / np.ptp(BOUNDS, axis=1)
    diagnostics = pd.DataFrame({"parameter": NAMES, "estimate": parameters,
                                "lower": BOUNDS[:, 0], "upper": BOUNDS[:, 1],
                                "near_lower_0_5pct": position < .005,
                                "near_upper_0_5pct": position > .995})
    return parameters, row, pd.DataFrame(runs), diagnostics


def grid_for(record):
    # Merge in native minute coordinates before conversion: constructing an hour
    # arange can produce near-duplicate times that collapse on CSV round trips.
    horizon_min = float(record.time_min.iloc[-1])
    return np.unique(np.r_[np.arange(0, horizon_min, 1.), horizon_min,
                           record.time_min.to_numpy()]) / 60.


def time_fraction(times, glucose, low=-np.inf, high=np.inf, *, strict=False):
    """Exact duration fraction for a piecewise-linear glucose curve."""
    total = 0.
    for t0, t1, g0, g1 in zip(times[:-1], times[1:], glucose[:-1], glucose[1:]):
        if abs(g1 - g0) < 1e-12:
            fraction = float(low < g0 < high) if strict else float(low <= g0 <= high)
        else:
            crossing = sorted([(low - g0) / (g1 - g0), (high - g0) / (g1 - g0)])
            fraction = max(0., min(1., crossing[1]) - max(0., crossing[0]))
        total += (t1 - t0) * fraction
    return 100. * total / (times[-1] - times[0])


def control_cost(times, glucose, controls, config):
    horizon = float(times[-1])
    width = config.control_bin_minutes / 60.
    durations = np.minimum(width, horizon - np.arange(len(controls)) * width)
    return float(np.trapz((times / horizon) * (glucose - 70.) * (glucose - 110.), times)
                 + config.control_penalty * np.sum(durations * controls**2))


def optimize_control(record, reference, parameters, config=None):
    cfg = config or ExperimentConfig()
    times = grid_for(record)
    width = cfg.control_bin_minutes / 60.
    zeros = np.zeros(int(np.ceil(times[-1] / width)))
    baseline = simulate(record, reference, parameters, times, cfg, zeros)

    def objective(u):
        states = simulate(record, reference, parameters, times, cfg, u)
        return control_cost(times, states[:, 1], u, cfg)

    # Include zero explicitly: do not report an optimizer failure as an improvement.
    candidates = [(objective(zeros), zeros, "zero-control fallback", False)]
    logs = []
    for number, start in enumerate([zeros, np.full(len(zeros), .1)]):
        fit = minimize(objective, start, method="L-BFGS-B",
                       bounds=[(0., cfg.control_max)] * len(zeros),
                       options={"maxiter": cfg.control_maxiter, "ftol": 1e-8, "maxls": 30})
        logs.append({"start": number + 1, "objective": float(fit.fun),
                     "success": bool(fit.success), "nit": int(fit.nit), "message": str(fit.message)})
        if np.isfinite(fit.fun):
            candidates.append((float(fit.fun), fit.x, f"optimizer start {number + 1}", bool(fit.success)))
        print(f"  control start {number + 1}: objective={fit.fun:.3f}, success={fit.success}", flush=True)
    _, optimal, selected, success = min(candidates, key=lambda item: item[0])
    summaries, trajectories = [], []
    for label, controls in [("Recorded boluses only", zeros),
                             ("Recorded boluses + optimized unitless control", optimal)]:
        states = baseline if not np.any(controls) else simulate(record, reference, parameters, times, cfg, controls)
        glucose = states[:, 1]
        durations = np.minimum(width, times[-1] - np.arange(len(controls)) * width)
        summaries.append({"scenario": label, "paper_objective": control_cost(times, glucose, controls, cfg),
                          "mean_glucose": float(np.trapz(glucose, times) / times[-1]),
                          "min_glucose": float(glucose.min()), "max_glucose": float(glucose.max()),
                          "time_70_110_pct": time_fraction(times, glucose, 70, 110),
                          "time_70_180_pct": time_fraction(times, glucose, 70, 180),
                          "time_below_70_pct": time_fraction(times, glucose, high=70, strict=True),
                          "time_above_180_pct": time_fraction(times, glucose, low=180, strict=True),
                          "total_unitless_control": float(np.sum(controls * durations)),
                          "max_unitless_control": float(controls.max()),
                          "selected_control": selected, "optimizer_success": success})
        indices = np.minimum((times / width).astype(int), len(controls) - 1)
        trajectories.append(pd.DataFrame({"scenario": label, "time_min": times * 60,
                                          "glucose": glucose, "quick_insulin": states[:, 2],
                                          "bolus_depot": states[:, 3], "control": controls[indices]}))
    bins = pd.DataFrame({"start_min": np.arange(len(optimal)) * cfg.control_bin_minutes,
                         "end_min": np.minimum(np.arange(1, len(optimal) + 1) * cfg.control_bin_minutes,
                                               times[-1] * 60), "unitless_control": optimal})
    return pd.DataFrame(summaries), pd.concat(trajectories, ignore_index=True), bins, pd.DataFrame(logs)
