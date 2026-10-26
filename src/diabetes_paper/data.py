from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TYPE_LABELS = {0.0: "Healthy", 0.5: "Prediabetes", 2.0: "Type 2 diabetes"}
STATUS_ORDER = ["Healthy", "Prediabetes", "Type 2 diabetes"]


@dataclass(frozen=True)
class PreparedData:
    hall: pd.DataFrame
    hall_participants: pd.DataFrame
    meal_points: pd.DataFrame
    meal_qc: pd.DataFrame
    meal_metrics: pd.DataFrame
    audit: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_hall(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(path)
    frame = pd.read_excel(source)
    _require_columns(frame, {"id", "time", "gl", "age", "type"}, source)

    frame = frame.copy()
    frame["id"] = frame["id"].astype(str).str.strip()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["gl"] = pd.to_numeric(frame["gl"], errors="coerce")
    frame["type_code"] = pd.to_numeric(frame["type"], errors="coerce")
    frame["status"] = frame["type_code"].map(TYPE_LABELS)

    invalid = frame[["id", "time", "gl", "status"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"{source} contains {int(invalid.sum())} invalid required rows")

    inconsistent = frame.groupby("id")["type_code"].nunique().gt(1)
    if inconsistent.any():
        bad = inconsistent[inconsistent].index.tolist()
        raise ValueError(f"Participants have inconsistent type labels: {bad}")

    grouped = frame.groupby("id", sort=True)
    participants = grouped.agg(
        type_code=("type_code", "first"),
        status=("status", "first"),
        age=("age", "first"),
        n_cgm=("gl", "size"),
        start=("time", "min"),
        end=("time", "max"),
        mean_glucose=("gl", "mean"),
        sd_glucose=("gl", "std"),
        median_glucose=("gl", "median"),
        min_glucose=("gl", "min"),
        max_glucose=("gl", "max"),
    ).reset_index()
    participants["cv_glucose_pct"] = (
        100.0 * participants["sd_glucose"] / participants["mean_glucose"]
    )
    participants["observed_days"] = (
        participants["end"] - participants["start"]
    ).dt.total_seconds() / 86400.0

    thresholds = {
        "time_below_70_pct": frame["gl"].lt(70),
        "time_70_110_pct": frame["gl"].between(70, 110, inclusive="both"),
        "time_70_140_pct": frame["gl"].between(70, 140, inclusive="both"),
        "time_above_140_pct": frame["gl"].gt(140),
        "time_above_180_pct": frame["gl"].gt(180),
    }
    for column, indicator in thresholds.items():
        values = indicator.groupby(frame["id"]).mean().mul(100.0)
        participants[column] = participants["id"].map(values)

    return frame, participants


def prepare_meals(
    path: str | Path,
    participant_metadata: pd.DataFrame,
    min_event_points: int = 24,
    expected_window_minutes: float = 180.0,
    duration_tolerance_minutes: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = Path(path)
    raw = pd.read_excel(source)
    _require_columns(raw, {"Meal", "userID", "time", "GlucoseValue"}, source)

    raw = raw.copy()
    raw["userID"] = raw["userID"].astype(str).str.strip()
    raw["Meal"] = raw["Meal"].astype(str).str.strip()
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    glucose_text = raw["GlucoseValue"].astype(str).str.strip()
    raw["censored_low"] = glucose_text.str.casefold().eq("low")
    raw["glucose"] = pd.to_numeric(raw["GlucoseValue"], errors="coerce")
    raw.loc[raw["censored_low"], "glucose"] = 40.0
    invalid_required = raw[["userID", "Meal", "time"]].isna().any(axis=1)
    invalid_glucose = raw["glucose"].isna()

    audit: dict[str, Any] = {
        "meal_min_event_points": int(min_event_points),
        "meal_expected_window_minutes": float(expected_window_minutes),
        "meal_duration_tolerance_minutes": float(duration_tolerance_minutes),
        "meal_input_rows": int(len(raw)),
        "meal_censored_low_rows": int(raw["censored_low"].sum()),
        "meal_invalid_required_rows": int(invalid_required.sum()),
        "meal_unrecognized_glucose_rows": int((invalid_glucose & ~raw["censored_low"]).sum()),
    }
    raw = raw.loc[~invalid_required & ~invalid_glucose].copy()

    collapsed = (
        raw.groupby(["userID", "Meal", "time"], as_index=False, sort=True)
        .agg(
            glucose=("glucose", "mean"),
            source_rows=("glucose", "size"),
            censored_low=("censored_low", "max"),
        )
        .sort_values(["userID", "Meal", "time"])
    )
    audit["meal_duplicate_rows_collapsed"] = int(len(raw) - len(collapsed))

    metadata = participant_metadata.set_index("id")[["type_code", "status", "age"]]
    point_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []
    for (user_id, meal), event in collapsed.groupby(["userID", "Meal"], sort=True):
        event = event.sort_values("time").copy()
        start = event["time"].iloc[0]
        end = event["time"].iloc[-1]
        duration = float((end - start).total_seconds() / 60.0)
        unique_points = int(len(event))
        duration_ok = abs(duration - expected_window_minutes) <= duration_tolerance_minutes
        points_ok = unique_points >= min_event_points
        has_metadata = user_id in metadata.index
        usable = bool(duration_ok and points_ok and has_metadata)
        reasons: list[str] = []
        if not duration_ok:
            reasons.append("incomplete_180_min_window")
        if not points_ok:
            reasons.append("too_few_unique_points")
        if not has_metadata:
            reasons.append("missing_hall_metadata")

        event_id = f"{user_id}__{meal.replace(' ', '_')}"
        qc_rows.append(
            {
                "event_id": event_id,
                "userID": user_id,
                "Meal": meal,
                "unique_points": unique_points,
                "source_rows": int(event["source_rows"].sum()),
                "duplicate_rows": int(event["source_rows"].sum() - unique_points),
                "censored_low_points": int(event["censored_low"].sum()),
                "start": start,
                "end": end,
                "duration_min": duration,
                "usable": usable,
                "excluded_reason": ";".join(reasons),
            }
        )

        # Only complete windows have an unambiguous meal time: window start + 30 min.
        event["event_id"] = event_id
        event["relative_min"] = (event["time"] - start).dt.total_seconds() / 60.0 - 30.0
        event["weight"] = np.where(
            event["source_rows"].gt(1) | event["censored_low"], 0.8, 1.0
        )
        event["usable"] = usable
        event["meal_type"] = meal.split()[0]
        try:
            event["replicate"] = int(meal.split()[-1])
        except (ValueError, IndexError):
            event["replicate"] = np.nan
        if has_metadata:
            event["type_code"] = float(metadata.loc[user_id, "type_code"])
            event["status"] = str(metadata.loc[user_id, "status"])
            event["age"] = float(metadata.loc[user_id, "age"])
        else:
            event["type_code"] = np.nan
            event["status"] = np.nan
            event["age"] = np.nan
        point_frames.append(event)

    points = pd.concat(point_frames, ignore_index=True)
    qc = pd.DataFrame(qc_rows)
    audit.update(
        {
            "meal_unique_points": int(len(collapsed)),
            "meal_events_total": int(len(qc)),
            "meal_events_usable": int(qc["usable"].sum()),
            "meal_events_excluded": int((~qc["usable"]).sum()),
            "meal_participants_total": int(qc["userID"].nunique()),
            "meal_participants_with_usable_event": int(
                qc.loc[qc["usable"], "userID"].nunique()
            ),
        }
    )
    return points, qc, audit


def meal_response_metrics(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    usable = points.loc[points["usable"]].copy()
    for event_id, event in usable.groupby("event_id", sort=True):
        event = event.sort_values("relative_min")
        pre = event.loc[event["relative_min"].lt(0)]
        post = event.loc[event["relative_min"].ge(0)]
        if pre.empty or post.empty:
            continue
        baseline = float(pre["glucose"].median())
        peak_index = post["glucose"].idxmax()
        t_hours = post["relative_min"].to_numpy(dtype=float) / 60.0
        excursion = post["glucose"].to_numpy(dtype=float) - baseline
        rows.append(
            {
                "event_id": event_id,
                "userID": event["userID"].iloc[0],
                "Meal": event["Meal"].iloc[0],
                "meal_type": event["meal_type"].iloc[0],
                "replicate": event["replicate"].iloc[0],
                "type_code": event["type_code"].iloc[0],
                "status": event["status"].iloc[0],
                "baseline_glucose": baseline,
                "meal_time_glucose": float(
                    np.interp(
                        0.0,
                        event["relative_min"].to_numpy(dtype=float),
                        event["glucose"].to_numpy(dtype=float),
                    )
                ),
                "peak_glucose": float(post.loc[peak_index, "glucose"]),
                "peak_excursion": float(post.loc[peak_index, "glucose"] - baseline),
                "time_to_peak_min": float(post.loc[peak_index, "relative_min"]),
                "auc_excursion_mg_dl_h": float(np.trapz(excursion, t_hours)),
                "positive_auc_mg_dl_h": float(np.trapz(np.maximum(excursion, 0.0), t_hours)),
                "postmeal_points": int(len(post)),
            }
        )
    return pd.DataFrame(rows)


def load_and_prepare(
    hall_path: str | Path,
    meal_path: str | Path,
    min_event_points: int = 24,
) -> PreparedData:
    hall, participants = load_hall(hall_path)
    points, qc, meal_audit = prepare_meals(
        meal_path, participants, min_event_points=min_event_points
    )
    metrics = meal_response_metrics(points)
    audit = {
        "hall_rows": int(len(hall)),
        "hall_participants": int(participants["id"].nunique()),
        "hall_healthy_participants": int(participants["type_code"].eq(0.0).sum()),
        "hall_prediabetes_participants": int(participants["type_code"].eq(0.5).sum()),
        "hall_type2_participants": int(participants["type_code"].eq(2.0).sum()),
        "hall_sex_nonmissing_rows": int(hall["sex"].notna().sum()) if "sex" in hall else 0,
        "hall_insulin_modality_nonmissing_rows": int(
            hall["insulinModality"].notna().sum()
        )
        if "insulinModality" in hall
        else 0,
        **meal_audit,
    }
    return PreparedData(hall, participants, points, qc, metrics, audit)
