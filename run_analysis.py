#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diabetes_paper.control import ControlConfig, optimize_all_controls
from diabetes_paper.data import STATUS_ORDER, load_and_prepare
from diabetes_paper.dataset_description import write_dataset_artifacts
from diabetes_paper.dynamics_exploration import write_exploration_artifacts
from diabetes_paper.estimation import FitConfig, fit_all_participants
from diabetes_paper.fit_evaluation import write_fit_artifacts
from diabetes_paper.model import ModelConfig
from diabetes_paper.reporting import (
    cohort_group_summary,
    group_parameter_summary,
    make_all_figures,
    parameter_group_tests,
    write_all_tables,
    write_analysis_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the CGM model, ablation, and research-only control simulation."
    )
    parser.add_argument(
        "--hall",
        type=Path,
        default=ROOT / "hall2018_full_another_version.xlsx",
    )
    parser.add_argument(
        "--meals", type=Path, default=ROOT / "pbio.2005143.s015.xlsx"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-event-points", type=int, default=24)
    parser.add_argument("--stage1-iterations", type=int, default=12)
    parser.add_argument("--stage1-popsize", type=int, default=5)
    parser.add_argument("--stage2-max-nfev", type=int, default=180)
    parser.add_argument("--max-participants", type=int, default=None)
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--control-maxiter", type=int, default=60)
    return parser.parse_args()


def _meal_group_summary(meal_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "baseline_glucose",
        "peak_glucose",
        "peak_excursion",
        "time_to_peak_min",
        "positive_auc_mg_dl_h",
    ]
    rows: list[dict[str, float | int | str]] = []
    for (status, meal_type), group in meal_metrics.groupby(["status", "meal_type"]):
        row: dict[str, float | int | str] = {
            "status": str(status),
            "meal_type": str(meal_type),
            "events": int(len(group)),
            "participants": int(group["userID"].nunique()),
        }
        for column in columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = float(group[column].std())
            row[f"{column}_median"] = float(group[column].median())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_config = ModelConfig()
    fit_config = FitConfig(
        seed=args.seed,
        stage1_iterations=args.stage1_iterations,
        stage1_popsize=args.stage1_popsize,
        stage2_max_nfev=args.stage2_max_nfev,
        model=model_config,
    )
    control_config = ControlConfig(optimizer_maxiter=args.control_maxiter)

    print("Loading and auditing workbooks...", flush=True)
    prepared = load_and_prepare(
        args.hall, args.meals, min_event_points=args.min_event_points
    )
    write_dataset_artifacts(prepared, args.output)
    fit_points = prepared.meal_points.loc[prepared.meal_points["usable"]].copy()
    if args.max_participants is not None:
        selected: list[str] = []
        per_group = max(1, int(np.ceil(args.max_participants / 3)))
        for status in STATUS_ORDER:
            users = (
                fit_points.loc[fit_points["status"].eq(status), "userID"]
                .drop_duplicates()
                .sort_values()
                .head(per_group)
                .tolist()
            )
            selected.extend(users)
        selected = selected[: args.max_participants]
        fit_points = fit_points.loc[fit_points["userID"].isin(selected)].copy()
        prepared.audit["analysis_participant_subset"] = selected

    print(
        f"Fitting {fit_points['userID'].nunique()} participants with "
        f"{fit_points['event_id'].nunique()} meal events...",
        flush=True,
    )
    participants, event_fits, predictions, comparisons = fit_all_participants(
        fit_points, fit_config, workers=max(1, args.workers)
    )

    if args.skip_control:
        control_summary = pd.DataFrame()
        control_trajectories = pd.DataFrame()
    else:
        print("Running research-only unitless optimal control...", flush=True)
        control_summary, control_trajectories = optimize_all_controls(
            participants, model_config, control_config
        )

    participant_ids = set(participants["userID"])
    analysis_meal_metrics = prepared.meal_metrics.loc[
        prepared.meal_metrics["userID"].isin(participant_ids)
    ].copy()
    audit = dict(prepared.audit)
    audit.update(
        {
            "fit_participants": int(len(participants)),
            "fit_events": int(len(event_fits)),
            "fit_observations": int(len(predictions)),
            "seed": args.seed,
            "workers": args.workers,
            "fit_config": asdict(fit_config),
            "control_config": asdict(control_config),
            "command_cwd": str(Path.cwd()),
            "cpu_count_visible": os.cpu_count(),
        }
    )
    tables = {
        "hall_participant_summary": prepared.hall_participants,
        "cohort_group_summary": cohort_group_summary(prepared.hall_participants),
        "meal_event_qc": prepared.meal_qc,
        "meal_points_prepared": prepared.meal_points,
        "meal_response_metrics": prepared.meal_metrics,
        "meal_response_group_summary": _meal_group_summary(analysis_meal_metrics),
        "participant_parameters": participants,
        "event_fit_metrics": event_fits,
        "model_predictions": predictions,
        "model_comparison": comparisons,
        "group_parameter_summary": group_parameter_summary(participants),
        "group_parameter_tests": parameter_group_tests(participants),
        "control_summary": control_summary,
        "control_trajectories": control_trajectories,
    }
    write_all_tables(args.output, audit, tables)
    fit_evaluation = write_fit_artifacts(args.output, participants, predictions, comparisons)
    write_exploration_artifacts(args.output, participants, predictions, comparisons, model=model_config)
    make_all_figures(
        args.output,
        prepared.hall_participants,
        prepared.meal_points,
        participants,
        predictions,
        control_summary,
        control_trajectories,
    )
    write_analysis_report(
        args.output / "analysis_report.md",
        audit,
        participants,
        comparisons,
        control_summary,
        fit_evaluation=fit_evaluation,
    )
    print(f"Analysis complete: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
