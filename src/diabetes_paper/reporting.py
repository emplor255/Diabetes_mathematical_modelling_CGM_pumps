from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu

from .data import STATUS_ORDER
from .fit_evaluation import evaluate_fits, fit_report_lines
from .dynamics_exploration import exploration_report_lines
from .control_evaluation import (
    control_report_lines,
    evaluate_control_cohort,
    write_control_tables,
)


PARAMETERS = ["alpha", "beta0", "beta1", "beta2_derived", "gamma", "delta"]


def cohort_group_summary(participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for status in STATUS_ORDER:
        group = participants.loc[participants["status"].eq(status)]
        if group.empty:
            continue
        rows.append(
            {
                "status": status,
                "participants": int(len(group)),
                "cgm_rows": int(group["n_cgm"].sum()),
                "age_mean": float(group["age"].mean()),
                "age_sd": float(group["age"].std()),
                "participant_mean_glucose_mean": float(group["mean_glucose"].mean()),
                "participant_mean_glucose_sd": float(group["mean_glucose"].std()),
                "cv_glucose_pct_median": float(group["cv_glucose_pct"].median()),
                "time_70_140_pct_median": float(group["time_70_140_pct"].median()),
                "time_above_140_pct_median": float(group["time_above_140_pct"].median()),
            }
        )
    return pd.DataFrame(rows)


def group_parameter_summary(participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for status in STATUS_ORDER:
        group = participants.loc[participants["status"].eq(status)]
        for parameter in PARAMETERS:
            values = group[parameter].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            rows.append(
                {
                    "status": status,
                    "parameter": parameter,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                }
            )
    return pd.DataFrame(rows)


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def parameter_group_tests(participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parameter in PARAMETERS:
        groups = [
            participants.loc[participants["status"].eq(status), parameter]
            .dropna()
            .to_numpy(dtype=float)
            for status in STATUS_ORDER
        ]
        if all(len(group) > 0 for group in groups):
            statistic, p_value = kruskal(*groups)
            rows.append(
                {
                    "parameter": parameter,
                    "contrast": "Kruskal-Wallis: all 3 groups",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
        healthy = groups[0]
        dysglycemia = np.r_[groups[1], groups[2]]
        if len(healthy) > 0 and len(dysglycemia) > 0:
            statistic, p_value = mannwhitneyu(
                healthy, dysglycemia, alternative="two-sided"
            )
            rows.append(
                {
                    "parameter": parameter,
                    "contrast": "Mann-Whitney: healthy vs dysglycemia",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_bh"] = _benjamini_hochberg(result["p_value"])
    return result


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 180})


def plot_cohort(participants: pd.DataFrame, destination: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(
        data=participants,
        x="status",
        y="mean_glucose",
        order=STATUS_ORDER,
        ax=axes[0],
        color="#8ecae6",
    )
    sns.stripplot(
        data=participants,
        x="status",
        y="mean_glucose",
        order=STATUS_ORDER,
        ax=axes[0],
        color="#023047",
        size=4,
        jitter=0.18,
    )
    axes[0].set(xlabel="", ylabel="Participant mean CGM (mg/dL)")
    sns.boxplot(
        data=participants,
        x="status",
        y="time_above_140_pct",
        order=STATUS_ORDER,
        ax=axes[1],
        color="#ffb703",
    )
    sns.stripplot(
        data=participants,
        x="status",
        y="time_above_140_pct",
        order=STATUS_ORDER,
        ax=axes[1],
        color="#6a3d00",
        size=4,
        jitter=0.18,
    )
    axes[1].set(xlabel="", ylabel="CGM readings >140 mg/dL (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    fig.suptitle("Hall2018 CGM cohort")
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_meal_profiles(points: pd.DataFrame, destination: Path) -> None:
    _style()
    usable = points.loc[points["usable"]].copy()
    participant_curves = (
        usable.groupby(["userID", "status", "meal_type", "relative_min"], as_index=False)[
            "glucose"
        ].mean()
    )
    meal_order = [meal for meal in ["PB", "Bar", "CF"] if meal in set(usable["meal_type"])]
    fig, axes = plt.subplots(1, len(meal_order), figsize=(6 * len(meal_order), 5), sharey=True)
    axes = np.atleast_1d(axes)
    palette = dict(zip(STATUS_ORDER, sns.color_palette("colorblind", len(STATUS_ORDER))))
    for axis, meal in zip(axes, meal_order):
        subset = participant_curves.loc[participant_curves["meal_type"].eq(meal)]
        sns.lineplot(
            data=subset,
            x="relative_min",
            y="glucose",
            hue="status",
            hue_order=STATUS_ORDER,
            estimator="mean",
            errorbar=("ci", 95),
            palette=palette,
            ax=axis,
        )
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set(title=meal, xlabel="Minutes from meal", ylabel="CGM (mg/dL)")
        if axis is not axes[0] and axis.get_legend() is not None:
            axis.get_legend().remove()
    fig.suptitle("Standardized meal responses (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def _representative_users(participants: pd.DataFrame, value: str) -> dict[str, str]:
    selected: dict[str, str] = {}
    for status in STATUS_ORDER:
        group = participants.loc[participants["status"].eq(status)].copy()
        if group.empty:
            continue
        median = group[value].median()
        row = group.iloc[(group[value] - median).abs().argmin()]
        selected[status] = str(row["userID"])
    return selected


def plot_fits(
    participants: pd.DataFrame, predictions: pd.DataFrame, destination: Path
) -> None:
    _style()
    selected = _representative_users(participants, "rmse_proposed")
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 4.5 * len(selected)), squeeze=False)
    colors = {meal: color for meal, color in zip(["PB", "Bar", "CF"], sns.color_palette("Set2", 3))}
    for axis, (status, user) in zip(axes[:, 0], selected.items()):
        subset = predictions.loc[predictions["userID"].eq(user)]
        for event_id, event in subset.groupby("event_id"):
            color = colors.get(str(event["meal_type"].iloc[0]), "gray")
            label = str(event["Meal"].iloc[0])
            axis.scatter(
                event["time_min"], event["observed_glucose"], color=color, s=14, alpha=0.55
            )
            axis.plot(
                event["time_min"], event["proposed_glucose"], color=color, linewidth=1.5, label=label
            )
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), ncol=3, fontsize=9)
        axis.set(
            title=f"{status}: {user}",
            xlabel="Minutes after meal",
            ylabel="CGM (mg/dL)",
        )
    fig.suptitle("Observed standardized meals and proposed-model fits")
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_parameters(participants: pd.DataFrame, destination: Path) -> None:
    _style()
    long = participants.melt(
        id_vars=["userID", "status"],
        value_vars=PARAMETERS,
        var_name="parameter",
        value_name="estimate",
    )
    graph = sns.catplot(
        data=long,
        x="status",
        y="estimate",
        col="parameter",
        col_wrap=3,
        kind="box",
        order=STATUS_ORDER,
        sharey=False,
        height=4,
        aspect=1.2,
        color="#90be6d",
    )
    graph.map_dataframe(
        sns.stripplot,
        x="status",
        y="estimate",
        order=STATUS_ORDER,
        color="#1b4332",
        size=3,
        jitter=0.18,
    )
    graph.set_axis_labels("", "Estimate")
    graph.set_titles("{col_name}")
    for axis in graph.axes.flatten():
        axis.tick_params(axis="x", rotation=20)
    graph.figure.suptitle("Patient-level kinetic estimates", y=1.02)
    graph.savefig(destination, bbox_inches="tight")
    plt.close(graph.figure)


def plot_control(
    participants: pd.DataFrame,
    control_summary: pd.DataFrame,
    trajectories: pd.DataFrame,
    destination: Path,
) -> None:
    if control_summary.empty or trajectories.empty:
        return
    _style()
    no_control = control_summary.loc[
        control_summary["scenario"].eq("No external control")
    ].copy()
    selected = _representative_users(no_control, "peak_glucose")
    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 4 * len(selected)), squeeze=False)
    for row_index, (status, user) in enumerate(selected.items()):
        subset = trajectories.loc[trajectories["userID"].eq(user)]
        glucose_axis, control_axis = axes[row_index]
        for scenario, trajectory in subset.groupby("scenario"):
            glucose_axis.plot(
                trajectory["time_min"], trajectory["glucose"], label=scenario, linewidth=2
            )
        glucose_axis.axhspan(70, 110, color="#90be6d", alpha=0.18)
        glucose_axis.set(
            title=f"{status}: {user}",
            xlabel="Minutes after meal",
            ylabel="Simulated glucose (mg/dL)",
        )
        glucose_axis.legend(fontsize=9)
        optimized = subset.loc[subset["scenario"].eq("Optimized unitless control")]
        control_axis.step(
            optimized["time_min"],
            optimized["control_intensity"],
            where="post",
            color="#d00000",
        )
        control_axis.set(
            title="Optimized control (not insulin units)",
            xlabel="Minutes after meal",
            ylabel="Unitless intensity",
        )
    fig.suptitle("Research-only optimal-control simulation")
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def write_all_tables(
    output_dir: Path,
    audit: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=str)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)


def make_all_figures(
    output_dir: Path,
    hall_participants: pd.DataFrame,
    meal_points: pd.DataFrame,
    fitted_participants: pd.DataFrame,
    predictions: pd.DataFrame,
    control_summary: pd.DataFrame,
    control_trajectories: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_cohort(hall_participants, figure_dir / "figure_1_cohort.png")
    plot_meal_profiles(meal_points, figure_dir / "figure_2_meal_profiles.png")
    plot_fits(fitted_participants, predictions, figure_dir / "figure_3_model_fits.png")
    plot_parameters(fitted_participants, figure_dir / "figure_4_parameters.png")
    plot_control(
        fitted_participants,
        control_summary,
        control_trajectories,
        figure_dir / "figure_5_optimal_control.png",
    )


def write_analysis_report(
    destination: Path,
    audit: dict[str, Any],
    participants: pd.DataFrame,
    comparisons: pd.DataFrame,
    control_summary: pd.DataFrame,
    fit_evaluation: dict[str, pd.DataFrame] | None = None,
) -> None:
    # Preserve the new fit section when regenerating a saved run's control report.
    prediction_path = destination.parent / "tables" / "model_predictions.csv"
    if fit_evaluation is None and prediction_path.is_file():
        fit_evaluation = evaluate_fits(participants, pd.read_csv(prediction_path), comparisons)
    proposed = comparisons.loc[comparisons["model"].eq("Proposed 3-state")]
    reduced = comparisons.loc[comparisons["model"].eq("Reduced 2-state")]
    comparison_wide = comparisons.pivot(index="userID", columns="model", values="bic")
    proposed_bic_wins = float(
        100.0
        * (
            comparison_wide["Proposed 3-state"]
            < comparison_wide["Reduced 2-state"]
        ).mean()
    )
    poor_identifiability = int(
        (participants["jacobian_condition_number"] > 100.0).sum()
    )
    beta0_near_lower = int(participants["beta0"].le(0.0100995).sum())
    gamma_near_upper = int(participants["gamma"].ge(9.950005).sum())
    lines = [
        "# Reproduction report",
        "",
        "## Data used",
        "",
        f"- Hall2018: {audit['hall_rows']:,} CGM rows from {audit['hall_participants']} participants "
        f"({audit['hall_healthy_participants']} healthy, {audit['hall_prediabetes_participants']} prediabetes, "
        f"{audit['hall_type2_participants']} type 2 diabetes).",
        f"- Standardized meals: {audit['meal_input_rows']:,} input rows, "
        f"{audit['meal_events_total']} event windows, and {audit['meal_events_usable']} retained full-span windows (internal gaps may remain).",
        f"- QC: {audit['meal_duplicate_rows_collapsed']} duplicate timestamp rows collapsed by their mean; "
        f"{audit['meal_censored_low_rows']} `Low` readings approximated by 40 mg/dL with weight 0.8 (not exact measured concentrations).",
        "",
        "Dataset coverage: [overview figure](figures/figure_6_dataset_overview.png), "
        "[cohort counts](tables/dataset_overview.csv), and "
        "[pointwise Figure 2 sample sizes](tables/meal_profile_sample_sizes.csv). "
        "The meal cohort is a nested subset of Hall2018, not an independent validation set. "
        "Figure 2 shows observed participant-averaged glucose and pointwise mean confidence intervals, "
        "not model fits, individual-response bands, or controller performance.",
        "",
        "## Parameter estimation and ablation",
        "",
        f"- Fitted participants: {len(participants)}.",
        f"- Proposed-model median participant RMSE: {proposed['rmse'].median():.2f} mg/dL.",
        f"- Reduced two-state median participant RMSE: {reduced['rmse'].median():.2f} mg/dL.",
        f"- The proposed model had lower BIC for {proposed_bic_wins:.1f}% of participants.",
        "- BIC therefore does not support a general advantage of the proposed model over the reduced ablation."
        if proposed_bic_wins <= 50.0
        else "- BIC favors the proposed model for a majority, but not all, participants.",
        f"- {int(participants['kinetic_bound_hits'].gt(0).sum())}/{len(participants)} fits had at least one kinetic parameter at a bound.",
        f"- Specifically, beta0 was near its lower bound in {beta0_near_lower}/{len(participants)} fits and gamma was near its upper bound in {gamma_near_upper}/{len(participants)} fits.",
        f"- {poor_identifiability}/{len(participants)} fits had normalized kinetic-Jacobian condition number >100 or infinite.",
        "",
        "The last two diagnostics are essential: CGM alone does not identify hidden insulin dynamics reliably. "
        "Group-level parameter differences in the CSV tables are exploratory, not mechanistic proof.",
        "",
    ]
    if fit_evaluation is not None:
        lines.extend(fit_report_lines(fit_evaluation))
    lines.extend(["## Optimal control", ""])
    if control_summary.empty:
        lines.append("Optimal-control simulation was skipped.")
    else:
        pivot = control_summary.pivot(index="userID", columns="scenario", values="paper_objective")
        improvement = (
            pivot["No external control"] - pivot["Optimized unitless control"]
        )
        controlled = control_summary.loc[
            control_summary["scenario"].eq("Optimized unitless control")
        ]
        saturated = int(controlled["time_at_control_upper_bound_pct"].gt(0).sum())
        lines.extend(
            [
                f"- Median reduction in the paper's objective: {improvement.median():.2f}.",
                f"- Median integrated control intensity: {controlled['total_control_intensity'].median():.3f} (unitless).",
                f"- {saturated}/{len(controlled)} optimized profiles reached the imposed upper control bound for at least one integration step.",
                "",
                "These are mathematical control intensities only. The source files contain neither insulin concentrations "
                "nor administered doses, so the result cannot be converted to units of insulin or used for treatment.",
            ]
        )
        evaluation = evaluate_control_cohort(
            control_summary, participants, seed=int(audit.get("seed", 20260815))
        )
        write_control_tables(destination.parent, evaluation, audit)
        lines.extend(["", *control_report_lines(evaluation, audit)])
    lines.extend(
        [
            "",
            "## Reproducibility notes",
            "",
            "- Meal time is defined as 30 minutes after the beginning of a full-span 180-minute S6 Data window (supplement file identifier .s015).",
            "- Only full-span windows with at least the configured minimum number of unique CGM points are fitted; internal gaps may remain.",
            "- beta2 is derived from the participant's pre-meal equilibrium; K=120 mg/dL and basal insulin=1 are fixed because insulin is unobserved.",
            "- The printed model's required absorption delay is implemented as an event-level fitted nuisance parameter.",
            "- Random seeds, bounds, optimizer settings, event QC, predictions, and control trajectories are exported under `tables/`.",
        ]
    )
    sweep_config_path = destination.parent / "tables" / "parameter_sweep_config.json"
    sweep_summary_path = destination.parent / "tables" / "parameter_sweep_summary.csv"
    if sweep_config_path.is_file() and sweep_summary_path.is_file():
        lines.extend(["", *exploration_report_lines(
            pd.read_csv(sweep_summary_path), json.loads(sweep_config_path.read_text(encoding="utf-8"))
        )])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
