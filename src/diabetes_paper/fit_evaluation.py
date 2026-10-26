"""Honest, reproducible presentation of saved in-sample fits; never refits data."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import STATUS_ORDER
from .estimation import KINETIC_BOUNDS
from .model import PARAMETER_NAMES


MODELS = {"Proposed 3-state": "proposed_glucose", "Reduced 2-state": "reduced_glucose"}
PARAMETERS = ["alpha", "beta0", "beta1", "beta2_derived", "gamma", "delta"]
SYMBOLS = {"alpha": r"$\alpha$", "beta0": r"$\beta_0$", "beta1": r"$\beta_1$",
           "beta2_derived": r"$\beta_2$ (derived)", "gamma": r"$\gamma$", "delta": r"$\delta$"}
COLORS = dict(zip(STATUS_ORDER, ["#0173b2", "#de8f05", "#029e73"]))
BOUND_FRACTION = 0.005  # Same 0.5%-of-search-range rule as the fitting pipeline.


def _validate(participants: pd.DataFrame, predictions: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    if participants.empty or predictions.empty:
        raise ValueError("Fit evaluation requires nonempty participant and prediction tables")
    if participants.userID.duplicated().any():
        raise ValueError("Duplicate participant IDs")
    if predictions.duplicated(["event_id", "time_min"]).any():
        raise ValueError("Duplicate event/timestamp predictions")
    if set(participants.userID) != set(predictions.userID):
        raise ValueError("Predictions must cover every fitted participant exactly")
    if not set(participants.status).issubset(STATUS_ORDER):
        raise ValueError("Unknown clinical-status group")
    expected = predictions.userID.map(participants.set_index("userID").status)
    if not predictions.status.eq(expected).all():
        raise ValueError("Participant and prediction status labels disagree")
    values = predictions[["time_min", "observed_glucose", *MODELS.values()]].to_numpy(float)
    if not np.isfinite(values).all() or predictions.time_min.lt(0).any():
        raise ValueError("Nonfinite predictions or negative fitting times")
    if not np.isfinite(participants[PARAMETERS].to_numpy(float)).all():
        raise ValueError("Nonfinite parameter estimates")
    if predictions.groupby("event_id")[["userID", "status", "Meal", "meal_type"]].nunique().gt(1).any().any():
        raise ValueError("Event metadata are inconsistent")
    if comparisons.duplicated(["userID", "model"]).any():
        raise ValueError("Duplicate model-comparison rows")
    expected_pairs = {(user, model) for user in participants.userID for model in MODELS}
    if set(zip(comparisons.userID, comparisons.model)) != expected_pairs:
        raise ValueError("Model comparisons must contain both models for every participant")
    if not comparisons.status.eq(comparisons.userID.map(participants.set_index("userID").status)).all():
        raise ValueError("Comparison status labels disagree")
    if not np.isfinite(comparisons[["sse", "aic", "bic"]].to_numpy(float)).all():
        raise ValueError("Nonfinite model-comparison metrics")


def _error_metrics(frame: pd.DataFrame, column: str) -> dict:
    observed = frame.observed_glucose.to_numpy(float)
    error = frame[column].to_numpy(float) - observed
    sse = float(np.sum(error ** 2))
    sst = float(np.sum((observed - observed.mean()) ** 2))
    return {"n_points": len(frame), "sse": sse, "sae": float(np.abs(error).sum()),
            "sst": sst, "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(error.mean()),
            "r2": 1.0 - sse / sst if sst > 0 else np.nan}


def _distribution(values: pd.Series, prefix: str) -> dict:
    values = values.dropna()
    if values.empty:
        return {f"{prefix}_{suffix}": np.nan for suffix in ["mean", "sd", "median", "q1", "q3"]}
    return {f"{prefix}_mean": values.mean(), f"{prefix}_sd": values.std(),
            f"{prefix}_median": values.median(), f"{prefix}_q1": values.quantile(0.25),
            f"{prefix}_q3": values.quantile(0.75)}


def parameter_comparison(participants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bounds = dict(zip(PARAMETER_NAMES, KINETIC_BOUNDS))
    for status in STATUS_ORDER:
        group = participants.loc[participants.status.eq(status)]
        if group.empty:
            continue
        for parameter in PARAMETERS:
            values = group[parameter]
            row = {"status": status, "parameter": parameter, "n": len(group),
                   **_distribution(values, "estimate"), "is_derived": parameter == "beta2_derived",
                   "lower_bound": np.nan, "upper_bound": np.nan,
                   "n_near_lower": np.nan, "n_near_upper": np.nan, "n_near_either": np.nan}
            if parameter in bounds:
                lower, upper = bounds[parameter]
                position = (values - lower) / (upper - lower)
                row.update(lower_bound=lower, upper_bound=upper,
                           n_near_lower=int(position.lt(BOUND_FRACTION).sum()),
                           n_near_upper=int(position.gt(1 - BOUND_FRACTION).sum()),
                           n_near_either=int((position.lt(BOUND_FRACTION) | position.gt(1 - BOUND_FRACTION)).sum()))
            rows.append(row)
    summary = pd.DataFrame(rows)
    contrasts = []
    for parameter in PARAMETERS:
        healthy = participants.loc[participants.status.eq("Healthy"), parameter]
        diabetes = participants.loc[participants.status.eq("Type 2 diabetes"), parameter]
        healthy_median = healthy.median() if len(healthy) else np.nan
        diabetes_median = diabetes.median() if len(diabetes) else np.nan
        contrasts.append({"parameter": parameter, "n_healthy": len(healthy), "n_type2": len(diabetes),
                          "healthy_median": healthy_median, "type2_median": diabetes_median,
                          "type2_minus_healthy_median": diabetes_median - healthy_median,
                          "type2_minus_healthy_mean": diabetes.mean() - healthy.mean()})
    return summary, pd.DataFrame(contrasts)


def select_fit_examples(events: pd.DataFrame) -> pd.DataFrame:
    """Highest event R² in each group; never claim these are representative cases."""
    selected = []
    for status in STATUS_ORDER:
        group = events.loc[events.status.eq(status) & events.model.eq("Proposed 3-state")]
        if group.empty:
            continue
        ranked = group.sort_values(["r2", "rmse", "event_id"], ascending=[False, True, True], na_position="last")
        row = ranked.iloc[0].to_dict()
        row["selection_rule"] = "Maximum event R2; ties by minimum RMSE then event ID"
        if group.r2.isna().all():
            row["selection_rule"] = "All event R2 undefined: minimum RMSE then event ID"
        row["candidate_events"] = len(group)
        selected.append(row)
    return pd.DataFrame(selected)


def evaluate_fits(
    participants: pd.DataFrame, predictions: pd.DataFrame, comparisons: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    _validate(participants, predictions, comparisons)
    event_rows, person_rows = [], []
    for model, column in MODELS.items():
        for event_id, frame in predictions.groupby("event_id", sort=True):
            event_rows.append({"model": model, "event_id": event_id,
                               **frame.iloc[0][["userID", "status", "Meal", "meal_type"]].to_dict(),
                               **_error_metrics(frame, column)})
        for user, frame in predictions.groupby("userID", sort=True):
            person_rows.append({"model": model, "userID": user, "status": frame.status.iloc[0],
                                "n_events": frame.event_id.nunique(), **_error_metrics(frame, column)})
    events = pd.DataFrame(event_rows)
    persons = pd.DataFrame(person_rows)
    persons = persons.merge(comparisons[["userID", "model", "sse", "aic", "bic", "optimizer_success"]],
                            on=["userID", "model"], suffixes=("", "_saved"), validate="one_to_one")
    if not np.allclose(persons.sse, persons.sse_saved, rtol=1e-8, atol=1e-8):
        raise ValueError("Saved SSE disagrees with recomputed predictions; input files may be from different runs")
    metadata = participants.set_index("userID")
    for column, source in [("n_points", "n_observations"), ("n_events", "n_events")]:
        if not persons[column].eq(persons.userID.map(metadata[source])).all():
            raise ValueError("Saved observation/event counts disagree with predictions")
    for model, source in [("Proposed 3-state", "rmse_proposed"), ("Reduced 2-state", "rmse_reduced")]:
        if source in metadata:
            check = persons.loc[persons.model.eq(model)]
            if not np.allclose(check.rmse, check.userID.map(metadata[source]), rtol=1e-8, atol=1e-8):
                raise ValueError("Saved participant RMSE disagrees with predictions")
    persons = persons.drop(columns="sse_saved")
    paired = persons.pivot(index=["userID", "status"], columns="model", values=["rmse", "aic", "bic"])
    changes = pd.DataFrame(index=paired.index)
    for metric in ["rmse", "aic", "bic"]:
        changes[f"{metric}_proposed_minus_reduced"] = paired[(metric, "Proposed 3-state")] - paired[(metric, "Reduced 2-state")]
    changes = changes.reset_index()

    cohort = []
    for status in ["All participants", *STATUS_ORDER]:
        group_ids = participants.userID if status == "All participants" else participants.loc[participants.status.eq(status), "userID"]
        if len(group_ids) == 0:
            continue
        for model, column in MODELS.items():
            points = predictions.loc[predictions.userID.isin(group_ids)]
            people = persons.loc[persons.userID.isin(group_ids) & persons.model.eq(model)]
            meals = events.loc[events.userID.isin(group_ids) & events.model.eq(model)]
            error = _error_metrics(points, column)
            within_sst = meals.sst.sum()
            row = {"group": status, "model": model, "participants": len(people), "events": len(meals),
                   "observations": len(points), "pooled_rmse": error["rmse"], "pooled_mae": error["mae"],
                   "pooled_bias": error["bias"], "pooled_r2_global_mean": error["r2"],
                   "r2_within_event": 1 - meals.sse.sum() / within_sst if within_sst > 0 else np.nan,
                   "events_r2_negative": int(meals.r2.lt(0).sum()), "events_r2_undefined": int(meals.r2.isna().sum()),
                   "optimizer_successes": int(people.optimizer_success.eq(True).sum()),
                   **_distribution(people.rmse, "participant_rmse"),
                   **_distribution(people.mae, "participant_mae"),
                   **_distribution(meals.rmse, "event_rmse"), **_distribution(meals.r2, "event_r2")}
            strictly_post = points.loc[points.time_min.gt(0)]
            if not strictly_post.empty:
                post_error = _error_metrics(strictly_post, column)
                row.update(strictly_postmeal_observations=len(strictly_post),
                           strictly_postmeal_rmse=post_error["rmse"], strictly_postmeal_mae=post_error["mae"])
            delta = changes.loc[changes.userID.isin(group_ids)]
            for metric in ["rmse", "aic", "bic"]:
                difference = delta[f"{metric}_proposed_minus_reduced"]
                row[f"participants_lower_{metric}"] = int((difference.lt(-1e-9) if model == "Proposed 3-state" else difference.gt(1e-9)).sum())
            cohort.append(row)
    parameters, contrasts = parameter_comparison(participants)
    return {"cohort": pd.DataFrame(cohort), "participants": persons, "events": events,
            "paired": changes, "parameters": parameters, "contrasts": contrasts,
            "selected": select_fit_examples(events)}


def _number(value: float, precision: int = 2) -> str:
    return f"{value:.{precision}f}" if np.isfinite(value) else "--"


def _iqr(row: pd.Series, prefix: str, precision: int = 2) -> str:
    return (f"{_number(row[prefix + '_median'], precision)} "
            f"[{_number(row[prefix + '_q1'], precision)}, {_number(row[prefix + '_q3'], precision)}]")


def fit_table_display(evaluation: dict[str, pd.DataFrame]) -> pd.DataFrame:
    overall = evaluation["cohort"].query("group == 'All participants'").set_index("model")
    specifications = [
        ("Participants", "participants", "count"), ("Meal windows", "events", "count"),
        ("CGM observations", "observations", "count"),
        ("Pooled RMSE (mg/dL)", "pooled_rmse", "number"),
        ("Pooled MAE (mg/dL)", "pooled_mae", "number"),
        ("Participant RMSE, median [Q1, Q3]", "participant_rmse", "iqr"),
        ("Participant MAE, median [Q1, Q3]", "participant_mae", "iqr"),
        ("Within-event R2", "r2_within_event", "r2"),
        ("Event R2, median [Q1, Q3]", "event_r2", "iqr_r2"),
        ("Events with R2 < 0", "events_r2_negative", "events_count"),
        ("Lower participant RMSE", "participants_lower_rmse", "people_count"),
        ("Lower nominal AIC", "participants_lower_aic", "people_count"),
        ("Lower nominal BIC", "participants_lower_bic", "people_count"),
    ]
    rows = []
    for label, metric, format_type in specifications:
        row = {"Outcome": label}
        for model in MODELS:
            s = overall.loc[model]
            if format_type.startswith("iqr"):
                value = _iqr(s, metric, 3 if format_type == "iqr_r2" else 2)
            elif format_type.endswith("_count"):
                denominator = s.events if format_type == "events_count" else s.participants
                value = f"{int(s[metric])}/{int(denominator)}"
            elif format_type == "count":
                value = f"{int(s[metric]):,}"
            else:
                value = _number(s[metric], 3 if format_type == "r2" else 2)
            row[model] = value
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_table_display(evaluation: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = evaluation["parameters"].set_index(["parameter", "status"])
    contrast = evaluation["contrasts"].set_index("parameter")
    rows = []
    for parameter in PARAMETERS:
        row = {"Parameter": parameter}
        for status in STATUS_ORDER:
            if (parameter, status) not in summary.index:
                row[status] = "--"
                continue
            values = summary.loc[(parameter, status)]
            # Significant digits preserve the small beta0 values instead of rounding to zero.
            row[status] = f"{values.estimate_median:.3g} [{values.estimate_q1:.3g}, {values.estimate_q3:.3g}]"
        delta = contrast.loc[parameter, "type2_minus_healthy_median"]
        row["T2D - Healthy median"] = f"{delta:+.3g}" if np.isfinite(delta) else "--"
        rows.append(row)
    return pd.DataFrame(rows)


def bound_table_display(evaluation: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = evaluation["parameters"].set_index(["parameter", "status"])
    bounds = dict(zip(PARAMETER_NAMES, KINETIC_BOUNDS))
    rows = []
    for parameter in PARAMETERS:
        row = {"Parameter": parameter, "Search interval":
               f"[{bounds[parameter][0]:g}, {bounds[parameter][1]:g}]" if parameter in bounds else "Derived"}
        for status in STATUS_ORDER:
            if (parameter, status) not in summary.index or parameter not in bounds:
                row[status] = "--"
                continue
            values = summary.loc[(parameter, status)]
            row[status] = f"{int(values.n_near_lower)}/{int(values.n_near_upper)} of {int(values.n)}"
        rows.append(row)
    return pd.DataFrame(rows)


def _write_tex(table: pd.DataFrame, path: Path, caption: str, label: str, note: str) -> None:
    def latex(text: str) -> str:
        text = str(text)
        if text in SYMBOLS:
            return SYMBOLS[text]
        return text.replace("R2", r"$R^2$").replace("<", r"$<$").replace("&", r"\&").replace("%", r"\%")
    lines = [r"% Requires booktabs and graphicx.", r"\begin{table}[htbp]", r"\centering",
             "\\caption{" + caption + "}", "\\label{" + label + "}", r"\small",
             r"\resizebox{\linewidth}{!}{%", "\\begin{tabular}{l" + "r" * (len(table.columns) - 1) + "}",
             r"\toprule", " & ".join(latex(c) for c in table.columns) + r" \\", r"\midrule"]
    lines.extend(" & ".join(latex(v) for v in row) + r" \\" for row in table.itertuples(index=False, name=None))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\par\smallskip", r"\begin{minipage}{\linewidth}\footnotesize",
              note, r"\end{minipage}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


FIT_NOTE = (
    r"All fitted participants and meal windows are included, with no outcome-based exclusions. "
    r"Errors are unweighted and in-sample at the observed times ($t\geq0$); the meal-time "
    r"observation also informs model initialization. Pooled RMSE and MAE weight observations equally; "
    r"participant summaries weight people equally. Within-event $R^2=1-\sum_e\mathrm{SSE}_e/"
    r"\sum_e\sum_t(G_{et}-\bar G_e)^2$, which removes between-event baseline differences. "
    r"Event summaries include negative $R^2$ values; constant-glucose events have undefined $R^2$. "
    r"Q1 and Q3 are quartiles, not confidence bounds. Lower-metric counts compare the two models "
    r"within each participant. AIC/BIC are nominal residual-based criteria from the original fit, "
    r"without a serial-correlation correction. These are training fits, not held-out validation."
)
PARAMETER_NOTE = (
    r"Entries are participant-level medians [Q1, Q3], not confidence intervals. "
    r"The final column is the T2D median minus the Healthy median; prediabetes is not pooled "
    r"with T2D. $\beta_2$ is derived from the equilibrium constraint, not independently fitted. "
    r"Time is measured in hours; $\alpha,\beta_0,\delta$ have units h$^{-1}$; "
    r"$\beta_1$ has units (model insulin unit $\times$ h)$^{-1}$; "
    r"$\gamma$ has units model insulin unit/h; and $\beta_2$ has units mg/dL/h under the "
    r"dimensionless-insulin convention. Insulin is not calibrated to clinical units. "
    r"Small subgroup sizes and frequent boundary estimates limit interpretation; see the "
    r"boundary-diagnostic table. Differences are descriptive, not hypothesis tests or proof "
    r"of physiological differences; no p-values are claimed."
)


def plot_selected_fits(predictions: pd.DataFrame, selected: pd.DataFrame, destination: Path) -> None:
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        n = len(selected)
        fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 6.1), squeeze=False,
                                 gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
        residual_limit = 1.0
        for j, row in enumerate(selected.itertuples()):
            points = predictions.loc[predictions.event_id.eq(row.event_id)].sort_values("time_min")
            ax, residual_ax = axes[:, j]
            ax.plot(points.time_min, points.reduced_glucose, color="#9b9b9b", linestyle="--", linewidth=1.5,
                    label="Reduced model", zorder=1)
            ax.plot(points.time_min, points.proposed_glucose, color="#176a9a", linewidth=2.2,
                    label="Proposed model", zorder=2)
            ax.scatter(points.time_min, points.observed_glucose, color="#252525", s=23, edgecolor="white",
                       linewidth=0.45, label="Observed CGM", zorder=3)
            ax.set_title(f"{row.status}\n{row.userID} · {row.Meal}", fontweight="bold", fontsize=11, pad=12)
            ax.text(0.98, 0.96, f"RMSE = {row.rmse:.2f} mg/dL\n$R^2$ = {_number(row.r2, 3)}  |  n = {row.n_points}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.90, edgecolor="none", pad=4))
            y = np.r_[points.observed_glucose, points.proposed_glucose, points.reduced_glucose]
            span = max(float(y.max() - y.min()), 10)
            ax.set_ylim(y.min() - 0.12 * span, y.max() + 0.30 * span)
            error = points.proposed_glucose - points.observed_glucose
            residual_limit = max(residual_limit, float(np.abs(error).max()) * 1.25)
            residual_ax.axhline(0, color="#59636e", linewidth=0.9)
            residual_ax.vlines(points.time_min, 0, error, color="#176a9a", linewidth=0.9, alpha=0.7)
            residual_ax.scatter(points.time_min, error, color="#176a9a", s=13)
            residual_ax.set_xlabel("Minutes after meal")
            for axis in (ax, residual_ax):
                axis.grid(alpha=0.15)
                axis.set_xlim(-3, 153)
                axis.set_xticks([0, 30, 60, 90, 120, 150])
        for axis in axes[1]:
            axis.set_ylim(-residual_limit, residual_limit)
        axes[0, 0].set_ylabel("Glucose (mg/dL)")
        axes[1, 0].set_ylabel("Fit − observed\n(mg/dL)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
        fig.suptitle("Selected best-R² meal fits (in-sample)", fontsize=16, y=0.995)
        subtitle = "Highest event R² within each group; selected examples are not representative cohort performance"
        if selected.r2.isna().any():
            subtitle = "Highest event R² within each group; undefined-R² groups use minimum RMSE (not representative cases)"
        fig.text(0.5, 0.94, subtitle,
                 ha="center", fontsize=10, color="#55616f")
        fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.14, hspace=0.16, wspace=0.25)
        for suffix in [".png", ".pdf"]:
            fig.savefig(destination.with_suffix(suffix), dpi=250, bbox_inches="tight")
        plt.close(fig)


def plot_fit_diagnostics(evaluation: dict[str, pd.DataFrame], destination: Path) -> None:
    people, events = evaluation["participants"], evaluation["events"]
    pivot = people.pivot(index=["userID", "status"], columns="model", values="rmse").reset_index()
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
        for status in STATUS_ORDER:
            group = pivot.loc[pivot.status.eq(status)]
            if not group.empty:
                axes[0].scatter(group["Reduced 2-state"], group["Proposed 3-state"],
                                color=COLORS[status], label=f"{status} (n={len(group)})", s=40, alpha=0.85)
        upper = max(pivot["Reduced 2-state"].max(), pivot["Proposed 3-state"].max()) * 1.12
        axes[0].plot([0, upper], [0, upper], "--", color="#888888", linewidth=1)
        axes[0].set(xlim=(0, upper), ylim=(0, upper), xlabel="Reduced-model RMSE (mg/dL)",
                    ylabel="Proposed-model RMSE (mg/dL)", title=f"A  |  Every fitted participant (n={len(pivot)})")
        axes[0].text(0.97, 0.04, "Below diagonal: lower proposed-model RMSE", transform=axes[0].transAxes,
                     ha="right", fontsize=8.5, color="#55616f")
        axes[0].legend(frameon=False, fontsize=9)
        for model, color in [("Proposed 3-state", "#176a9a"), ("Reduced 2-state", "#8a8a8a")]:
            errors = np.sort(events.loc[events.model.eq(model), "rmse"].to_numpy())
            axes[1].step(np.r_[0, errors], np.r_[0, np.arange(1, len(errors) + 1) / len(errors)],
                         where="post", color=color, linewidth=2, label=model)
        n_events = events.loc[events.model.eq("Proposed 3-state")].shape[0]
        axes[1].set(xlabel="Meal-event RMSE (mg/dL)", ylabel="Fraction of meal events",
                    ylim=(0, 1.02), xlim=(0, None), title=f"B  |  All meal-event errors (n={n_events})")
        axes[1].legend(frameon=False, fontsize=9, loc="lower right")
        for axis in axes:
            axis.grid(alpha=0.18)
        fig.suptitle("Cohort-wide in-sample fit diagnostics — no fit-quality exclusions", fontsize=14)
        fig.tight_layout()
        for suffix in [".png", ".pdf"]:
            fig.savefig(destination.with_suffix(suffix), dpi=250, bbox_inches="tight")
        plt.close(fig)


def fit_report_lines(evaluation: dict[str, pd.DataFrame]) -> list[str]:
    table = fit_table_display(evaluation)
    parameters = parameter_table_display(evaluation)
    lines = ["### Cohort-wide fit evaluation", ""]
    for frame in [table, parameters]:
        lines += ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        lines.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
        lines.append("")
    lines += [
        "All fitted participants and meals are included. These are unweighted in-sample errors, not held-out validation. "
        "Pooled RMSE/MAE weight observations equally; participant summaries weight people equally. "
        "Within-event R² uses the sum of within-meal centered glucose sums of squares as its denominator. "
        "Quartiles are descriptive, not confidence intervals. Meal-time observations inform initialization. "
        "AIC/BIC are nominal criteria without a serial-correlation correction.", "",
        "Parameter entries are median [Q1, Q3]; the final column is the difference of group medians. "
        "Prediabetes is not combined with T2D. beta2 is derived, insulin is uncalibrated, and "
        "boundary estimates / weak identifiability preclude strong physiological claims. "
        "No significance tests are claimed for these descriptive comparisons.", "",
        "The [selected-fit figure](figures/figure_3_selected_model_fits.png) shows the highest event R² "
        "within each group, not typical cases. All observed points in each selected event are shown; "
        "curves connect the saved model predictions without smoothing the data. "
        "See [all-fit diagnostics](figures/figure_7_fit_diagnostics.png) for unselected cohort coverage.", "",
        "Exports: [Excel workbook](tables/parameter_estimation_evaluation.xlsx), "
        "[overall fit table](tables/fit_overall.tex), [parameter comparison](tables/parameter_group_comparison.tex), "
        "[boundary diagnostics](tables/parameter_bound_diagnostics.tex).", "",
    ]
    return lines


def write_fit_artifacts(
    output: Path, participants: pd.DataFrame, predictions: pd.DataFrame, comparisons: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    evaluation = evaluate_fits(participants, predictions, comparisons)
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    names = {"cohort": "fit_cohort_statistics", "participants": "fit_participant_statistics",
             "events": "fit_event_statistics", "paired": "fit_model_paired_comparison",
             "parameters": "parameter_comparison_statistics", "contrasts": "parameter_healthy_vs_type2",
             "selected": "selected_fit_examples"}
    for key, name in names.items():
        evaluation[key].to_csv(tables / f"{name}.csv", index=False)
    overall = fit_table_display(evaluation)
    parameters = parameter_table_display(evaluation)
    bounds = bound_table_display(evaluation)
    counts = participants.status.value_counts()
    group_sizes = ", ".join(f"{status}: n={counts.get(status, 0)}" for status in STATUS_ORDER)
    _write_tex(overall, tables / "fit_overall.tex", "Overall in-sample goodness of fit across all fitted participants and meals.",
               "tab:fit-overall", FIT_NOTE)
    _write_tex(parameters, tables / "parameter_group_comparison.tex",
               "Participant-level kinetic estimates by clinical status (" + group_sizes + ").",
               "tab:parameter-group-comparison", PARAMETER_NOTE)
    _write_tex(bounds, tables / "parameter_bound_diagnostics.tex", "Boundary diagnostics for participant-level kinetic estimates.",
               "tab:parameter-bound-diagnostics",
               r"Entries are lower/upper boundary counts out of the group size. A value is near a bound "
               r"when its distance is less than 0.5\% of the full search-interval width, matching the fitting "
               r"pipeline. These counts denote proximity, not necessarily exact optimizer contact. "
               r"The equilibrium-derived $\beta_2$ has no independently optimized search interval.")
    notes = pd.DataFrame({"Notes": [
        "In-sample fit evaluation; all fitted participants/events included. No refitting or outcome-based exclusions.",
        "Errors are recomputed from saved predictions with equal observation weights; optimization used robust weighted residuals.",
        "Within-event R2 = 1 - sum(event SSE) / sum(event-centered SST). Global-mean R2 is exported only as a secondary diagnostic.",
        "Participant summaries weight people equally; event summaries weight events equally. Repeated meals are not independent subjects.",
        "Time zero informs initial conditions; strictly-positive-time RMSE/MAE are also exported in the cohort sheet.",
        "Negative R2 values are retained. R2 for a constant observed series is undefined and omitted only from R2 quantiles; undefined counts are exported.",
        "AIC/BIC are nominal residual-based comparisons, without correction for serially correlated CGM errors.",
        "Group parameter estimates: median [Q1,Q3], not confidence intervals. T2D is not pooled with prediabetes. No p-values or significance claims.",
        "Bounds use the original 0.5% of search-range criterion. beta2 is derived; insulin is in uncalibrated model units.",
        "Selected illustration: highest event R2 per group; ties use minimum RMSE then event ID. Not representative performance.",
    ]})
    with pd.ExcelWriter(tables / "parameter_estimation_evaluation.xlsx", engine="openpyxl") as writer:
        for sheet, frame in [("Overall fit", overall), ("Cohort numeric", evaluation["cohort"]),
                             ("Participant fits", evaluation["participants"]), ("Event fits", evaluation["events"]),
                             ("Paired comparison", evaluation["paired"]), ("Parameter comparison", parameters),
                             ("Parameter numeric", evaluation["parameters"]), ("Healthy vs T2D", evaluation["contrasts"]),
                             ("Parameter bounds", bounds), ("Selected examples", evaluation["selected"]), ("Notes", notes)]:
            frame.to_excel(writer, sheet_name=sheet, index=False)
            worksheet = writer.sheets[sheet]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cells in worksheet.columns:
                width = min(75, max(12, max(len(str(c.value or "")) for c in cells) + 2))
                worksheet.column_dimensions[cells[0].column_letter].width = width
    plot_selected_fits(predictions, evaluation["selected"], figures / "figure_3_selected_model_fits")
    plot_fit_diagnostics(evaluation, figures / "figure_7_fit_diagnostics")
    return evaluation
