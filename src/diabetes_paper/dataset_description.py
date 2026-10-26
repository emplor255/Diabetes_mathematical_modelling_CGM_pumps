"""Descriptive dataset artifacts; no model fitting or control optimization."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .data import PreparedData, STATUS_ORDER


MEAL_ORDER = ["PB 1", "PB 2", "Bar 1", "Bar 2", "CF 1", "CF 2"]
AVAILABILITY_ORDER = ["Absent", "Excluded", "Retained"]
AVAILABILITY_COLORS = ["#e6e9ed", "#dfaa55", "#348b83"]
STATUS_COLORS = ["#0173b2", "#de8f05", "#029e73"]


def meal_availability(prepared: PreparedData) -> pd.DataFrame:
    """One row per nominal participant/meal slot, distinguishing absent from QC-failed."""
    qc = prepared.meal_qc
    if qc.duplicated(["userID", "Meal"]).any():
        raise ValueError("QC must contain at most one row per participant/meal slot")
    unexpected = set(qc["Meal"]).difference(MEAL_ORDER)
    if unexpected:
        raise ValueError(f"Unexpected standardized meal labels: {sorted(unexpected)}")
    users = sorted(qc["userID"].unique())
    grid = pd.MultiIndex.from_product(
        [users, MEAL_ORDER], names=["userID", "Meal"]
    ).to_frame(index=False)
    grid = grid.merge(qc, on=["userID", "Meal"], how="left", validate="one_to_one")
    grid = grid.merge(
        prepared.hall_participants[["id", "status"]].rename(columns={"id": "userID"}),
        on="userID", how="left", validate="many_to_one",
    )
    if grid["status"].isna().any():
        raise ValueError("Cannot describe meal participants without Hall metadata")
    grid["availability"] = np.select(
        [grid["event_id"].isna(), grid["usable"].eq(True)],
        ["Absent", "Retained"], default="Excluded",
    )
    grid["meal_type"] = grid["Meal"].str.split().str[0]
    grid["replicate"] = grid["Meal"].str.split().str[-1].astype(int)
    return grid


def dataset_group_summary(prepared: PreparedData, availability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for status in ["All participants", *STATUS_ORDER]:
        hall = prepared.hall_participants
        slots = availability
        if status != "All participants":
            hall = hall.loc[hall["status"].eq(status)]
            slots = slots.loc[slots["status"].eq(status)]
        meal_people = hall.loc[hall["id"].isin(slots["userID"])]
        row = {
            "status": status,
            "hall_participants": len(hall),
            "hall_cgm_rows": int(hall["n_cgm"].sum()),
            "hall_age_mean": hall["age"].mean(),
            "hall_age_sd": hall["age"].std(),
            "meal_participants": len(meal_people),
            "meal_age_mean": meal_people["age"].mean(),
            "meal_age_sd": meal_people["age"].std(),
            "nominal_meal_slots": len(slots),
            "observed_windows": int(slots["availability"].ne("Absent").sum()),
            "excluded_windows": int(slots["availability"].eq("Excluded").sum()),
            "retained_windows": int(slots["availability"].eq("Retained").sum()),
            "absent_windows": int(slots["availability"].eq("Absent").sum()),
            "retained_participants": slots.loc[
                slots["availability"].eq("Retained"), "userID"
            ].nunique(),
        }
        for meal in ["PB", "Bar", "CF"]:
            row[f"retained_{meal}_windows"] = int(
                (slots["availability"].eq("Retained") & slots["meal_type"].eq(meal)).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def meal_profile_sample_sizes(points: pd.DataFrame) -> pd.DataFrame:
    """The actual pointwise denominators of the existing Figure 2 (no interpolation)."""
    usable = points.loc[points["usable"]]
    return usable.groupby(["status", "meal_type", "relative_min"], as_index=False).agg(
        participants=("userID", "nunique"), events=("event_id", "nunique")
    )


def _write_cohort_table(summary: pd.DataFrame, path: Path) -> None:
    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Composition of the supplied Hall2018 extract and its standardized-meal subset. "
        r"Meal-window counts are after the present analysis's quality control. "
        r"The meal subset is nested within the Hall cohort, not an independent dataset.}",
        r"\label{tab:dataset-cohort}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& Hall & Hall CGM & Meal & \multicolumn{3}{c}{Retained meal windows} \\",
        r"\cmidrule(lr){5-7}",
        r"Status & persons & readings & persons & PB & Bar & CF \\",
        r"\midrule",
    ]
    for status in [*STATUS_ORDER, "All participants"]:
        r = summary.loc[summary["status"].eq(status)].iloc[0]
        label = "Normoglycemic" if status == "Healthy" else status
        if status == "All participants":
            rows.append(r"\midrule")
        counts = [int(r[c]) for c in [
            "hall_participants", "hall_cgm_rows", "meal_participants",
            "retained_PB_windows", "retained_Bar_windows", "retained_CF_windows",
        ]]
        rows.append(label + " & " + " & ".join(f"{n:,}" for n in counts) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(rows), encoding="utf-8")


def plot_dataset_overview(
    prepared: PreparedData, summary: pd.DataFrame, availability: pd.DataFrame,
    destination: Path,
) -> None:
    """Cohort denominators, QC accounting, and all participant/meal slots in one figure."""
    totals = summary.iloc[0]
    groups = summary.set_index("status").loc[STATUS_ORDER]
    usable = prepared.meal_points.loc[prepared.meal_points["usable"]]
    qc = prepared.meal_qc
    excluded = qc.loc[~qc["usable"]]
    incomplete = excluded["excluded_reason"].str.contains("incomplete_180_min_window").sum()
    sparse_only = excluded["excluded_reason"].eq("too_few_unique_points").sum()
    other = len(excluded) - incomplete - sparse_only
    users = availability[["userID", "status"]].drop_duplicates()
    users["group_order"] = users["status"].map(dict(zip(STATUS_ORDER, range(3))))
    users = users.sort_values(["group_order", "userID"])
    matrix = availability.pivot(index="userID", columns="Meal", values="availability")
    matrix = matrix.reindex(index=users["userID"], columns=MEAL_ORDER)
    values = matrix.apply(lambda col: col.map(dict(zip(AVAILABILITY_ORDER, range(3))))).to_numpy()

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 250, "pdf.fonttype": 42,
    }):
        fig = plt.figure(figsize=(13.2, 10.0), facecolor="white")
        layout = fig.add_gridspec(
            2, 2, width_ratios=[1.08, 1.0], height_ratios=[0.82, 1.18],
            left=0.10, right=0.96, bottom=0.10, top=0.88, wspace=0.51, hspace=0.48,
        )
        ax_cohort = fig.add_subplot(layout[0, 0])
        ax_flow = fig.add_subplot(layout[1, 0])
        ax_matrix = fig.add_subplot(layout[:, 1])
        fig.suptitle("Dataset composition and standardized-meal coverage", fontsize=17, y=0.975)
        fig.text(
            0.5, 0.937,
            f"{int(totals.hall_participants)} Hall participants  |  "
            f"{int(totals.meal_participants)} meal participants  |  "
            f"{int(totals.retained_windows)} retained meal windows",
            ha="center", color="#475569", fontsize=11,
        )

        y = np.arange(3)
        ax_cohort.barh(y - 0.18, groups.hall_participants, height=0.31, color="#52738b", label="Hall extract")
        ax_cohort.barh(y + 0.18, groups.meal_participants, height=0.31, color="#348b83", label="Meal subset")
        for offset, column in [(-0.18, "hall_participants"), (0.18, "meal_participants")]:
            for i, count in enumerate(groups[column]):
                ax_cohort.text(count + 0.6, i + offset, str(count), va="center", fontsize=11)
        ax_cohort.set_yticks(y, ["Normoglycemic", "Prediabetes", "Type 2 diabetes"])
        ax_cohort.invert_yaxis()
        ax_cohort.set_xlim(0, max(groups.hall_participants.max(), 1) * 1.17)
        ax_cohort.set_xlabel("Number of participants")
        ax_cohort.set_title("A  |  Nested participant cohorts", loc="left", fontweight="bold", pad=14)
        ax_cohort.xaxis.grid(True, color="#e7ebef", linewidth=0.7)
        ax_cohort.set_axisbelow(True)
        ax_cohort.legend(frameon=False, loc="lower right", fontsize=9)

        ax_flow.set_title("B  |  Meal-window quality control", loc="left", fontweight="bold", pad=14)
        ax_flow.set_axis_off()
        steps = [
            (0.89, f"{int(totals.nominal_meal_slots)} nominal meal slots\n"
             f"{int(totals.meal_participants)} people × 3 meal types × 2 repeats", "#edf1f5"),
            (0.60, f"{int(totals.observed_windows)} observed windows\n"
             f"{prepared.audit['meal_input_rows']:,} input CGM rows", "#edf1f5"),
            (0.28, f"{int(totals.retained_windows)} retained windows\n"
             f"{int(totals.retained_participants)} people; {len(usable):,} unique CGM points", "#dcefeb"),
        ]
        for yy, text, color in steps:
            ax_flow.text(0.5, yy, text, transform=ax_flow.transAxes, ha="center", va="center",
                         fontsize=11, linespacing=1.5,
                         bbox=dict(boxstyle="round,pad=0.65", facecolor=color, edgecolor="none"))
        for top, bottom in [(0.79, 0.70), (0.49, 0.38)]:
            ax_flow.annotate("", xy=(0.5, bottom), xytext=(0.5, top), xycoords="axes fraction",
                             arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.4))
        ax_flow.text(0.57, 0.745, f"{int(totals.absent_windows)} absent", transform=ax_flow.transAxes,
                     va="center", fontsize=9.5, color="#64748b")
        ax_flow.text(0.57, 0.435, f"{int(totals.excluded_windows)} excluded", transform=ax_flow.transAxes,
                     va="center", fontsize=9.5, color="#9a641b")
        detail = f"Excluded: {incomplete} incomplete spans + {sparse_only} sparse full-span windows"
        if other:
            detail += f" + {other} other"
        ax_flow.text(0.5, 0.10, detail, transform=ax_flow.transAxes, ha="center", fontsize=8.5, color="#475569")
        ax_flow.text(
            0.5, 0.015,
            f"Retained points: {usable.relative_min.lt(0).sum():,} pre-meal / "
            f"{usable.relative_min.ge(0).sum():,} post-meal (including t = 0)",
            transform=ax_flow.transAxes, ha="center", fontsize=9, color="#475569",
        )

        cmap = ListedColormap(AVAILABILITY_COLORS)
        ax_matrix.pcolormesh(
            np.arange(7) - 0.5, np.arange(len(users) + 1) - 0.5, values,
            cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
            shading="flat", edgecolors="white", linewidth=0.7, rasterized=False,
        )
        ax_matrix.set_xlim(-0.5, 5.5)
        ax_matrix.set_ylim(len(users) - 0.5, -0.5)
        ax_matrix.set_title("C  |  Coverage for every meal participant", loc="left", fontweight="bold", pad=14)
        ax_matrix.set_xticks(np.arange(6), MEAL_ORDER)
        ax_matrix.xaxis.tick_top()
        ax_matrix.tick_params(axis="x", top=False, labelsize=9)
        # Public study identifiers, not names. Retain the complete identifier in CSV.
        ax_matrix.set_yticks(np.arange(len(users)), users["userID"].str.replace("2133-", "", regex=False))
        ax_matrix.tick_params(axis="y", length=0, labelsize=8)
        ax_matrix.set_ylabel("Participant (ID suffix)", labelpad=8)
        ax_matrix.set_xticks(np.arange(-0.5, 6, 1), minor=True)
        ax_matrix.set_yticks(np.arange(-0.5, len(users), 1), minor=True)
        ax_matrix.grid(which="minor", color="white", linewidth=1)
        ax_matrix.tick_params(which="minor", bottom=False, left=False, top=False)
        for spine in ax_matrix.spines.values():
            spine.set_visible(False)
        start = 0
        labels = ["Normoglycemic", "Prediabetes", "Type 2 diabetes"]
        for status, label, color in zip(STATUS_ORDER, labels, STATUS_COLORS):
            size = int(users.status.eq(status).sum())
            if not size:
                continue
            end = start + size
            if start:
                ax_matrix.axhline(start - 0.5, color="white", linewidth=3)
            ax_matrix.text(6.1, (start + end - 1) / 2, f"{label}\n(n = {size})", ha="left",
                           va="center", fontsize=9, color=color, rotation=90)
            start = end
        for tick, status in zip(ax_matrix.get_yticklabels(), users.status):
            tick.set_color(dict(zip(STATUS_ORDER, STATUS_COLORS))[status])
        handles = [Patch(facecolor=color, label=f"{label} ({int(availability.availability.eq(label).sum())})")
                   for label, color in zip(AVAILABILITY_ORDER, AVAILABILITY_COLORS)]
        ax_matrix.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                         ncol=3, frameon=False, fontsize=9, columnspacing=1.0)
        fig.text(0.5, 0.037,
                 f"Retained = {prepared.audit['meal_expected_window_minutes']:g}-minute span "
                 f"(±{prepared.audit['meal_duration_tolerance_minutes']:g} min), "
                 f"≥{prepared.audit['meal_min_event_points']} unique points, and linked participant metadata. "
                 "Internal missing samples can remain.",
                 ha="center", fontsize=9, color="#475569")
        for suffix in [".png", ".pdf"]:
            fig.savefig(destination.with_suffix(suffix), bbox_inches="tight", facecolor="white")
        plt.close(fig)


def write_dataset_artifacts(prepared: PreparedData, output: Path) -> None:
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    availability = meal_availability(prepared)
    summary = dataset_group_summary(prepared, availability)
    summary.to_csv(tables / "dataset_overview.csv", index=False)
    availability.to_csv(tables / "dataset_meal_availability.csv", index=False)
    meal_profile_sample_sizes(prepared.meal_points).to_csv(
        tables / "meal_profile_sample_sizes.csv", index=False
    )
    usable = prepared.meal_points.loc[prepared.meal_points["usable"]]
    audit = {
        **prepared.audit,
        "description_scope": "All supplied data; not a model-fitting subset or held-out cohort",
        "nominal_meal_slots": int(len(availability)),
        "absent_meal_slots": int(availability.availability.eq("Absent").sum()),
        "retained_unique_points": len(usable),
        "retained_premeal_points": int(usable.relative_min.lt(0).sum()),
        "retained_postmeal_points_including_zero": int(usable.relative_min.ge(0).sum()),
        "meal_supplement_label": "S6 Data (file identifier .s015)",
        "source_article_doi": "10.1371/journal.pbio.2005143",
        "source_meal_supplement_doi": "10.1371/journal.pbio.2005143.s015",
    }
    (tables / "dataset_description_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    _write_cohort_table(summary, tables / "dataset_overview.tex")
    plot_dataset_overview(prepared, summary, availability, figures / "figure_6_dataset_overview")
