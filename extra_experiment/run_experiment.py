#!/usr/bin/env python3
"""Fit explicit analysis windows and export exploratory recorded-input controls."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extra_experiment.experiment import (
    ExperimentConfig, fit_record, grid_for, optimize_control, prepare_record, simulate,
)

SOURCE_COMMIT = "28a82c1bdad2635396cab9c6776f71336a141ae9"
SOURCES = {"case_01": "Mai_Hue_24_08.csv", "case_02": "patient.csv"}


def select_analysis_window(frame, case, case01_window="first_meal"):
    """Predeclare the crop from Eating timestamps, never from BG or fit quality."""
    prepare_record(frame)  # Validate the complete source before selecting rows.
    timestamp = pd.to_datetime(frame.Timestamp.astype(str).str.strip(), format="%H:%M")
    minute = timestamp.dt.hour * 60 + timestamp.dt.minute
    first_meal = int(minute[pd.to_numeric(frame.Eating) > 0].iloc[0])
    mode = "first_meal" if case == "case_01" and case01_window == "first_meal" else "full_record"
    selected = frame.copy()
    selection = {"mode": mode, "source_rows": len(frame)}
    if mode == "first_meal":
        # Preserve all rows at a retained timestamp, including split bolus/meal rows.
        meal_times = np.unique(minute[pd.to_numeric(frame.Eating) > 0])
        next_meal = int(meal_times[1]) if len(meal_times) > 1 else None
        keep = (minute >= first_meal - 30) & (minute <= first_meal + 150)
        if next_meal is not None:
            keep &= minute < next_meal
        selected = frame.loc[keep].copy()
        selection.update({"premeal_minutes": 30, "postmeal_minutes": 150,
                          "stop_before_next_meal": True,
                          "rule": "first logged meal -30 to +150 min, excluding next meal and later; no interpolated endpoints"})
    selected_minute = minute.loc[selected.index]
    selection.update({"selected_rows": len(selected),
                      "actual_start_clock": str(selected.Timestamp.iloc[0]),
                      "actual_end_clock": str(selected.Timestamp.iloc[-1]),
                      "first_meal_clock": f"{first_meal // 60:02d}:{first_meal % 60:02d}",
                      "actual_start_relative_to_meal_min": int(selected_minute.iloc[0] - first_meal),
                      "actual_end_relative_to_meal_min": int(selected_minute.iloc[-1] - first_meal)})
    return selected.reset_index(drop=True), selection


def get_source(path, filename, download):
    url = f"https://raw.githubusercontent.com/emplor255/Diabetes-Control/{SOURCE_COMMIT}/{filename}"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}; run with --download")
        with urlopen(url, timeout=30) as response:
            contents = response.read()
        # Validate before saving; never replace an existing local file.
        import io
        prepare_record(pd.read_csv(io.BytesIO(contents)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    return {"source_url": url, "source_commit": SOURCE_COMMIT,
            "local_input": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_filename": filename,
            "provenance_note": "URL identifies intended source; local SHA256 identifies exact bytes used"}


def plot_case(record, row, predictions, trajectories, case, output, selection=None):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 220, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 1]}, constrained_layout=True)
    x = record.time_min / 60
    axes[0].scatter(x, record.glucose, s=16, color="#263238", alpha=.75, label="Recorded BG")
    base = trajectories.loc[trajectories.scenario.eq("Recorded boluses only")]
    first_phase = selection is not None and selection["mode"] == "first_meal"
    label = "First-meal phase fit (logged inputs)" if first_phase else "Full-record fit (logged inputs)"
    subtitle = f" — first meal, {selection['actual_start_clock']}–{selection['actual_end_clock']}" if first_phase else ""
    axes[0].plot(base.time_min / 60, base.glucose, color="#2471A3", lw=2, label=label)
    axes[0].set(ylabel="Glucose (mg/dL; assumed)", title=f"{case}{subtitle}: in-sample fit | RMSE={row['rmse']:.2f}, R²={row['r2']:.3f}")
    axes[0].legend(loc="best")
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].scatter(x, predictions.residual, s=14, color="#2471A3")
    axes[1].set_ylabel("Fit − observed\n(mg/dL)")
    meal = record.loc[record.eating > 0]
    bolus = record.loc[record.bolus > 0]
    axes[2].stem(meal.time_min / 60, meal.eating, linefmt="C2-", markerfmt="C2o", basefmt=" ", label="Eating")
    second = axes[2].twinx()
    second.stem(bolus.time_min / 60, bolus.bolus, linefmt="C1-", markerfmt="C1x", basefmt=" ", label="Bolus")
    xlabel = "Hours since selected window start" if first_phase else "Hours since first record"
    axes[2].set(xlabel=xlabel, ylabel="Eating (native units)")
    second.set_ylabel("Bolus (native units)")
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"{case}_fitting.{ext}")
    plt.close(fig)
    plot_control_case(record, trajectories, case, output, selection)


def plot_control_case(record, trajectories, case, output, selection=None):
    """Presentation-only labels/range; saved trajectories and optimization stay unchanged."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 220, "pdf.fonttype": 42})
    first_phase = selection is not None and selection["mode"] == "first_meal"
    subtitle = f" — first meal, {selection['actual_start_clock']}–{selection['actual_end_clock']}" if first_phase else ""
    xlabel = "Hours since selected window start" if first_phase else "Hours since first record"
    display_labels = {
        "Recorded boluses only": "Conventionally calculated boluses",
        "Recorded boluses + optimized unitless control": "Optimized control",
    } if case == "case_01" else {}
    target_low, target_high = (70, 130) if case == "case_01" else (70, 110)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    for (scenario, curve), color in zip(trajectories.groupby("scenario", sort=False), ["#2471A3", "#C0392B"]):
        axes[0].plot(curve.time_min / 60, curve.glucose, label=display_labels.get(scenario, scenario), color=color, lw=2,
                     ls="-" if scenario == "Recorded boluses only" else "--")
        axes[1].step(curve.time_min / 60, curve.control, where="post", color=color, lw=1.5)
    target_label = "Displayed target: 70–130" if case == "case_01" else "Paper target: 70–110"
    axes[0].axhspan(target_low, target_high, color="#27AE60", alpha=.13, label=target_label)
    axes[0].axhline(70, color="#B03A2E", ls="--", lw=.8)
    axes[0].set(ylabel="Simulated glucose (mg/dL)", title=f"{case}{subtitle}: model-conditional control\nNOT an insulin dose recommendation")
    axes[0].legend(fontsize=8)
    if case == "case_01":
        xlabel += "\nDisplayed target: 70–130; optimization objective unchanged (70–110 mg/dL)."
    axes[1].set(xlabel=xlabel, ylabel="Additional control\n(dimensionless / h)")
    if first_phase:
        meal_hour = float(record.loc[record.eating > 0, "time_min"].iloc[0]) / 60.
        for ax in axes:
            ax.axvline(meal_hour, color="gray", lw=.8, ls=":")
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"{case}_control.{ext}")
    plt.close(fig)


def write_report(output, fits, summaries, audits):
    lines = ["# Additional recorded-input experiments", "",
             "Research simulations only. No insulin dosing recommendation or clinical efficacy claim.", "",
             "## Input audit", "",
             "| Case | Rows / unique times | Meals / boluses | Max gap (min) | Reference BG |",
             "|---|---:|---:|---:|---:|"]
    for case, audit in audits.items():
        lines.append(f"| {case} | {audit['raw_rows']} / {audit['unique_times']} | "
                     f"{audit['recorded_meal_events']} / {audit['recorded_bolus_events']} | "
                     f"{audit['largest_gap_min']:g} | {audit['reference_glucose']:g} |")
    lines += ["", "## Analysis windows", ""]
    for case, audit in audits.items():
        selection = audit.get("selection")
        if selection:
            lines.append(f"- {case}: {selection['mode']}, {selection['actual_start_clock']}–"
                         f"{selection['actual_end_clock']}; {selection['selected_rows']} of "
                         f"{selection['source_rows']} source rows. First meal: {selection['first_meal_clock']}.")
        else:
            lines.append(f"- {case}: full source record, unchanged from the preceding experiment.")
        config = audit.get("config", {})
        lines.append(f"  Fit loss: {config.get('fit_loss', 'soft_l1')}; "
                     f"{config.get('fit_starts', '?')} starts ({config.get('fit_start_strategy', 'local')}); "
                     f"max evaluations/start: {config.get('fit_max_nfev', '?')}.")
    lines += ["- The first-meal window is selected by logged Eating times, not glucose values or fit quality. "
              "No endpoints or missing observations are interpolated. Its fit metrics do not evaluate "
              "the rest of the day and are not directly comparable with full-day scores.", "",
              "## In-sample fit (all times within each analysis window, one parameter vector per case)", "",
              "| Case | N | RMSE | MAE | R² | Optimizer success |",
              "|---|---:|---:|---:|---:|---|"]
    for row in fits.itertuples():
        lines.append(f"| {row.case} | {row.n_observations} | {row.rmse:.3f} | {row.mae:.3f} | "
                     f"{row.r2:.3f} | {row.optimizer_success} |")
    lines += ["", "## Paired simulations with the same recorded meals and boluses", "",
              "| Case / scenario | Objective | Peak BG | Min BG | Time 70–110 (%) | Below 70 (%) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summaries.itertuples():
        lines.append(f"| {row.case} / {row.scenario} | {row.paper_objective:.2f} | {row.max_glucose:.2f} | "
                     f"{row.min_glucose:.2f} | {row.time_70_110_pct:.2f} | {row.time_below_70_pct:.2f} |")
    lines += ["", "## Case-specific findings", ""]
    for row in fits.itertuples():
        if row.r2 < 0:
            lines.append(f"- {row.case}: R²={row.r2:.3f} is negative: the fitted trajectory has greater "
                         "squared error than the observed-mean constant predictor. Control results "
                         "from this poorly fitting model must not be presented as evidence of effectiveness.")
        lines.append(f"- {row.case}: data-only log-parameter Jacobian condition number "
                     f"{row.data_log_jacobian_condition:.3g}; inspect parameter boundaries and "
                     "multistart variability before interpreting individual coefficients.")
        pair = summaries.loc[summaries.case.eq(row.case)]
        baseline, optimized = pair.iloc[0], pair.iloc[1]
        delta_low = optimized.time_below_70_pct - baseline.time_below_70_pct
        if delta_low > 1e-8:
            lines.append(f"- {row.case}: optimized time below 70 increases by {delta_low:.6f} percentage "
                         "points despite a lower objective. The paper objective does not enforce a safety constraint.")
        if abs(optimized.paper_objective - baseline.paper_objective) < .1:
            lines.append(f"- {row.case}: objective improvement is less than 0.1 in its native units; "
                         "the two simulated glucose curves are essentially coincident. No meaningful "
                         "clinical benefit can be inferred.")
    lines += ["", "## Interpretation and limitations", "",
              "- BG is provisionally interpreted as mg/dL; source CSVs do not declare units. "
              "Eating and bolus amounts are used in native units, with fitted effective gains. "
              "Do not interpret gains as physiological dose-conversion factors.",
              "- The seven freely fitted parameters are five shared kinetic coefficients and two "
              "input gains. K=120 mg/dL and basal insulin=1 are fixed; beta2 is derived at the assumed reference.",
              "- A fourth, first-order bolus depot is an explicit extension of the paper model, "
              "not part of the Hall2018 fit. Its absorption rate is assumed to be 1/h, not estimated "
              "or calibrated for a drug. Inputs are impulses, not interpolated dose rates.",
              "- No measured insulin concentration, basal-dose history, insulin-on-board history, "
              "or diagnostic group is supplied. Initial digestive/bolus depots are assumed zero. "
              "Initial quick insulin is the reference equilibrium. These are strong assumptions.",
              "- case_02 begins with BG=65 and a meal/bolus, without a premeal baseline; its first "
              "BG is used as reference by assumption. Fifteen points cannot establish reliable "
              "identification of seven parameters. See bound flags, multistart results, and Jacobian diagnostics.",
              "- Noise labels are preserved, not assigned undocumented reliability weights. "
              "Duplicate-time BG is averaged, meal/bolus amounts are summed. No observations "
              "are discarded for a poor fit; no interpolated observations enter the fit.",
              "- All scores are in-sample. There is no holdout validation, clinical validation, "
              "uncertainty propagation, or between-patient statistical inference from these two records.",
              "- The comparator retains recorded boluses. Optimization adds nonnegative dimensionless "
              "control directly to the model insulin equation; it does not recommend replacing recorded "
              "boluses. All future recorded inputs are known: this is offline open-loop optimization.",
              "- The original paper objective has no hard hypoglycaemia constraint and may be negative. "
              "A smaller objective does not guarantee safer glucose. Inspect minimum BG and time below 70, "
              "including any worsening. Duration metrics use piecewise-linear trajectories, not sample counts.",
              "- A zero-control fallback is always a candidate. Optimizer status is exported even "
              "when a nonconverged candidate has the lowest finite objective.",
              "- Integration is rechecked with tighter tolerances; numerical_check.csv records "
              "the maximum glucose difference for each scenario.", ""]
    (output / "report.md").write_text("\n".join(lines))


def write_latex_table(frame, path, caption, label):
    """Small booktabs export without pandas' optional Jinja2 dependency."""
    def escape(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + "l" * len(frame.columns) + "}", r"\toprule",
             " & ".join(escape(c) for c in frame.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(escape(x) for x in row) + r" \\" for row in frame.itertuples(index=False, name=None)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_manuscript_tables(output, fits, summaries):
    write_latex_table(fits[["case", "n_observations", "rmse", "mae", "r2"]].rename(
        columns={"case": "Case", "n_observations": "N", "rmse": "RMSE", "mae": "MAE", "r2": "$R^2$"}),
        output / "fitting_table.tex",
        "Exploratory in-sample fitting within the selected analysis windows. "
        "BG units provisionally mg/dL; see the input audit for window definitions.", "tab:extra-fitting")
    compact = summaries.copy()
    compact["scenario"] = compact.scenario.map({"Recorded boluses only": "Recorded",
        "Recorded boluses + optimized unitless control": "Recorded + control"})
    write_latex_table(compact[["case", "scenario", "paper_objective", "min_glucose", "max_glucose",
                               "time_70_110_pct", "time_below_70_pct"]].rename(columns={
        "case": "Case", "scenario": "Scenario", "paper_objective": "Objective", "min_glucose": "Min BG",
        "max_glucose": "Peak BG", "time_70_110_pct": "70--110 (%)", "time_below_70_pct": "Below 70 (%)"}),
        output / "control_table.tex", "Model-conditional control with recorded boluses retained; not clinical dosing.",
        "tab:extra-control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "extra_experiment/results")
    parser.add_argument("--fit-starts", type=int, default=4)
    parser.add_argument("--fit-max-nfev", type=int, default=200)
    parser.add_argument("--fit-start-strategy", choices=["local", "mixed"], default="local")
    parser.add_argument("--fit-loss", choices=["soft_l1", "linear"], default="soft_l1")
    parser.add_argument("--control-maxiter", type=int, default=80)
    parser.add_argument("--reuse-fits", action="store_true", help="Require identical input hashes and fit configuration")
    parser.add_argument("--cases", nargs="+", choices=list(SOURCES), default=list(SOURCES),
                        help="Recompute these cases; preserve the other case's saved artifacts and summary rows")
    parser.add_argument("--case01-window", choices=["first_meal", "full_record"], default="first_meal")
    args = parser.parse_args()
    cfg = ExperimentConfig(fit_starts=args.fit_starts, fit_max_nfev=args.fit_max_nfev,
                           control_maxiter=args.control_maxiter,
                           fit_start_strategy=args.fit_start_strategy, fit_loss=args.fit_loss)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    all_fits, all_summaries, audits, numerical = [], [], {}, []
    unselected = set(SOURCES) - set(args.cases)
    if unselected:
        # Require existing complete summaries rather than silently dropping case 2.
        saved_fits = pd.read_csv(output / "fit_summary.csv")
        saved_control = pd.read_csv(output / "control_summary.csv")
        saved_numerical = pd.read_csv(output / "numerical_check.csv")
        saved_audit = json.loads((output / "data_audit.json").read_text())
        for case in sorted(unselected):
            rows = saved_fits.loc[saved_fits.case.eq(case)]
            if len(rows) != 1 or case not in saved_audit:
                raise ValueError(f"Missing saved results for unselected {case}")
            all_fits.extend(rows.to_dict("records"))
            all_summaries.append(saved_control.loc[saved_control.case.eq(case)].copy())
            numerical.extend(saved_numerical.loc[saved_numerical.case.eq(case)].to_dict("records"))
            audits[case] = saved_audit[case]
    for case, filename in SOURCES.items():
        if case not in args.cases:
            continue
        print(f"Processing {case} ({filename})...", flush=True)
        path = ROOT / "extra_experiment/data" / filename
        provenance = get_source(path, filename, args.download)
        selected, selection = select_analysis_window(pd.read_csv(path), case, args.case01_window)
        record, audit = prepare_record(selected)
        audit.update(provenance)
        audit["selection"] = selection
        audit["config"] = asdict(cfg)
        audits[case] = audit
        cache = output / f"{case}_fit.json"
        if args.reuse_fits and cache.exists():
            saved = json.loads(cache.read_text())
            if (saved["sha256"] != provenance["sha256"] or asdict(ExperimentConfig(**saved["config"])) != asdict(cfg)
                    or saved.get("selection") != selection):
                raise ValueError("Cached fit input/configuration mismatch")
            parameters, row = np.asarray(saved["parameters"]), saved["summary"]
        else:
            parameters, row, runs, diagnostics = fit_record(record, audit["reference_glucose"], cfg)
            runs.to_csv(output / f"{case}_fit_multistart.csv", index=False)
            diagnostics.to_csv(output / f"{case}_parameter_diagnostics.csv", index=False)
            cache.write_text(json.dumps({"sha256": provenance["sha256"], "config": asdict(cfg), "selection": selection,
                                         "parameters": parameters.tolist(), "summary": row}, indent=2))
        row["case"] = case
        all_fits.append(row)
        record.to_csv(output / f"{case}_prepared.csv", index=False)
        prediction = record[["time_min", "glucose"]].rename(columns={"glucose": "observed_glucose"}).copy()
        prediction["fitted_glucose"] = simulate(record, audit["reference_glucose"], parameters,
                                                record.time_min.to_numpy() / 60., cfg)[:, 1]
        prediction["residual"] = prediction.fitted_glucose - prediction.observed_glucose
        prediction.to_csv(output / f"{case}_predictions.csv", index=False)
        summary, trajectory, bins, logs = optimize_control(record, audit["reference_glucose"], parameters, cfg)
        summary.insert(0, "case", case)
        all_summaries.append(summary)
        trajectory.to_csv(output / f"{case}_trajectories.csv", index=False)
        bins.to_csv(output / f"{case}_control_bins.csv", index=False)
        logs.to_csv(output / f"{case}_control_optimizer.csv", index=False)
        for scenario, curve in trajectory.groupby("scenario", sort=False):
            controls = np.zeros(len(bins)) if scenario == "Recorded boluses only" else bins.unitless_control.to_numpy()
            tight = simulate(record, audit["reference_glucose"], parameters, curve.time_min.to_numpy() / 60.,
                             replace(cfg, rtol=1e-8, atol=1e-10), controls)
            difference = float(np.max(abs(tight[:, 1] - curve.glucose.to_numpy())))
            numerical.append({"case": case, "scenario": scenario, "max_BG_difference_mg_dl": difference})
            if difference > .05:
                raise RuntimeError(f"Numerical tolerance check failed: {difference}")
        plot_case(record, row, prediction, trajectory, case, figures, selection)
        (output / "data_audit.json").write_text(json.dumps(audits, indent=2))
    fits = pd.DataFrame(all_fits).sort_values("case", kind="stable")
    summaries = pd.concat(all_summaries, ignore_index=True).sort_values("case", kind="stable")
    numerical.sort(key=lambda row: row["case"])
    audits = dict(sorted(audits.items()))
    (output / "data_audit.json").write_text(json.dumps(audits, indent=2))
    fits.to_csv(output / "fit_summary.csv", index=False)
    summaries.to_csv(output / "control_summary.csv", index=False)
    pd.DataFrame(numerical).to_csv(output / "numerical_check.csv", index=False)
    changes = []
    for case, group in summaries.groupby("case", sort=False):
        baseline, optimized = group.iloc[0], group.iloc[1]
        changes.append({"case": case, **{f"delta_{name}": float(optimized[name] - baseline[name])
                        for name in ["paper_objective", "max_glucose", "min_glucose", "time_70_110_pct",
                                     "time_70_180_pct", "time_below_70_pct"]}})
    pd.DataFrame(changes).to_csv(output / "control_paired_changes.csv", index=False)
    with pd.ExcelWriter(output / "extra_experiment_summary.xlsx") as writer:
        fits.to_excel(writer, sheet_name="fit", index=False)
        summaries.to_excel(writer, sheet_name="control", index=False)
        pd.DataFrame(changes).to_excel(writer, sheet_name="paired_changes", index=False)
        pd.DataFrame(numerical).to_excel(writer, sheet_name="numerical_check", index=False)
    write_manuscript_tables(output, fits, summaries)
    write_report(output, fits, summaries, audits)
    print(f"Finished: {output}", flush=True)


if __name__ == "__main__":
    main()
