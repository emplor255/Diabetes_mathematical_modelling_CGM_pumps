import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diabetes_paper.fit_evaluation import (
    evaluate_fits, fit_table_display, parameter_comparison, parameter_table_display,
    select_fit_examples, write_fit_artifacts,
)


def synthetic_fits():
    participant_rows, prediction_rows, comparison_rows = [], [], []
    definitions = [
        ("A", "Healthy", [([100, 105, 110, 115], [0, 2, 2, 2]), ([200, 201], [0, 10])]),
        ("B", "Prediabetes", [([100, 100, 100], [0, 1, -1])]),
        ("C", "Type 2 diabetes", [([90, 95, 100], [0, 1, 1])]),
    ]
    for user, status, meals in definitions:
        n_points = sum(len(observed) for observed, _ in meals)
        participant_rows.append({
            "userID": user, "status": status, "n_events": len(meals), "n_observations": n_points,
            "alpha": 1.0, "beta0": 1e-4, "beta1": 0.5, "beta2_derived": 400,
            "gamma": 10, "delta": 2,
        })
        for index, (observed, errors) in enumerate(meals):
            for k, (glucose, error) in enumerate(zip(observed, errors)):
                prediction_rows.append({
                    "userID": user, "status": status, "event_id": f"{user}__PB_{index + 1}",
                    "Meal": f"PB {index + 1}", "meal_type": "PB", "time_min": 5 * k,
                    "observed_glucose": glucose, "proposed_glucose": glucose + error,
                    "reduced_glucose": glucose + 2, "weight": 1.0,
                })
        for model in ["Proposed 3-state", "Reduced 2-state"]:
            sse = sum(np.sum(np.asarray(errors, dtype=float) ** 2) for _, errors in meals) if model.startswith("Proposed") else 4 * n_points
            comparison_rows.append({"userID": user, "status": status, "model": model, "sse": sse,
                                    "aic": sse + 10, "bic": sse + 20, "optimizer_success": True})
    return pd.DataFrame(participant_rows), pd.DataFrame(prediction_rows), pd.DataFrame(comparison_rows)


class FitEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.participants, self.predictions, self.comparisons = synthetic_fits()

    def evaluate(self, predictions=None, comparisons=None):
        return evaluate_fits(self.participants, self.predictions if predictions is None else predictions,
                             self.comparisons if comparisons is None else comparisons)

    def test_metrics_denominators_and_no_quality_exclusions(self):
        result = self.evaluate()
        overall = result["cohort"].query("group == 'All participants' and model == 'Proposed 3-state'").iloc[0]
        self.assertEqual((overall.participants, overall.events, overall.observations), (3, 4, 12))
        error = self.predictions.proposed_glucose - self.predictions.observed_glucose
        self.assertAlmostEqual(overall.pooled_rmse, np.sqrt(np.mean(error ** 2)))
        self.assertAlmostEqual(overall.pooled_mae, abs(error).mean())
        people = result["participants"].query("model == 'Proposed 3-state'")
        self.assertAlmostEqual(overall.participant_rmse_median, people.rmse.median())
        self.assertNotAlmostEqual(overall.pooled_rmse, people.rmse.mean())
        sst = ((self.predictions.observed_glucose - self.predictions.groupby("event_id").observed_glucose.transform("mean")) ** 2).sum()
        self.assertAlmostEqual(overall.r2_within_event, 1 - (error ** 2).sum() / sst)
        self.assertLess(overall.r2_within_event, overall.pooled_r2_global_mean)
        self.assertEqual(overall.events_r2_negative, 1)
        self.assertEqual(overall.events_r2_undefined, 1)
        self.assertEqual(overall.strictly_postmeal_observations, 8)
        self.assertEqual(len(result["events"]), 8)  # Four events for each model, including bad/flat fits.

    def test_selection_is_explicit_deterministic_and_not_rmse_only(self):
        result = self.evaluate()
        selected = result["selected"].set_index("status")
        self.assertEqual(selected.loc["Healthy", "event_id"], "A__PB_1")
        self.assertEqual(selected.loc["Healthy", "candidate_events"], 2)
        self.assertIn("undefined", selected.loc["Prediabetes", "selection_rule"])
        shuffled = select_fit_examples(result["events"].sample(frac=1, random_state=17))
        pd.testing.assert_frame_equal(result["selected"], shuffled)

    def test_group_separation_derived_parameter_and_bound_definition(self):
        p = self.participants.copy()
        width = 2 - 1e-4
        p.loc[p.status.eq("Prediabetes"), "beta0"] = 1e-4 + 0.0049 * width
        p.loc[p.status.eq("Type 2 diabetes"), "beta0"] = 1e-4 + 0.0051 * width
        p.loc[p.status.eq("Type 2 diabetes"), "beta1"] = 0.2
        summary, contrasts = parameter_comparison(p)
        summary = summary.set_index(["status", "parameter"])
        self.assertEqual(summary.loc[("Healthy", "beta0"), "n_near_lower"], 1)
        self.assertEqual(summary.loc[("Prediabetes", "beta0"), "n_near_lower"], 1)
        self.assertEqual(summary.loc[("Type 2 diabetes", "beta0"), "n_near_lower"], 0)
        self.assertTrue(np.isnan(summary.loc[("Healthy", "beta2_derived"), "n_near_lower"]))
        contrast = contrasts.set_index("parameter").loc["beta1"]
        self.assertEqual((contrast.n_healthy, contrast.n_type2), (1, 1))
        self.assertAlmostEqual(contrast.type2_minus_healthy_median, -0.3)

    def test_inconsistent_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.evaluate(pd.concat([self.predictions, self.predictions.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "every fitted participant"):
            self.evaluate(self.predictions.loc[self.predictions.userID.ne("C")])
        broken = self.predictions.copy()
        broken.loc[0, "status"] = "Prediabetes"
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            self.evaluate(broken)
        broken = self.predictions.copy()
        broken.loc[0, "proposed_glucose"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.evaluate(broken)
        broken = self.comparisons.copy()
        broken.loc[0, "sse"] += 1
        with self.assertRaisesRegex(ValueError, "SSE disagrees"):
            self.evaluate(comparisons=broken)
        with self.assertRaisesRegex(ValueError, "both models"):
            self.evaluate(comparisons=self.comparisons.iloc[1:])

    def test_exports_and_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = write_fit_artifacts(output, self.participants, self.predictions, self.comparisons)
            workbook = pd.read_excel(output / "tables/parameter_estimation_evaluation.xlsx", sheet_name=None)
            self.assertEqual(len(workbook["Participant fits"]), 6)
            self.assertEqual(len(workbook["Event fits"]), 8)
            self.assertEqual(len(workbook["Selected examples"]), 3)
            self.assertEqual(len(fit_table_display(result)), 13)
            self.assertIn("0.0001", parameter_table_display(result).set_index("Parameter").loc["beta0", "Healthy"])
            for relative in ["figures/figure_3_selected_model_fits.png", "figures/figure_3_selected_model_fits.pdf",
                             "figures/figure_7_fit_diagnostics.pdf", "tables/fit_overall.tex",
                             "tables/parameter_group_comparison.tex", "tables/parameter_bound_diagnostics.tex"]:
                self.assertGreater((output / relative).stat().st_size, 0)


class SavedFitIntegrationTests(unittest.TestCase):
    def test_full_cohort_metrics_match_saved_predictions(self):
        tables = Path(__file__).resolve().parents[1] / "outputs/tables"
        if not (tables / "model_predictions.csv").exists():
            self.skipTest("Full saved run is not available")
        p = pd.read_csv(tables / "participant_parameters.csv")
        if len(p) != 30:
            self.skipTest("Current saved run is a fitting subset, not the reference full run")
        result = evaluate_fits(p, pd.read_csv(tables / "model_predictions.csv"), pd.read_csv(tables / "model_comparison.csv"))
        overall = result["cohort"].query("group == 'All participants' and model == 'Proposed 3-state'").iloc[0]
        self.assertEqual((overall.participants, overall.events, overall.observations), (30, 147, 4452))
        self.assertAlmostEqual(overall.pooled_rmse, 10.832582506273141)
        self.assertAlmostEqual(overall.pooled_mae, 7.000472868438382)
        self.assertAlmostEqual(overall.r2_within_event, 0.6653257899453455)
        self.assertEqual(overall.events_r2_negative, 22)
        self.assertEqual(overall.participants_lower_bic, 14)
        self.assertEqual(result["selected"].event_id.tolist(), ["2133-025__CF_2", "2133-036__Bar_1", "2133-018__PB_1"])
        bounds = result["parameters"]
        self.assertEqual(bounds.loc[bounds.parameter.eq("beta0"), "n_near_lower"].sum(), 24)
        self.assertEqual(bounds.loc[bounds.parameter.eq("gamma"), "n_near_upper"].sum(), 21)
        # Integration: the normal report writer preserves the fit tables and then
        # continues to the control section; this path does not optimize anything.
        from diabetes_paper.reporting import write_analysis_report
        audit = json.loads((tables / "data_audit.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "analysis_report.md"
            write_analysis_report(report, audit, p, pd.read_csv(tables / "model_comparison.csv"),
                                  pd.DataFrame(), fit_evaluation=result)
            content = report.read_text()
            self.assertIn("Cohort-wide fit evaluation", content)
            self.assertIn("10.83", content)
            self.assertLess(content.index("Cohort-wide fit evaluation"), content.index("## Optimal control"))


if __name__ == "__main__":
    unittest.main()
