import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from diabetes_paper.control_evaluation import (
    BASELINE, OPTIMIZED, evaluate_control_cohort, write_control_tables,
)


def example_data():
    participants = pd.DataFrame({"userID": ["A", "B", "C"],
                                 "status": ["Healthy", "Healthy", "Prediabetes"]})
    # Each row is one participant, never one observation/meal window.
    times = [(50, 60, 0, 0), (100, 80, 0, 5), (0, 100, 0, 0)]
    rows = []
    for person, (tir_before, tir_after, low_before, low_after) in zip(participants.itertuples(), times):
        for scenario, tir, low, objective, effort in (
            (BASELINE, tir_before, low_before, -100, 0),
            (OPTIMIZED, tir_after, low_after, -120, 2),
        ):
            rows.append({
                "userID": person.userID, "status": person.status, "scenario": scenario,
                "optimizer_success": person.userID != "C",  # Must not be excluded.
                "paper_objective": objective, "total_control_intensity": effort,
                "peak_glucose": 130 if scenario == BASELINE else 120,
                "min_glucose": 65 if low else 80,
                "time_70_110_pct": tir, "time_below_70_pct": low,
                "time_above_110_pct": 100 - tir - low,
                "time_at_control_upper_bound_pct": 0,
            })
    return participants, pd.DataFrame(rows)


class ControlEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.participants, self.summary = example_data()

    def evaluate(self, summary=None):
        return evaluate_control_cohort(
            self.summary if summary is None else summary,
            self.participants, seed=17, bootstrap_resamples=2000,
        )

    def test_id_pairing_participant_weighting_and_worsening(self):
        result = self.evaluate(self.summary.sample(frac=1, random_state=51))
        overall = result.statistics.query("group == 'All participants'").set_index("metric")
        tir = overall.loc["time_70_110_pct"]
        # [10, -20, 100]: all individuals have equal weight despite unequal group size.
        self.assertAlmostEqual(tir.delta_mean, 30)
        self.assertEqual((tir.n_improved, tir.n_unchanged, tir.n_worsened), (2, 0, 1))
        self.assertAlmostEqual(tir.no_control_mean, 50)
        self.assertAlmostEqual(tir.optimized_mean, 80)
        diag = result.diagnostics.iloc[0]
        self.assertEqual(diag.n_participants, 3)
        self.assertEqual(diag.optimizer_successes, 2)
        self.assertEqual(diag.new_participants_below70, 1)

    def test_bootstrap_paired_constant_change_and_singleton(self):
        result = self.evaluate()
        objective = result.statistics.query("group == 'All participants' and metric == 'paper_objective'").iloc[0]
        self.assertAlmostEqual(objective.delta_mean, -20)
        self.assertAlmostEqual(objective.delta_ci95_low, -20)
        self.assertAlmostEqual(objective.delta_ci95_high, -20)
        singleton = result.statistics.query("group == 'Prediabetes'")
        self.assertTrue(singleton.delta_ci95_low.isna().all())
        pd.testing.assert_frame_equal(result.statistics, self.evaluate().statistics)

    def test_missing_duplicate_and_status_mismatch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "every fitted participant"):
            self.evaluate(self.summary.iloc[1:])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.evaluate(pd.concat([self.summary, self.summary.iloc[:1]]))
        broken = self.summary.copy()
        broken.loc[0, "status"] = "Prediabetes"
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            self.evaluate(broken)

    def test_nonfinite_values_and_bad_percentages_are_rejected(self):
        broken = self.summary.copy()
        broken.loc[0, "peak_glucose"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.evaluate(broken)
        broken = self.summary.copy()
        broken.loc[0, "time_70_110_pct"] = 99
        with self.assertRaisesRegex(ValueError, "sum to 100"):
            self.evaluate(broken)

    def test_exports_preserve_all_pairs_and_numeric_signs(self):
        evaluation = self.evaluate()
        with tempfile.TemporaryDirectory() as directory:
            write_control_tables(Path(directory), evaluation, {})
            root = Path(directory) / "tables"
            numeric = pd.read_csv(root / "control_cohort_statistics.csv")
            objective = numeric.query("group == 'All participants' and metric == 'paper_objective'").iloc[0]
            self.assertEqual(objective.delta_mean, -20)
            workbook = pd.read_excel(root / "control_evaluation.xlsx", sheet_name=None)
            self.assertEqual(len(workbook["Overall"]), 6)
            self.assertEqual(len(workbook["Paired changes"]), 3)
            self.assertIn("Better/same/worse", (root / "control_evaluation.tex").read_text())


if __name__ == "__main__":
    unittest.main()
