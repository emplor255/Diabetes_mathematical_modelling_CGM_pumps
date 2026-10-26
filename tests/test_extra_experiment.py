import sys
from pathlib import Path
import unittest
import tempfile
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extra_experiment.experiment import (
    ExperimentConfig, control_cost, grid_for, optimize_control, prepare_record, simulate, time_fraction,
)
from diabetes_paper.model import ModelConfig, simulate_meal


def frame():
    return pd.DataFrame({"Timestamp": ["06:00", "06:30", "07:00", "07:30"],
                         "BG": [100., 100., 120., 110.], "Eating": [0., 20., 0., 0.],
                         "Insulin Bolus": [0., 2., 0., 0.], "Noise": ["N", "P", "Y", "N"]})


class ExtraExperimentTests(unittest.TestCase):
    def setUp(self):
        self.record, self.audit = prepare_record(frame())
        self.parameters = np.array([1.2, .08, .02, 1., .8, 3., .5])

    def test_duplicates_preserve_inputs_and_mean_bg(self):
        raw = frame()
        duplicate = raw.iloc[[1]].copy()
        duplicate["Eating"] = 8.
        duplicate["BG"] = 102.
        duplicate["Insulin Bolus"] = .5
        combined = pd.concat([raw.iloc[:2], duplicate, raw.iloc[2:]], ignore_index=True)
        record, audit = prepare_record(combined)
        self.assertEqual(audit["duplicate_rows_collapsed"], 1)
        self.assertEqual(record.glucose.iloc[1], 101.)
        self.assertEqual(record.eating.sum(), 28.)
        self.assertEqual(record.bolus.sum(), 2.5)

    def test_reject_invalid_input_and_ambiguous_rollover(self):
        for column, value in [("BG", np.nan), ("BG", -2), ("Eating", -1),
                              ("Insulin Bolus", np.inf), ("Timestamp", "05:00")]:
            raw = frame()
            raw.loc[2, column] = value
            with self.assertRaises(ValueError):
                prepare_record(raw)

    def test_no_premeal_baseline_is_flagged_not_fabricated(self):
        raw = frame()
        raw.loc[0, "Eating"] = 5
        raw.loc[0, "BG"] = 65
        _, audit = prepare_record(raw)
        self.assertEqual(audit["reference_glucose"], 65.)
        self.assertEqual(audit["premeal_baseline_points"], 0)
        self.assertIn("no premeal", audit["reference_rule"])

    def test_no_input_equilibrium_and_original_rhs_agree(self):
        record = self.record.copy()
        record["eating"] = 0.
        record["bolus"] = 0.
        times = np.linspace(0, 1.5, 31)
        result = simulate(record, 100., self.parameters, times)
        np.testing.assert_allclose(result[:, 1], 100., atol=1e-6)
        record.loc[0, "eating"] = 20.
        result = simulate(record, 100., self.parameters, times)
        original = simulate_meal(times, self.parameters[:5], 60., 0., 100., 100., ModelConfig())
        # Original routine stores D(0) before its t=0 jump; the extension exposes
        # the right-continuous D(0). G/Q and all positive-time states must agree.
        np.testing.assert_allclose(result[:, 1:3], original[:, 1:3], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(result[1:, :3], original[1:], rtol=1e-4, atol=1e-4)

    def test_exact_event_jump_not_duplicated_and_bolus_depot_mass(self):
        eps = 1e-7
        times = np.array([0., .5 - eps, .5, .5 + eps, 1., 1.5])
        states = simulate(self.record, 100., self.parameters, times)
        self.assertAlmostEqual(states[2, 0] - states[1, 0], 60., places=4)
        self.assertAlmostEqual(states[2, 3], 1., places=6)
        self.assertAlmostEqual(states[-1, 3], np.exp(-1.), places=5)
        self.assertAlmostEqual(states[2, 1], states[1, 1], places=4)
        self.assertAlmostEqual(states[2, 2], states[1, 2], places=4)
        no_bolus = simulate(self.record, 100., self.parameters, times, bolus_multiplier=0.)
        self.assertLess(states[-1, 1], no_bolus[-1, 1])

    def test_control_validation_and_time_weighted_metrics(self):
        times = np.array([0., 1.5])
        with self.assertRaises(ValueError):
            simulate(self.record, 100., self.parameters, times, controls=[-1., 0.])
        with self.assertRaises(ValueError):
            simulate(self.record, 100., self.parameters, times, controls=[0.])
        self.assertAlmostEqual(time_fraction(np.array([0., 2.]), np.array([60., 100.]), 70, 90), 50.)
        self.assertEqual(time_fraction(times, [70., 70.], high=70., strict=True), 0.)
        cfg = ExperimentConfig()
        controls = np.array([2., 3.])
        base = control_cost(times, np.array([90., 90.]), np.zeros(2), cfg)
        cost = control_cost(times, np.array([90., 90.]), controls, cfg)
        self.assertAlmostEqual(cost - base, 4. + .5 * 9.)

    def test_output_grid_has_no_roundtrip_duplicate_times(self):
        times = grid_for(self.record)
        self.assertEqual(len(times), 91)
        self.assertTrue(np.all(np.diff(times * 60. / 60.) > 0))

    def test_zero_control_fallback(self):
        from types import SimpleNamespace
        failed = SimpleNamespace(fun=np.inf, x=np.ones(2), success=False, nit=1, message="test failure")
        with patch("extra_experiment.experiment.minimize", return_value=failed):
            summary, curves, bins, logs = optimize_control(self.record, 100., self.parameters)
        self.assertTrue(bins.unitless_control.eq(0.).all())
        self.assertEqual(summary.paper_objective.iloc[0], summary.paper_objective.iloc[1])
        self.assertTrue(summary.selected_control.eq("zero-control fallback").all())
        self.assertFalse(logs.success.any())
        self.assertFalse(summary.optimizer_success.any())

    def test_latex_export_has_no_optional_template_dependency(self):
        from extra_experiment.run_experiment import write_latex_table
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.tex"
            write_latex_table(pd.DataFrame({"Case": ["case_01"], "Below (%)": [2.5]}),
                              path, "Example", "tab:example")
            content = path.read_text()
            self.assertIn(r"case\_01", content)
            self.assertIn(r"Below (\%)", content)
            self.assertIn(r"\bottomrule", content)


@unittest.skipUnless((ROOT / "extra_experiment/results/fit_summary.csv").exists(), "Run extra experiment first")
class SavedExtraExperimentTests(unittest.TestCase):
    def test_saved_metrics_inputs_and_comparators(self):
        output = ROOT / "extra_experiment/results"
        audits = json.loads((output / "data_audit.json").read_text())
        fits = pd.read_csv(output / "fit_summary.csv")
        control = pd.read_csv(output / "control_summary.csv")
        self.assertEqual(set(fits.case), {"case_01", "case_02"})
        for row in fits.itertuples():
            audit = audits[row.case]
            path = ROOT / audit["local_input"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), audit["sha256"])
            pred = pd.read_csv(output / f"{row.case}_predictions.csv")
            error = pred.fitted_glucose - pred.observed_glucose
            self.assertAlmostEqual(float(np.sqrt(np.mean(error**2))), row.rmse, places=8)
            self.assertEqual(len(pred), audit["unique_times"])
            curves = pd.read_csv(output / f"{row.case}_trajectories.csv")
            baseline = curves.loc[curves.scenario.eq("Recorded boluses only")]
            common = pd.merge_asof(pred.sort_values("time_min"), baseline.sort_values("time_min"),
                                   on="time_min", direction="nearest", tolerance=1e-7)
            np.testing.assert_allclose(common.fitted_glucose, common.glucose, atol=1e-6)
            pair = control.loc[control.case.eq(row.case)]
            self.assertLessEqual(pair.paper_objective.iloc[1], pair.paper_objective.iloc[0] + 1e-7)
            for scenario in curves.scenario.unique():
                times = curves.loc[curves.scenario.eq(scenario), "time_min"].to_numpy()
                self.assertTrue(np.all(np.diff(times) > 0))
        numerical = pd.read_csv(output / "numerical_check.csv")
        self.assertLess(numerical.max_BG_difference_mg_dl.max(), .05)


if __name__ == "__main__":
    unittest.main()
