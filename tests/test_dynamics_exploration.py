import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from diabetes_paper.dynamics_exploration import (
    SWEEP_PARAMETERS, SweepConfig, fit_performance_by_id, no_meal_equilibrium,
    run_parameter_sweep, select_reference, write_exploration_artifacts,
)
from diabetes_paper.fit_evaluation import evaluate_fits
from diabetes_paper.model import (
    PARAMETER_NAMES, ModelConfig, derive_beta2, proposed_rhs,
    quick_insulin_equilibrium, simulate_meal,
)


def reference_participants():
    model = ModelConfig()
    coefficients = np.array([1.2, 0.08, 0.02, 1.0, 0.8])
    values = {name: value for name, value in zip(PARAMETER_NAMES, coefficients)}
    return pd.DataFrame([
        {"userID": user, "status": status, "rmse_proposed": rmse,
         "reference_glucose": 100.0, "median_meal_amplitude": 120.0,
         "median_absorption_delay_min": 18.0,
         "beta2_derived": derive_beta2(coefficients, 100, model), **values}
        for user, status, rmse in [("ID-10", "Healthy", 8.0), ("ID-2", "Prediabetes", 10.0)]
    ])


class ParameterExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.participants = reference_participants()
        cls.config = SweepConfig(factors=(0.5, 1.0, 1.5), horizon_hours=6.0, step_minutes=5.0)
        cls.trajectories, cls.summary, cls.metadata = run_parameter_sweep(cls.participants, sweep=cls.config)

    def test_reference_selection_and_natural_id_order(self):
        selected = select_reference(self.participants)
        self.assertEqual(selected.userID, "ID-2")  # Natural ID resolves the symmetric RMSE tie.
        self.assertEqual(select_reference(self.participants, "ID-10").userID, "ID-10")
        with self.assertRaisesRegex(ValueError, "not in the fitted cohort"):
            select_reference(self.participants, "missing")
        rows = []
        for user, error in [("ID-10", 99.0), ("ID-2", 3.0), ("ID-1", 4.0)]:
            for model in ["Proposed 3-state", "Reduced 2-state"]:
                rows.append({"userID": user, "status": "Healthy", "n_events": 2,
                             "n_points": 50, "rmse": error, "mae": error / 2, "model": model})
        table = fit_performance_by_id(pd.DataFrame(rows).sample(frac=1, random_state=11))
        self.assertEqual(table.userID.tolist(), ["ID-1", "ID-2", "ID-10"])
        self.assertEqual(table.id_order.tolist(), [1, 2, 3])
        self.assertEqual(table.rmse_proposed.max(), 99)  # No exclusion of unfavorable fits.

    def test_only_one_coefficient_varies_and_all_initial_states_are_fixed(self):
        self.assertEqual(len(self.summary), 18)
        nominal = self.metadata["nominal_coefficients"]
        for row in self.summary.to_dict("records"):
            for name in SWEEP_PARAMETERS:
                expected = nominal[name] * (row["factor"] if name == row["varied_parameter"] else 1)
                self.assertAlmostEqual(row[f"value_{name}"], expected)
        initial = self.trajectories.loc[self.trajectories.time_min.eq(0)]
        self.assertEqual(len(initial), 18)
        np.testing.assert_allclose(initial[["digestive_glucose", "glucose", "quick_insulin"]],
                                   np.tile(self.metadata["initial_state_before_absorption"], (18, 1)))
        self.assertEqual(self.summary.meal_amplitude.nunique(), 1)
        self.assertEqual(self.summary.absorption_delay_min.nunique(), 1)
        self.assertEqual(self.trajectories.time_min.max(), 360.0)
        self.assertIn(18.0, self.trajectories.time_min.unique())
        beta2_fixed = self.summary.loc[self.summary.varied_parameter.ne("beta2"), "value_beta2"]
        self.assertEqual(beta2_fixed.nunique(), 1)

    def test_nominal_curves_identical_and_match_default_simulator(self):
        nominal = self.trajectories.loc[self.trajectories.factor.eq(1)].sort_values("time_min")
        baseline = nominal.loc[nominal.varied_parameter.eq("alpha")]
        for _, group in nominal.groupby("varied_parameter"):
            np.testing.assert_array_equal(group.glucose.to_numpy(), baseline.glucose.to_numpy())
        reference = select_reference(self.participants)
        direct = simulate_meal(
            baseline.time_min.to_numpy() / 60, reference[list(PARAMETER_NAMES)].to_numpy(float),
            reference.median_meal_amplitude, reference.median_absorption_delay_min / 60,
            reference.reference_glucose, reference.reference_glucose,
        )
        np.testing.assert_array_equal(direct[:, 1], baseline.glucose.to_numpy())

    def test_equilibrium_jacobian_matches_finite_differences(self):
        reference = select_reference(self.participants)
        coefficients = reference[list(PARAMETER_NAMES)].to_numpy(float)
        model = ModelConfig()
        equilibrium = no_meal_equilibrium(coefficients, reference.beta2_derived, model)
        state = np.array([0, equilibrium["equilibrium_glucose_mg_dl"], equilibrium["equilibrium_quick_insulin"]])
        jacobian = np.empty((3, 3))
        for index in range(3):
            step = np.zeros(3)
            step[index] = 1e-5
            jacobian[:, index] = (proposed_rhs(0, state + step, coefficients, reference.beta2_derived, model)
                                  - proposed_rhs(0, state - step, coefficients, reference.beta2_derived, model)) / (2e-5)
        actual = np.sort_complex(np.linalg.eigvals(jacobian))
        reported = np.sort_complex(np.array([complex(equilibrium[f"eigenvalue_{i}_real_per_hour"],
                                                     equilibrium[f"eigenvalue_{i}_imag_per_hour"]) for i in range(1, 4)]))
        np.testing.assert_allclose(actual, reported, rtol=1e-7, atol=1e-8)
        self.assertAlmostEqual(equilibrium["equilibrium_glucose_mg_dl"], 100)
        self.assertLess(self.summary.equilibrium_rhs_max_abs.max(), 1e-8)
        self.assertTrue(self.summary.locally_asymptotically_stable.all())

    def test_invalid_scenarios_and_mismatched_model_rejected(self):
        for factors in [(0, 1), (1, 1), (0.5, 1.5), (np.nan, 1)]:
            with self.assertRaises(ValueError):
                run_parameter_sweep(self.participants, sweep=SweepConfig(factors=factors))
        with self.assertRaisesRegex(ValueError, "beyond the absorption delay"):
            run_parameter_sweep(self.participants, sweep=SweepConfig(horizon_hours=0.1))
        with self.assertRaisesRegex(ValueError, "configuration disagree"):
            run_parameter_sweep(self.participants, model=ModelConfig(k_half_mg_dl=150))


class SavedExplorationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = Path(__file__).resolve().parents[1] / "outputs/tables"
        if not (cls.tables / "participant_parameters.csv").is_file():
            raise unittest.SkipTest("Saved fitted results are not available")
        cls.participants = pd.read_csv(cls.tables / "participant_parameters.csv")
        cls.predictions = pd.read_csv(cls.tables / "model_predictions.csv")
        cls.comparisons = pd.read_csv(cls.tables / "model_comparison.csv")
        cls.audit = json.loads((cls.tables / "data_audit.json").read_text())
        cls.model = ModelConfig(**cls.audit["fit_config"]["model"])

    def test_exported_figures_and_original_inputs_preserved(self):
        p_before = self.participants.copy(deep=True)
        predictions_before = self.predictions.copy(deep=True)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            table, summary, metadata = write_exploration_artifacts(
                output, self.participants, self.predictions, self.comparisons,
                sweep=SweepConfig(horizon_hours=4, step_minutes=3), model=self.model,
            )
            self.assertEqual(len(table), len(self.participants))
            self.assertEqual(table.n_observations.sum(), len(self.predictions))
            self.assertEqual(len(summary), 30)
            self.assertTrue(np.isfinite(summary.peak_glucose_mg_dl).all())
            for filename in ["figure_8_fit_by_id.png", "figure_8_fit_by_id.pdf",
                             "figure_9_parameter_sweep.png", "figure_9_parameter_sweep.pdf"]:
                self.assertGreater((output / "figures" / filename).stat().st_size, 0)
            config = json.loads((output / "tables/parameter_sweep_config.json").read_text())
            self.assertEqual(config["reference_id"], metadata["reference_id"])
            self.assertIn("fixed", config["beta2_policy"])
            caption = (output / "manuscript/fit_by_id_and_dynamics.tex").read_text()
            self.assertIn(metadata["reference_id"], caption)
            self.assertIn("unvalidated extrapolation", caption)
            pd.testing.assert_frame_equal(p_before, self.participants)
            pd.testing.assert_frame_equal(predictions_before, self.predictions)

    def test_default_simulator_still_reproduces_saved_reference_meals(self):
        participant = select_reference(self.participants)
        events = pd.read_csv(self.tables / "event_fit_metrics.csv")
        events = events.loc[events.userID.eq(participant.userID)]
        prepared = pd.read_csv(self.tables / "meal_points_prepared.csv")
        for event in events.itertuples():
            points = self.predictions.loc[self.predictions.event_id.eq(event.event_id)].sort_values("time_min")
            observations = prepared.loc[prepared.event_id.eq(event.event_id)].sort_values("relative_min")
            initial_glucose = np.interp(0.0, observations.relative_min, observations.glucose)
            states = simulate_meal(points.time_min.to_numpy() / 60,
                                   participant[list(PARAMETER_NAMES)].to_numpy(float),
                                   event.meal_amplitude, event.absorption_delay_min / 60,
                                   initial_glucose, participant.reference_glucose, self.model)
            np.testing.assert_allclose(states[:, 1], points.proposed_glucose, rtol=1e-7, atol=1e-6)

