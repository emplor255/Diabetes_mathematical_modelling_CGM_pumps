import unittest

import numpy as np

from diabetes_paper.control import ControlConfig, simulate_piecewise_control
from diabetes_paper.model import (
    ModelConfig,
    derive_beta2,
    proposed_rhs,
    quick_insulin_equilibrium,
    simulate_meal,
    simulate_reduced_model,
)


class ProposedModelTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.parameters = np.array([1.2, 0.08, 0.02, 1.0, 0.8])
        self.reference = 100.0

    def test_derived_beta2_makes_reference_an_equilibrium(self):
        beta2 = derive_beta2(self.parameters, self.reference, self.config)
        quick = quick_insulin_equilibrium(
            self.reference, self.parameters, self.config
        )
        derivative = proposed_rhs(
            0.0,
            np.array([0.0, self.reference, quick]),
            self.parameters,
            beta2,
            self.config,
        )
        np.testing.assert_allclose(derivative, np.zeros(3), atol=1e-10)

    def test_zero_meal_stays_at_equilibrium(self):
        times = np.linspace(0.0, 2.5, 31)
        states = simulate_meal(
            times,
            self.parameters,
            meal_amplitude=0.0,
            absorption_delay_hours=0.25,
            initial_glucose=self.reference,
            reference_glucose=self.reference,
            config=self.config,
        )
        np.testing.assert_allclose(states[:, 1], self.reference, atol=1e-5)
        self.assertTrue(np.all(states[:, 0] >= 0))
        self.assertTrue(np.all(states[:, 1:] > 0))

    def test_meal_simulation_has_expected_shape(self):
        times = np.arange(0.0, 2.5 + 1e-9, 5.0 / 60.0)
        states = simulate_meal(
            times,
            self.parameters,
            meal_amplitude=120.0,
            absorption_delay_hours=0.3,
            initial_glucose=98.0,
            reference_glucose=self.reference,
            config=self.config,
        )
        self.assertEqual(states.shape, (len(times), 3))
        self.assertTrue(np.isfinite(states).all())
        self.assertGreater(states[:, 1].max(), states[0, 1])

    def test_exploration_overrides_are_explicit_and_defaults_unchanged(self):
        times = np.linspace(0, 2.5, 31)
        beta2 = derive_beta2(self.parameters, self.reference, self.config)
        quick0 = quick_insulin_equilibrium(self.reference, self.parameters, self.config)
        default = simulate_meal(times, self.parameters, 100, 0.3, self.reference, self.reference, self.config)
        explicit = simulate_meal(times, self.parameters, 100, 0.3, self.reference, self.reference, self.config,
                                 beta2_override=beta2, initial_quick_insulin_override=quick0)
        np.testing.assert_array_equal(default, explicit)
        higher_source = simulate_meal(times, self.parameters, 0, 0.3, self.reference, self.reference, self.config,
                                     beta2_override=1.5 * beta2, initial_quick_insulin_override=quick0)
        self.assertGreater(higher_source[-1, 1], self.reference)
        for kwargs in [{"beta2_override": -1}, {"beta2_override": np.nan},
                       {"initial_quick_insulin_override": -1}, {"initial_quick_insulin_override": np.inf}]:
            with self.assertRaises(ValueError):
                simulate_meal(times, self.parameters, 100, 0.3, self.reference, self.reference, **kwargs)


class AblationAndControlTests(unittest.TestCase):
    def test_reduced_model_zero_meal_relaxes_to_reference(self):
        times = np.linspace(0.0, 3.0, 50)
        glucose = simulate_reduced_model(
            times,
            alpha=1.0,
            clearance=0.5,
            meal_amplitude=0.0,
            absorption_delay_hours=0.2,
            initial_glucose=120.0,
            reference_glucose=100.0,
        )
        self.assertAlmostEqual(glucose[0], 120.0)
        self.assertLess(abs(glucose[-1] - 100.0), abs(glucose[0] - 100.0))

    def test_control_simulator_is_finite_and_positive(self):
        control_config = ControlConfig(
            horizon_hours=1.0,
            integration_step_minutes=2.0,
            control_bin_minutes=15.0,
        )
        controls = np.zeros(4)
        times, states, applied = simulate_piecewise_control(
            [1.0, 0.05, 0.01, 1.0, 0.8],
            reference_glucose=100.0,
            meal_amplitude=100.0,
            absorption_delay_hours=0.25,
            controls=controls,
            control_config=control_config,
        )
        self.assertEqual(len(times), len(states))
        self.assertEqual(len(times), len(applied))
        self.assertTrue(np.isfinite(states).all())
        self.assertTrue(np.all(states[:, 1] > 0))


if __name__ == "__main__":
    unittest.main()
