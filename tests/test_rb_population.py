import unittest
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.single_qubit_randomized_benchmarking.analysis import fit_raw_data
from calibration_utils.single_qubit_randomized_benchmarking.plotting import plot_individual_data_with_fit


class TestRbPopulation(unittest.TestCase):
    def test_analysis_defines_ground_population_as_one_minus_state(self):
        state = np.array([[[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]])
        ds = xr.Dataset(
            {"state": (("qubit", "depths", "nb_of_sequences"), state)},
            coords={"qubit": ["q1"], "depths": [1, 2, 3], "nb_of_sequences": [0, 1]},
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(use_state_discrimination=True),
            outcomes={},
        )

        fake_fit = xr.DataArray(
            np.zeros((1, 3)),
            dims=("qubit", "fit_vals"),
            coords={"qubit": ["q1"], "fit_vals": ["a", "offset", "decay"]},
        )
        with (
            patch(
                "calibration_utils.single_qubit_randomized_benchmarking.analysis.fit_decay_exp",
                return_value=fake_fit,
            ),
            patch(
                "calibration_utils.single_qubit_randomized_benchmarking.analysis._extract_relevant_fit_parameters",
                side_effect=lambda fit, node: (fit, {}),
            ),
        ):
            fit, _ = fit_raw_data(ds, node)

        np.testing.assert_array_equal(fit.population, 1 - state)
        np.testing.assert_array_equal(fit.averaged_data, (1 - state).mean(axis=2))

    def test_plot_uses_ground_population_without_an_extra_inversion(self):
        population = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])
        fit = xr.Dataset(
            {
                "population": (("depths", "nb_of_sequences"), population),
                "averaged_data": ("depths", population.mean(axis=1)),
                "fit_data": ("fit_vals", [0.5, 0.4, -0.1]),
                "error_per_gate": xr.DataArray(0.01),
            },
            coords={
                "depths": [1, 2, 3],
                "nb_of_sequences": [0, 1],
                "fit_vals": ["a", "offset", "decay"],
            },
        )
        figure, axis = plt.subplots()

        plot_individual_data_with_fit(axis, fit, {"qubit": "q1"}, fit)

        plotted_population = np.asarray(axis.lines[0].get_ydata(), dtype=float)
        np.testing.assert_allclose(plotted_population, population.mean(axis=1))
        self.assertEqual(axis.get_ylabel(), "Ground-state population")
        plt.close(figure)

    def test_failed_decay_fit_preserves_population_and_marks_failure(self):
        state = np.zeros((1, 4, 2))
        ds = xr.Dataset(
            {"state": (("qubit", "depths", "nb_of_sequences"), state)},
            coords={"qubit": ["q1"], "depths": [1, 2, 3, 4], "nb_of_sequences": [0, 1]},
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(use_state_discrimination=True),
            outcomes={},
            log=lambda message: None,
        )

        with patch(
            "calibration_utils.single_qubit_randomized_benchmarking.analysis.fit_decay_exp",
            side_effect=AttributeError("fit returned None"),
        ):
            fit, results = fit_raw_data(ds, node)

        self.assertIn("population", fit)
        self.assertTrue(np.isnan(fit.fit_data).all())
        self.assertFalse(results["q1"].success)


if __name__ == "__main__":
    unittest.main()
