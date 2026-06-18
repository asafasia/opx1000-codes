import unittest

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration_utils.readout_power_optimization.plotting import (
    plot_individual_data_with_fit,
    plot_raw_data_with_fit,
)


class ReadoutPowerPlottingTests(unittest.TestCase):
    def make_dataset_and_fit(self):
        amp_prefactors = [0.5, 1.0, 1.5]
        runs = np.arange(100)
        ground_i = np.tile(np.array([-0.3e-3, -0.5e-3, -0.8e-3])[:, None], (1, runs.size))
        excited_i = np.tile(np.array([0.4e-3, 0.9e-3, 1.1e-3])[:, None], (1, runs.size))
        zeros = np.zeros_like(ground_i)
        ds = xr.Dataset(
            {
                "I": (("amp_prefactor", "state", "n_runs"), np.stack([ground_i, excited_i], axis=1)),
                "Q": (("amp_prefactor", "state", "n_runs"), np.stack([zeros, zeros], axis=1)),
            },
            coords={
                "state": [0, 1],
                "n_runs": runs,
                "amp_prefactor": amp_prefactors,
                "qubit": "q1",
            },
        )
        fit = xr.Dataset(
            {
                "fit_data": (
                    ("amp_prefactor", "fit_vals"),
                    [[0.70, 0.95], [0.92, 0.90], [0.86, 0.76]],
                ),
                "optimal_amp": 1.0e-3,
                "best_fidelity": 0.92,
            },
            coords={
                "amp_prefactor": amp_prefactors,
                "fit_vals": ["meas_fidelity", "outliers"],
                "readout_amplitude": ("amp_prefactor", [0.5e-3, 1.0e-3, 1.5e-3]),
                "qubit": "q1",
            },
        )
        return ds, fit

    def test_plot_draws_error_bars_bands_and_optimum_marker(self):
        ds, fit = self.make_dataset_and_fit()
        figure, axis = plt.subplots()

        plot_individual_data_with_fit(axis, ds, {"qubit": "q1"}, fit)

        labels = axis.get_legend_handles_labels()[1]
        self.assertIn("Assignment fidelity", labels)
        self.assertIn("Non-outlier probability", labels)
        self.assertTrue(any(label.startswith("Optimal readout amplitude") for label in labels))
        self.assertTrue(any(label.startswith("Best fidelity") for label in labels))
        self.assertGreaterEqual(len(axis.collections), 4)
        self.assertEqual(axis.get_xlabel(), "Readout amplitude [mV]")
        self.assertEqual(axis.get_ylabel(), "Probability")
        self.assertEqual(axis.get_ylim(), (-0.03, 1.03))
        plt.close(figure)

    def test_full_plot_adds_separation_subplot_below_probability_panel(self):
        ds, fit = self.make_dataset_and_fit()
        figure = plot_raw_data_with_fit(
            ds.expand_dims(qubit=["q1"]),
            [type("Qubit", (), {"name": "q1", "grid_location": "0,0"})()],
            fit.expand_dims(qubit=["q1"]),
        )

        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(figure.axes[0].get_ylabel(), "Probability")
        self.assertEqual(figure.axes[1].get_ylabel(), "Separation [mV]")
        self.assertIn("IQ center separation", figure.axes[1].get_legend_handles_labels()[1])
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
