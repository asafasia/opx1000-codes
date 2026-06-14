import unittest
from types import SimpleNamespace

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")

from calibration_utils.iq_blobs.plotting import plot_iq_blobs_dashboard


class IQBlobsPlottingTests(unittest.TestCase):
    def test_dashboard_contains_all_three_views_and_correct_centers(self):
        runs = np.arange(4)
        raw = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), [[-4e-3, -4e-3, -4e-3, -4e-3]]),
                "Qg": (("qubit", "n_runs"), [[2e-3, 2e-3, 2e-3, 2e-3]]),
                "Ie": (("qubit", "n_runs"), [[6e-3, 6e-3, 6e-3, 6e-3]]),
                "Qe": (("qubit", "n_runs"), [[-2e-3, -2e-3, -2e-3, -2e-3]]),
            },
            coords={"qubit": ["q1"], "n_runs": runs},
        )
        fit = xr.Dataset(
            {
                "Ig_rot": (("qubit", "n_runs"), [[-2e-3, -2e-3, -2e-3, -2e-3]]),
                "Qg_rot": (("qubit", "n_runs"), [[1e-3, 1e-3, 1e-3, 1e-3]]),
                "Ie_rot": (("qubit", "n_runs"), [[3e-3, 3e-3, 3e-3, 3e-3]]),
                "Qe_rot": (("qubit", "n_runs"), [[-1e-3, -1e-3, -1e-3, -1e-3]]),
                "rus_threshold": ("qubit", [-2e-3]),
                "ge_threshold": ("qubit", [0.5e-3]),
                "gg": ("qubit", [0.9]),
                "ge": ("qubit", [0.1]),
                "eg": ("qubit", [0.2]),
                "ee": ("qubit", [0.8]),
                "success": ("qubit", [True]),
                "separation_to_width": ("qubit", [3.0]),
                "readout_fidelity": ("qubit", [85.0]),
                "iw_angle": ("qubit", [-np.pi / 4]),
            },
            coords={"qubit": ["q1"], "n_runs": runs},
        )

        fig = plot_iq_blobs_dashboard(raw, [SimpleNamespace(name="q1")], fit)

        self.assertEqual(len(fig.axes), 3)
        lines_by_label = {line.get_label(): line for line in fig.axes[0].lines}
        center_points = [lines_by_label["Ground center"], lines_by_label["Prepared center"]]
        self.assertEqual(tuple(center_points[0].get_data()), ((-4.0,), (2.0,)))
        self.assertEqual(tuple(center_points[1].get_data()), ((6.0,), (-2.0,)))
        self.assertIn("fitted rotation=-45.0 deg", fig.axes[0].get_title())
        threshold_line = lines_by_label["Threshold"]
        threshold_i, threshold_q = threshold_line.get_data()
        rotated_i = np.cos(-np.pi / 4) * threshold_i - np.sin(-np.pi / 4) * threshold_q
        np.testing.assert_allclose(rotated_i, 0.5)
        self.assertEqual(len(fig.axes[1].texts), 4)
        self.assertAlmostEqual(fig.axes[0].get_position().y0, fig.axes[1].get_position().y0)
        self.assertLess(fig.axes[2].get_position().y1, fig.axes[0].get_position().y0)
        self.assertLessEqual(fig.axes[2].get_position().x0, fig.axes[0].get_position().x0)
        self.assertGreaterEqual(fig.axes[2].get_position().x1, fig.axes[1].get_position().x1)

    def test_dashboard_draws_kde_contours(self):
        axis = np.linspace(-2e-3, 2e-3, 20)
        i_grid, q_grid = np.meshgrid(axis, axis)
        density = np.exp(-((i_grid / 1e-3) ** 2 + (q_grid / 1e-3) ** 2))
        raw = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), [[-1e-3, -0.5e-3, 0.0]]),
                "Qg": (("qubit", "n_runs"), [[0.0, 0.5e-3, -0.5e-3]]),
                "Ie": (("qubit", "n_runs"), [[0.0, 0.5e-3, 1e-3]]),
                "Qe": (("qubit", "n_runs"), [[0.5e-3, -0.5e-3, 0.0]]),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(3)},
        )
        fit = xr.Dataset(
            {
                "Ig_rot": (("qubit", "n_runs"), [[-1e-3, -0.5e-3, 0.0]]),
                "Qg_rot": (("qubit", "n_runs"), [[0.0, 0.5e-3, -0.5e-3]]),
                "Ie_rot": (("qubit", "n_runs"), [[0.0, 0.5e-3, 1e-3]]),
                "Qe_rot": (("qubit", "n_runs"), [[0.5e-3, -0.5e-3, 0.0]]),
                "rus_threshold": ("qubit", [-0.5e-3]),
                "ge_threshold": ("qubit", [0.0]),
                "gg": ("qubit", [0.9]),
                "ge": ("qubit", [0.1]),
                "eg": ("qubit", [0.2]),
                "ee": ("qubit", [0.8]),
                "success": ("qubit", [True]),
                "separation_to_width": ("qubit", [3.0]),
                "readout_fidelity": ("qubit", [85.0]),
                "iw_angle": ("qubit", [-np.pi / 4]),
                "ground_kde_I": (("qubit", "kde_y", "kde_x"), [i_grid]),
                "ground_kde_Q": (("qubit", "kde_y", "kde_x"), [q_grid]),
                "ground_kde_density": (("qubit", "kde_y", "kde_x"), [density]),
                "ground_kde_95_level": ("qubit", [0.2]),
                "prepared_kde_I": (("qubit", "kde_y", "kde_x"), [i_grid]),
                "prepared_kde_Q": (("qubit", "kde_y", "kde_x"), [q_grid]),
                "prepared_kde_density": (("qubit", "kde_y", "kde_x"), [density]),
                "prepared_kde_95_level": ("qubit", [0.2]),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(3)},
        )

        fig = plot_iq_blobs_dashboard(raw, [SimpleNamespace(name="q1")], fit)
        labels = {line.get_label() for line in fig.axes[0].lines}

        self.assertIn("Ground 95% KDE", labels)
        self.assertIn("Prepared 95% KDE", labels)


if __name__ == "__main__":
    unittest.main()
