import unittest
from types import SimpleNamespace

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration_utils.iq_blobs.plotting import (
    plot_individual_histograms,
    plot_individual_iq_blobs,
    plot_iq_blobs_dashboard,
)
from calibration_utils.iq_blobs_ef.plotting import (
    plot_individual_iq_blobs as plot_individual_iq_blobs_ef,
)


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
        self.assertEqual(lines_by_label["Ground"].get_alpha(), 0.35)
        self.assertEqual(lines_by_label["Prepared"].get_alpha(), 0.35)
        center_points = [lines_by_label["Ground center"], lines_by_label["Prepared center"]]
        self.assertEqual(tuple(center_points[0].get_data()), ((-4.0,), (2.0,)))
        self.assertEqual(tuple(center_points[1].get_data()), ((6.0,), (-2.0,)))
        self.assertGreater(center_points[0].get_zorder(), lines_by_label["Ground"].get_zorder())
        self.assertGreater(center_points[1].get_zorder(), lines_by_label["Prepared"].get_zorder())
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

    def test_dashboard_title_includes_run_metadata(self):
        runs = np.arange(2)
        raw = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), [[-4e-3, -4e-3]]),
                "Qg": (("qubit", "n_runs"), [[2e-3, 2e-3]]),
                "Ie": (("qubit", "n_runs"), [[6e-3, 6e-3]]),
                "Qe": (("qubit", "n_runs"), [[-2e-3, -2e-3]]),
            },
            coords={"qubit": ["q1"], "n_runs": runs},
        )
        fit = xr.Dataset(
            {
                "Ig_rot": (("qubit", "n_runs"), [[-2e-3, -2e-3]]),
                "Ie_rot": (("qubit", "n_runs"), [[3e-3, 3e-3]]),
                "rus_threshold": ("qubit", [-2e-3]),
                "ge_threshold": ("qubit", [0.5e-3]),
                "gg": ("qubit", [0.9]),
                "ge": ("qubit", [0.1]),
                "eg": ("qubit", [0.2]),
                "ee": ("qubit", [0.8]),
                "success": ("qubit", [True]),
                "separation_to_width": ("qubit", [3.0]),
                "readout_fidelity": ("qubit", [85.0]),
                "iw_angle": ("qubit", [0.0]),
            },
            coords={"qubit": ["q1"], "n_runs": runs},
        )
        readout = SimpleNamespace(length=1200, amplitude=0.045)
        qubit = SimpleNamespace(
            name="q1",
            resonator=SimpleNamespace(operations={"readout": readout}),
        )

        fig = plot_iq_blobs_dashboard(
            raw,
            [qubit],
            fit,
            run_metadata={
                "operation": "readout",
                "reset_type": "active",
                "num_shots": 25000,
                "pi_repetitions": 3,
            },
        )

        parameter_text = next(text for text in fig.texts if text.get_gid() == "iq_blobs_parameters")
        self.assertEqual(fig._suptitle.get_text(), "IQ blobs calibration")
        self.assertIn("Parameters", parameter_text.get_text())
        self.assertIn("readout length=1200 ns", parameter_text.get_text())
        self.assertIn("readout amp=0.045 V", parameter_text.get_text())
        self.assertIn("active reset=True", parameter_text.get_text())
        self.assertIn("num reps=25000", parameter_text.get_text())
        self.assertIn("pi reps=3", parameter_text.get_text())

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

    def test_three_state_dashboard_draws_pairwise_threshold_lines(self):
        runs = np.arange(4)
        raw = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), [[-2e-3, -2e-3, -2e-3, -2e-3]]),
                "Qg": (("qubit", "n_runs"), [[0.0, 0.0, 0.0, 0.0]]),
                "Ie": (("qubit", "n_runs"), [[0.0, 0.0, 0.0, 0.0]]),
                "Qe": (("qubit", "n_runs"), [[0.0, 0.0, 0.0, 0.0]]),
                "If": (("qubit", "n_runs"), [[2e-3, 2e-3, 2e-3, 2e-3]]),
                "Qf": (("qubit", "n_runs"), [[0.0, 0.0, 0.0, 0.0]]),
            },
            coords={"qubit": ["q1"], "n_runs": runs},
        )
        fit = xr.Dataset(
            {
                "Ig_rot": (("qubit", "n_runs"), [[-2e-3, -2e-3, -2e-3, -2e-3]]),
                "Ie_rot": (("qubit", "n_runs"), [[0.0, 0.0, 0.0, 0.0]]),
                "If_rot": (("qubit", "n_runs"), [[2e-3, 2e-3, 2e-3, 2e-3]]),
                "rus_threshold": ("qubit", [-1e-3]),
                "ge_threshold": ("qubit", [0.0]),
                "success": ("qubit", [True]),
                "separation_to_width": ("qubit", [3.0]),
                "readout_fidelity": ("qubit", [95.0]),
                "iw_angle": ("qubit", [0.0]),
                "state_confusion_matrix": (
                    ("qubit", "prepared_state", "measured_state"),
                    [np.eye(3)],
                ),
                "threshold_line_midpoint": (
                    ("qubit", "threshold", "IQ"),
                    [[[-1e-3, 0.0], [1e-3, 0.0], [0.0, 0.0]]],
                ),
                "threshold_line_normal": (
                    ("qubit", "threshold", "IQ"),
                    [[[2e-3, 0.0], [2e-3, 0.0], [4e-3, 0.0]]],
                ),
            },
            coords={
                "qubit": ["q1"],
                "n_runs": runs,
                "prepared_state": ["g", "e", "f"],
                "measured_state": ["g", "e", "f"],
                "threshold": ["ge", "ef", "gf"],
                "IQ": ["I", "Q"],
            },
        )

        fig = plot_iq_blobs_dashboard(raw, [SimpleNamespace(name="q1")], fit)
        labels = {line.get_label() for line in fig.axes[0].lines}

        self.assertIn("GE threshold", labels)
        self.assertIn("EF threshold", labels)
        self.assertIn("GF threshold", labels)
        self.assertNotIn("RUS Threshold", labels)
        self.assertNotIn("Threshold", labels)

    def test_large_thresholds_do_not_expand_blob_or_histogram_limits(self):
        raw = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), [[-2e-3, -1e-3, 0.0]]),
                "Qg": (("qubit", "n_runs"), [[-1e-3, 0.0, 1e-3]]),
                "Ie": (("qubit", "n_runs"), [[1e-3, 2e-3, 3e-3]]),
                "Qe": (("qubit", "n_runs"), [[-1e-3, 0.0, 1e-3]]),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(3)},
        )
        fit = xr.Dataset(
            {
                "Ig_rot": (("qubit", "n_runs"), [[-2e-3, -1e-3, 0.0]]),
                "Ie_rot": (("qubit", "n_runs"), [[1e-3, 2e-3, 3e-3]]),
                "rus_threshold": ("qubit", [10.0]),
                "ge_threshold": ("qubit", [20.0]),
                "iw_angle": ("qubit", [0.0]),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(3)},
        )

        blob_figure, blob_axis = plt.subplots()
        plot_individual_iq_blobs(blob_axis, raw, {"qubit": "q1"}, fit.sel(qubit="q1"))
        histogram_figure, histogram_axis = plt.subplots()
        plot_individual_histograms(histogram_axis, raw, {"qubit": "q1"}, fit.sel(qubit="q1"))

        self.assertLess(blob_axis.get_xlim()[1], 10)
        self.assertLess(histogram_axis.get_xlim()[1], 10)
        plt.close(blob_figure)
        plt.close(histogram_figure)

    def test_gef_iq_blobs_draw_centers_above_transparent_points(self):
        runs = np.arange(3)
        fit = xr.Dataset(
            {
                "Ig": ("n_runs", [-2e-3, -1e-3, 0.0]),
                "Qg": ("n_runs", [0.0, 0.0, 0.0]),
                "Ie": ("n_runs", [1e-3, 2e-3, 3e-3]),
                "Qe": ("n_runs", [0.0, 0.0, 0.0]),
                "If": ("n_runs", [4e-3, 5e-3, 6e-3]),
                "Qf": ("n_runs", [0.0, 0.0, 0.0]),
                "I_g_center": -1e-3,
                "Q_g_center": 0.0,
                "I_e_center": 2e-3,
                "Q_e_center": 0.0,
                "I_f_center": 5e-3,
                "Q_f_center": 0.0,
            },
            coords={"n_runs": runs},
        )
        figure, axis = plt.subplots()

        plot_individual_iq_blobs_ef(axis, xr.Dataset(), {"qubit": "q1"}, fit)

        lines_by_label = {line.get_label(): line for line in axis.lines}
        self.assertEqual(lines_by_label["Ground"].get_alpha(), 0.15)
        self.assertEqual(lines_by_label["Excited"].get_alpha(), 0.15)
        self.assertEqual(lines_by_label["Second Excited"].get_alpha(), 0.15)
        self.assertGreater(lines_by_label["G"].get_zorder(), lines_by_label["Ground"].get_zorder())
        self.assertGreater(lines_by_label["E"].get_zorder(), lines_by_label["Excited"].get_zorder())
        self.assertGreater(
            lines_by_label["F"].get_zorder(),
            lines_by_label["Second Excited"].get_zorder(),
        )
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
