import unittest
import json
import tempfile
from types import SimpleNamespace

import numpy as np
import xarray as xr

from calibration_utils.iq_blobs.analysis import (
    READOUT_FIDELITY_SUCCESS_THRESHOLD_PERCENT,
    SEPARATION_TO_WIDTH_SUCCESS_THRESHOLD,
    fit_raw_data,
    save_fit_results,
)


class IQBlobsAnalysisTests(unittest.TestCase):
    def make_node(self):
        qubit = SimpleNamespace(name="q1")
        return SimpleNamespace(namespace={"qubits": [qubit]})

    def test_success_threshold_allows_57_percent_readout_fidelity(self):
        self.assertEqual(READOUT_FIDELITY_SUCCESS_THRESHOLD_PERCENT, 57)
        self.assertEqual(SEPARATION_TO_WIDTH_SUCCESS_THRESHOLD, 0.1)

    def test_overlapping_clouds_fail_calibration(self):
        rng = np.random.default_rng(4)
        noise = rng.normal(0, 1e-5, (1, 2000))
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), noise),
                "Qg": (("qubit", "n_runs"), rng.normal(0, 1e-5, (1, 2000))),
                "Ie": (("qubit", "n_runs"), noise.copy()),
                "Qe": (("qubit", "n_runs"), rng.normal(0, 1e-5, (1, 2000))),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(2000)},
        )

        fit, results = fit_raw_data(ds, self.make_node())

        self.assertFalse(results["q1"].success)
        self.assertLess(float(fit.separation_to_width.sel(qubit="q1")), 1)

    def test_well_separated_clouds_pass_calibration(self):
        rng = np.random.default_rng(5)
        ground_I = rng.normal(-5e-5, 5e-6, (1, 2000))
        excited_I = rng.normal(5e-5, 5e-6, (1, 2000))
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), ground_I),
                "Qg": (("qubit", "n_runs"), rng.normal(0, 5e-6, (1, 2000))),
                "Ie": (("qubit", "n_runs"), excited_I),
                "Qe": (("qubit", "n_runs"), rng.normal(0, 5e-6, (1, 2000))),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(2000)},
        )

        _, results = fit_raw_data(ds, self.make_node())

        self.assertTrue(results["q1"].success)
        self.assertEqual(results["q1"].rus_threshold, results["q1"].ge_threshold)

    def test_fit_results_include_threshold_fidelity_matrix_and_average(self):
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), [[-3.0, -2.0, 0.2, 2.5]]),
                "Qg": (("qubit", "n_runs"), [[0.0, 0.0, 0.0, 0.0]]),
                "Ie": (("qubit", "n_runs"), [[-0.1, 0.4, 2.0, 3.0]]),
                "Qe": (("qubit", "n_runs"), [[0.0, 0.0, 0.0, 0.0]]),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(4)},
        )

        fit, results = fit_raw_data(ds, self.make_node())

        expected_matrix = np.asarray([[0.5, 0.5], [0.0, 1.0]])
        np.testing.assert_allclose(fit.fidelity_matrix.sel(qubit="q1").values, expected_matrix)
        np.testing.assert_allclose(results["q1"].fidelity_matrix, expected_matrix)
        self.assertAlmostEqual(results["q1"].average_fidelity, 75.0)
        self.assertAlmostEqual(results["q1"].readout_fidelity, results["q1"].average_fidelity)
        expected_matrix_std = np.sqrt(expected_matrix * (1 - expected_matrix) / 4)
        expected_fidelity_std = 50 * np.sqrt(
            expected_matrix_std[0, 0] ** 2 + expected_matrix_std[1, 1] ** 2
        )
        np.testing.assert_allclose(
            fit.fidelity_matrix_std.sel(qubit="q1").values,
            expected_matrix_std,
        )
        np.testing.assert_allclose(results["q1"].fidelity_matrix_std, expected_matrix_std)
        self.assertAlmostEqual(results["q1"].readout_fidelity_std, expected_fidelity_std)
        self.assertAlmostEqual(results["q1"].average_fidelity_std, expected_fidelity_std)

    def test_save_fit_results_writes_average_and_matrix(self):
        fit_results = {
            "q1": {
                "average_fidelity": 75.0,
                "readout_fidelity": 75.0,
                "fidelity_matrix": [[0.5, 0.5], [0.0, 1.0]],
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            output_path = save_fit_results(directory, fit_results)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["q1"]["average_fidelity"], 75.0)
        self.assertEqual(saved["q1"]["fidelity_matrix"], [[0.5, 0.5], [0.0, 1.0]])

    def test_three_state_clouds_add_centers_and_confusion_matrix(self):
        rng = np.random.default_rng(9)
        runs = 1000
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), rng.normal(-6e-5, 2e-6, (1, runs))),
                "Qg": (("qubit", "n_runs"), rng.normal(0, 2e-6, (1, runs))),
                "Ie": (("qubit", "n_runs"), rng.normal(0, 2e-6, (1, runs))),
                "Qe": (("qubit", "n_runs"), rng.normal(0, 2e-6, (1, runs))),
                "If": (("qubit", "n_runs"), rng.normal(6e-5, 2e-6, (1, runs))),
                "Qf": (("qubit", "n_runs"), rng.normal(0, 2e-6, (1, runs))),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(runs)},
        )

        fit, results = fit_raw_data(ds, self.make_node())

        self.assertEqual(results["q1"].state_labels, ["g", "e", "f"])
        self.assertEqual(np.asarray(results["q1"].center_matrix).shape, (3, 2))
        self.assertEqual(np.asarray(results["q1"].confusion_matrix).shape, (3, 3))
        self.assertEqual(results["q1"].threshold_pairs, ["ge", "ef", "gf"])
        self.assertEqual(np.asarray(results["q1"].threshold_line_midpoints).shape, (3, 2))
        self.assertEqual(np.asarray(results["q1"].threshold_line_normals).shape, (3, 2))
        self.assertEqual(np.asarray(results["q1"].confusion_matrix_std).shape, (3, 3))
        centers = np.asarray(results["q1"].center_matrix)
        midpoints = np.asarray(results["q1"].threshold_line_midpoints)
        np.testing.assert_allclose(midpoints[0], 0.5 * (centers[0] + centers[1]))
        np.testing.assert_allclose(midpoints[1], 0.5 * (centers[1] + centers[2]))
        np.testing.assert_allclose(midpoints[2], 0.5 * (centers[0] + centers[2]))
        np.testing.assert_allclose(
            fit.state_confusion_matrix.sel(qubit="q1").values,
            np.eye(3),
            atol=0.02,
        )

    def test_kde_regions_enclose_95_percent_of_each_blob(self):
        rng = np.random.default_rng(7)
        runs = 500
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), rng.normal(-5e-5, 7e-6, (1, runs))),
                "Qg": (("qubit", "n_runs"), rng.normal(0, 4e-6, (1, runs))),
                "Ie": (("qubit", "n_runs"), rng.normal(5e-5, 6e-6, (1, runs))),
                "Qe": (("qubit", "n_runs"), rng.normal(0, 5e-6, (1, runs))),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(runs)},
        )

        fit, _ = fit_raw_data(ds, self.make_node())

        for state in ("ground", "prepared"):
            level = float(fit[f"{state}_kde_95_level"].sel(qubit="q1"))
            fraction = float(fit[f"{state}_kde_enclosed_fraction"].sel(qubit="q1"))
            self.assertTrue(np.isfinite(level))
            self.assertAlmostEqual(fraction, 0.95, delta=0.01)

    def test_kde_regions_use_acquired_iq_coordinates(self):
        rng = np.random.default_rng(8)
        runs = 500
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), rng.normal(-4e-5, 3e-6, (1, runs))),
                "Qg": (("qubit", "n_runs"), rng.normal(-4e-5, 3e-6, (1, runs))),
                "Ie": (("qubit", "n_runs"), rng.normal(4e-5, 3e-6, (1, runs))),
                "Qe": (("qubit", "n_runs"), rng.normal(4e-5, 3e-6, (1, runs))),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(runs)},
        )

        fit, _ = fit_raw_data(ds, self.make_node())

        ground_i_grid = fit.ground_kde_I.sel(qubit="q1").values
        ground_q_grid = fit.ground_kde_Q.sel(qubit="q1").values
        self.assertLess(np.nanmean(ground_i_grid), 0)
        self.assertLess(np.nanmean(ground_q_grid), 0)

    def test_degenerate_blob_does_not_break_kde_analysis(self):
        runs = 100
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), np.full((1, runs), -5e-5)),
                "Qg": (("qubit", "n_runs"), np.zeros((1, runs))),
                "Ie": (("qubit", "n_runs"), np.full((1, runs), 5e-5)),
                "Qe": (("qubit", "n_runs"), np.zeros((1, runs))),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(runs)},
        )

        fit, _ = fit_raw_data(ds, self.make_node())

        self.assertTrue(np.isnan(float(fit.ground_kde_95_level.sel(qubit="q1"))))
        self.assertTrue(np.isnan(float(fit.prepared_kde_95_level.sel(qubit="q1"))))

    def test_rotation_aligns_blob_means_with_positive_i_axis(self):
        rng = np.random.default_rng(6)
        ground_i = rng.normal(-4e-5, 2e-6, (1, 2000))
        ground_q = rng.normal(-4e-5, 2e-6, (1, 2000))
        excited_i = rng.normal(4e-5, 2e-6, (1, 2000))
        excited_q = rng.normal(4e-5, 2e-6, (1, 2000))
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs"), ground_i),
                "Qg": (("qubit", "n_runs"), ground_q),
                "Ie": (("qubit", "n_runs"), excited_i),
                "Qe": (("qubit", "n_runs"), excited_q),
            },
            coords={"qubit": ["q1"], "n_runs": np.arange(2000)},
        )

        fit, results = fit_raw_data(ds, self.make_node())

        self.assertAlmostEqual(results["q1"].iw_angle, -np.pi / 4, places=2)
        self.assertAlmostEqual(
            float((fit.Qe_rot.mean(dim="n_runs") - fit.Qg_rot.mean(dim="n_runs")).sel(qubit="q1")),
            0.0,
            places=7,
        )
        self.assertGreater(
            float((fit.Ie_rot.mean(dim="n_runs") - fit.Ig_rot.mean(dim="n_runs")).sel(qubit="q1")),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
