import unittest
from types import SimpleNamespace

import numpy as np
import xarray as xr

from calibration_utils.iq_blobs.analysis import fit_raw_data


class IQBlobsAnalysisTests(unittest.TestCase):
    def make_node(self):
        qubit = SimpleNamespace(name="q1")
        return SimpleNamespace(namespace={"qubits": [qubit]})

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
