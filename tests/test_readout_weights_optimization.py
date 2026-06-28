import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from calibration_utils.readout_weights_optimization import (
    kernel_to_segments,
    process_sliced_traces,
    save_kernel_artifacts,
)


class ReadoutWeightsOptimizationTests(unittest.TestCase):
    def test_calibration_proposes_use_kernel_instead_of_overwriting_basic_weights(self):
        source = Path("calibrations_v2/10d_readout_weights_optimization.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(".readout.use_kernel", source)
        self.assertNotIn(".readout.integration_weights\"]", source)

    def test_process_sliced_traces_builds_profile_kernel(self):
        ds = xr.Dataset(
            {
                "IIg": (("qubit", "time_slice"), [[0.0, 0.0]]),
                "IQg": (("qubit", "time_slice"), [[0.0, 0.0]]),
                "QIg": (("qubit", "time_slice"), [[0.0, 0.0]]),
                "QQg": (("qubit", "time_slice"), [[0.0, 0.0]]),
                "IIe": (("qubit", "time_slice"), [[1.0, 0.0]]),
                "IQe": (("qubit", "time_slice"), [[0.0, 0.0]]),
                "QIe": (("qubit", "time_slice"), [[0.0, 0.5]]),
                "QQe": (("qubit", "time_slice"), [[0.0, 0.0]]),
            },
            coords={"qubit": ["q1"], "time_slice": [1, 2]},
        )

        analysed = process_sliced_traces(ds, slice_length_ns=40)

        np.testing.assert_allclose(analysed.profile_kernel.sel(qubit="q1"), [1.0, 0.0])
        np.testing.assert_allclose(analysed.time_ns, [40, 80])
        self.assertEqual(kernel_to_segments([1.0, -0.5], 40), [[1.0, 40], [-0.5, 40]])

    def test_save_kernel_artifacts_overwrites_one_file_per_qubit(self):
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "time_slice"), [[0.0, 0.0]]),
                "Qg": (("qubit", "time_slice"), [[0.0, 0.0]]),
                "Ie": (("qubit", "time_slice"), [[1.0, 0.0]]),
                "Qe": (("qubit", "time_slice"), [[0.0, 0.5]]),
                "ground_trace": (("qubit", "time_slice"), [[0.0 + 0.0j, 0.0 + 0.0j]]),
                "excited_trace": (("qubit", "time_slice"), [[1.0 + 0.0j, 0.0 + 0.5j]]),
                "subtracted_trace": (("qubit", "time_slice"), [[1.0 + 0.0j, 0.0 + 0.5j]]),
                "optimal_complex_trace": (("qubit", "time_slice"), [[1.0 + 0.0j, 0.0 + 0.5j]]),
                "profile_kernel": (("qubit", "time_slice"), [[1.0, 0.0]]),
            },
            coords={"qubit": ["q1"], "time_slice": [1, 2], "time_ns": ("time_slice", [40, 80])},
            attrs={"slice_length_ns": 40},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = save_kernel_artifacts(
                profile_name="main",
                experiment_name="10d_readout_weights_optimization",
                analysed=ds,
                parameters={"num_shots": 2},
                root=root,
                now=datetime(2026, 6, 19, tzinfo=timezone.utc),
            )

            self.assertEqual(output, root / "main" / "kernels")
            self.assertFalse((output / "10d_readout_weights_optimization").exists())
            self.assertTrue((output / "q1_readout_kernel.npz").is_file())
            with np.load(output / "q1_readout_kernel.npz") as kernel_file:
                metadata = json.loads(str(kernel_file["metadata_json"]))
                self.assertEqual(metadata["qubits"], ["q1"])

            ds["profile_kernel"] = (("qubit", "time_slice"), [[0.5, -0.5]])
            save_kernel_artifacts(
                profile_name="main",
                experiment_name="10d_readout_weights_optimization",
                analysed=ds,
                parameters={"num_shots": 3},
                root=root,
                now=datetime(2026, 6, 20, tzinfo=timezone.utc),
            )

            kernel_files = list((root / "main" / "kernels").glob("*_readout_kernel.npz"))
            self.assertEqual([path.name for path in kernel_files], ["q1_readout_kernel.npz"])
            with np.load(output / "q1_readout_kernel.npz") as kernel_file:
                np.testing.assert_allclose(kernel_file["profile_kernel"], [0.5, -0.5])


if __name__ == "__main__":
    unittest.main()
