import unittest
from types import SimpleNamespace

import numpy as np
import xarray as xr

from calibration_utils.readout_gef_frequency_optimization.analysis import fit_raw_data


class GEFReadoutFrequencyAnalysisTests(unittest.TestCase):
    def test_uses_resolved_qubits_when_parameter_qubits_is_none(self):
        frequency = np.arange(5)
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "frequency"), [[0, 0, 0, 0, 0]]),
                "Qg": (("qubit", "frequency"), [[0, 0, 0, 0, 0]]),
                "Ie": (("qubit", "frequency"), [[1, 2, 3, 2, 1]]),
                "Qe": (("qubit", "frequency"), [[0, 0, 0, 0, 0]]),
                "If": (("qubit", "frequency"), [[0, 0, 0, 0, 0]]),
                "Qf": (("qubit", "frequency"), [[1, 2, 3, 2, 1]]),
            },
            coords={"qubit": ["q1"], "frequency": frequency},
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(qubits=None),
            namespace={"qubits": [SimpleNamespace(name="q1")]},
        )

        _, results = fit_raw_data(ds, node)

        self.assertIn("q1", results)
        self.assertTrue(results["q1"].success)


if __name__ == "__main__":
    unittest.main()
