import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from calibration_utils.resonator_spectroscopy.analysis import (
    _extract_relevant_fit_parameters,
    calculate_iq_separation,
)


class ResonatorSpectroscopyAnalysisTests(unittest.TestCase):
    def test_separation_is_center_distance_divided_by_pooled_width(self):
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs", "detuning"), [[[-1.0], [1.0]]]),
                "Qg": (("qubit", "n_runs", "detuning"), [[[0.0], [0.0]]]),
                "Im": (("qubit", "n_runs", "detuning"), [[[3.0], [5.0]]]),
                "Qm": (("qubit", "n_runs", "detuning"), [[[0.0], [0.0]]]),
            },
            coords={"qubit": ["q9"], "n_runs": [0, 1], "detuning": [0]},
        )

        separation = calculate_iq_separation(ds)

        self.assertAlmostEqual(float(separation.sel(qubit="q9", detuning=0)), 4.0)

    def test_zero_width_returns_nan(self):
        values = np.ones((1, 2, 1))
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs", "detuning"), values),
                "Qg": (("qubit", "n_runs", "detuning"), values),
                "Im": (("qubit", "n_runs", "detuning"), values * 2),
                "Qm": (("qubit", "n_runs", "detuning"), values * 2),
            },
            coords={"qubit": ["q9"], "n_runs": [0, 1], "detuning": [0]},
        )

        self.assertTrue(np.isnan(calculate_iq_separation(ds)).all())

    def test_acquisition_preserves_shots_without_stream_averaging(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "calibrations"
            / "02a_resonator_spectroscopy.py"
        ).read_text(encoding="utf-8")
        stream_processing = source.split("with stream_processing():", 1)[1]

        self.assertNotIn(".average()", stream_processing)
        self.assertIn('.buffer(len(dfs)).buffer(n_runs).save(f"Ig{i + 1}")', stream_processing)
        self.assertIn('"n_runs": xr.DataArray(np.arange(n_runs)', source)

    def test_selected_frequency_follows_maximum_separation(self):
        fit = xr.Dataset(
            {
                "position": ("qubit", [0.0]),
                "width": ("qubit", [1e6]),
            },
            coords={"qubit": ["q9"]},
        )
        spectroscopy_data = xr.Dataset(
            {
                "IQ_separation": (("qubit", "detuning"), [[1.0, 4.0, 2.0]]),
            },
            coords={"qubit": ["q9"], "detuning": [-1e6, 2e6, 5e6]},
        )
        qubit = type(
            "Qubit",
            (),
            {"resonator": type("Resonator", (), {"RF_frequency": 7.47e9})()},
        )()
        node = type(
            "Node",
            (),
            {
                "namespace": {"qubits": [qubit]},
                "parameters": type("Parameters", (), {"frequency_span_in_mhz": 30.0})(),
            },
        )()

        _, results = _extract_relevant_fit_parameters(fit, spectroscopy_data, node)

        self.assertEqual(results["q9"].frequency, 7.472e9)


if __name__ == "__main__":
    unittest.main()
