from types import SimpleNamespace
import unittest

import numpy as np
import xarray as xr

from calibration_utils.qubit_spectroscopy.analysis import (
    _extract_relevant_fit_parameters,
    fit_raw_data,
)


class QubitSpectroscopyAnalysisTests(unittest.TestCase):
    def make_node(self):
        qubit = SimpleNamespace(
            name="q1",
            xy=SimpleNamespace(
                RF_frequency=4.5e9,
                operations={
                    "saturation": SimpleNamespace(amplitude=0.2),
                    "x180": SimpleNamespace(length=40),
                },
            ),
            resonator=SimpleNamespace(
                operations={"readout": SimpleNamespace(integration_weights_angle=0.0)}
            ),
        )
        return SimpleNamespace(
            name="03a_qubit_spectroscopy",
            namespace={"qubits": [qubit]},
            parameters=SimpleNamespace(
                frequency_span_in_mhz=500,
                frequency_step_in_mhz=0.5,
                operation_amplitude_factor=1,
                target_peak_width=3e6,
                use_state_discrimination=True,
            ),
        )

    def test_large_suggested_saturation_amplitude_does_not_fail_valid_peak(self):
        node = self.make_node()
        node.parameters.use_state_discrimination = False
        fit = xr.Dataset(
            {
                "position": ("qubit", [0.0]),
                "width": ("qubit", [1.0]),
                "iw_angle": ("qubit", [0.0]),
            },
            coords={"qubit": ["q1"]},
        )

        fit_data, fit_results = _extract_relevant_fit_parameters(fit, node)

        self.assertTrue(bool(fit_data.sel(qubit="q1").success.values))
        self.assertTrue(fit_results["q1"].success)

    def test_good_qubit_spectroscopy_fit_uses_fit_maximum(self):
        node = self.make_node()
        detuning = np.linspace(-10e6, 10e6, 41)
        true_center = 1.37e6
        gamma = 1.8e6
        state = 0.1 + 0.8 * gamma**2 / ((detuning - true_center) ** 2 + gamma**2)
        sampled_max = float(detuning[np.argmax(state)])
        ds = xr.Dataset(
            {"state": (("qubit", "detuning"), state[np.newaxis, :])},
            coords={"qubit": ["q1"], "detuning": detuning},
        )

        fit_data, fit_results = fit_raw_data(ds, node)

        selected = fit_data.sel(qubit="q1")
        self.assertGreater(float(selected.fit_r_squared.values), 0.99)
        self.assertLess(abs(fit_results["q1"].relative_freq - true_center), 1e3)
        self.assertGreater(abs(fit_results["q1"].relative_freq - sampled_max), 1e5)

    def test_poor_qubit_spectroscopy_fit_uses_measured_maximum(self):
        node = self.make_node()
        detuning = np.linspace(-10e6, 10e6, 41)
        state = np.ones_like(detuning)
        state[10] = 2.0
        state[30] = 1.95
        measured_max = float(detuning[10])
        ds = xr.Dataset(
            {"state": (("qubit", "detuning"), state[np.newaxis, :])},
            coords={"qubit": ["q1"], "detuning": detuning},
        )

        fit_data, fit_results = fit_raw_data(ds, node)

        selected = fit_data.sel(qubit="q1")
        fit_r_squared = float(selected.fit_r_squared.values)
        self.assertTrue(not np.isfinite(fit_r_squared) or fit_r_squared < 0.8)
        self.assertEqual(fit_results["q1"].relative_freq, measured_max)

    def test_iq_qubit_spectroscopy_fit_uses_best_quadrature_r_squared(self):
        node = self.make_node()
        node.parameters.use_state_discrimination = False
        detuning = np.linspace(-10e6, 10e6, 41)
        true_center = 1.37e6
        gamma = 1.8e6
        signal = 0.1 + 0.8 * gamma**2 / ((detuning - true_center) ** 2 + gamma**2)
        i_data = np.linspace(-0.2, 0.2, detuning.size)
        q_data = signal
        ds = xr.Dataset(
            {
                "I": (("qubit", "detuning"), i_data[np.newaxis, :]),
                "Q": (("qubit", "detuning"), q_data[np.newaxis, :]),
                "IQ_abs": (
                    ("qubit", "detuning"),
                    np.sqrt(i_data**2 + q_data**2)[np.newaxis, :],
                ),
            },
            coords={"qubit": ["q1"], "detuning": detuning},
        )

        fit_data, fit_results = fit_raw_data(ds, node)

        selected = fit_data.sel(qubit="q1")
        self.assertEqual(str(selected.selected_quadrature.values), "Q")
        self.assertGreater(float(selected.fit_r_squared_Q.values), float(selected.fit_r_squared_I.values))
        self.assertLess(abs(fit_results["q1"].relative_freq - true_center), 1e3)


if __name__ == "__main__":
    unittest.main()
