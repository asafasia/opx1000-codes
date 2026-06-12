import unittest

import numpy as np
import xarray as xr

from calibration_utils.power_rabi.analysis import (
    _r_squared,
    _select_best_quadrature_fit,
    _target_amplitude_prefactor,
)


def make_fit(frequency, amplitude=1.0, phase=0.0, offset=0.0):
    return xr.DataArray(
        [amplitude, frequency, phase, offset],
        dims=["fit_vals"],
        coords={"fit_vals": ["a", "f", "phi", "offset"]},
    )


def make_amplitude_sweep():
    values = np.linspace(0, 2, 101)
    return xr.DataArray(values, dims=["amp_prefactor"], coords={"amp_prefactor": values})


class PowerRabiAnalysisTests(unittest.TestCase):
    def test_x180_applies_hardware_amplitude_correction(self):
        self.assertAlmostEqual(_target_amplitude_prefactor(0.5, 1, "x180"), 0.5)

    def test_x180_compensates_for_pi_repetitions(self):
        # Three repeated pulses triple the measured oscillation frequency,
        # but the calibrated amplitude must remain the single-pulse pi value.
        self.assertAlmostEqual(_target_amplitude_prefactor(1.5, 3, "x180"), 0.5)

    def test_x90_uses_quarter_period(self):
        self.assertAlmostEqual(_target_amplitude_prefactor(0.25, 1, "x90"), 0.5)

    def test_ef_x180_uses_full_pi_amplitude(self):
        self.assertAlmostEqual(_target_amplitude_prefactor(0.5, 1, "EF_x180"), 0.5)

    def test_negative_fit_frequency_is_handled(self):
        self.assertAlmostEqual(_target_amplitude_prefactor(np.float64(-0.5), 1, "x180"), 0.5)

    def test_selects_I_when_I_has_higher_r_squared(self):
        amplitude = make_amplitude_sweep()
        fit_I = make_fit(0.5)
        fit_Q = make_fit(1.5)
        data_I = np.cos(2 * np.pi * 0.5 * amplitude)
        data_Q = np.cos(2 * np.pi * 0.5 * amplitude)

        selected, score_I, score_Q, quadrature = _select_best_quadrature_fit(
            data_I, data_Q, fit_I, fit_Q, "amp_prefactor"
        )

        self.assertEqual(str(quadrature.values), "I")
        self.assertGreater(float(score_I.values), float(score_Q.values))
        self.assertEqual(float(selected.sel(fit_vals="f").values), 0.5)

    def test_selects_Q_and_uses_its_frequency(self):
        amplitude = make_amplitude_sweep()
        fit_I = make_fit(0.5)
        fit_Q = make_fit(1.0)
        data_I = np.cos(2 * np.pi * 0.75 * amplitude)
        data_Q = np.cos(2 * np.pi * 1.0 * amplitude)

        selected, _, _, quadrature = _select_best_quadrature_fit(
            data_I, data_Q, fit_I, fit_Q, "amp_prefactor"
        )

        self.assertEqual(str(quadrature.values), "Q")
        selected_frequency = float(selected.sel(fit_vals="f").values)
        self.assertEqual(_target_amplitude_prefactor(selected_frequency, 1, "x180"), 0.25)

    def test_constant_data_has_invalid_r_squared(self):
        amplitude = make_amplitude_sweep()
        constant = xr.zeros_like(amplitude)

        score = _r_squared(constant, make_fit(0.5), "amp_prefactor")

        self.assertTrue(np.isnan(float(score.values)))

    def test_invalid_I_fit_does_not_win_over_valid_Q_fit(self):
        amplitude = make_amplitude_sweep()
        data = np.cos(2 * np.pi * 0.5 * amplitude)
        invalid_fit = make_fit(np.nan)
        valid_fit = make_fit(0.5)

        _, _, _, quadrature = _select_best_quadrature_fit(
            data, data, invalid_fit, valid_fit, "amp_prefactor"
        )

        self.assertEqual(str(quadrature.values), "Q")


if __name__ == "__main__":
    unittest.main()
