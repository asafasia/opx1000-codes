from unittest.mock import patch

import numpy as np
import xarray as xr

from calibration_utils.ramsey.analysis import FIT_VALUES, _fit_ramsey_with_frequency_guess


def test_ramsey_fit_tries_configured_detuning_frequency_first():
    idle_time = np.linspace(0, 2000, 101)
    signal = 0.5 + 0.4 * np.cos(2 * np.pi * 0.0002 * idle_time)
    data = xr.DataArray(signal, dims=("idle_time",), coords={"idle_time": idle_time})
    starts = []

    def fake_curve_fit(model, time, values, p0, **kwargs):
        starts.append(p0[1])
        return np.asarray(p0), np.eye(5)

    with patch("calibration_utils.ramsey.analysis.curve_fit", side_effect=fake_curve_fit):
        fit = _fit_ramsey_with_frequency_guess(data, "idle_time", 0.0002)

    assert starts[0] == 0.0002
    assert fit.sizes["fit_vals"] == len(FIT_VALUES)


def test_ramsey_fit_recovers_frequency_with_an_outlier():
    idle_time = np.linspace(0, 4000, 201)
    expected_frequency = 0.0005
    signal = 0.5 + 0.4 * np.exp(-idle_time / 8000) * np.cos(
        2 * np.pi * expected_frequency * idle_time + 0.2
    )
    signal[80] = 1.8
    data = xr.DataArray(signal, dims=("idle_time",), coords={"idle_time": idle_time})

    fit = _fit_ramsey_with_frequency_guess(data, "idle_time", expected_frequency)

    fitted_frequency = float(fit.sel(fit_vals="f"))
    assert np.isfinite(fitted_frequency)
    assert abs(fitted_frequency - expected_frequency) < 5e-5
