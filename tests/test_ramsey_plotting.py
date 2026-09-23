import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.ramsey.plotting import (
    plot_individual_data_with_fit,
    ramsey_fft_spectrum,
)


FIT_VALUES = ["a", "f", "phi", "offset", "decay"]


def _fit_dataset() -> xr.Dataset:
    values = np.array(
        [
            [0.4, 0.01, 0.0, 0.5, 0.001],
            [0.4, 0.01, np.pi, 0.5, 0.001],
        ]
    )
    return xr.Dataset(
        {"fit": (("detuning_signs", "fit_vals"), values)},
        coords={"detuning_signs": [1, -1], "fit_vals": FIT_VALUES},
    )


def _ramsey_dataset(variable: str = "state") -> xr.Dataset:
    idle_time = np.linspace(0, 100, 11)
    signal = 0.5 + 0.4 * np.cos(2 * np.pi * 0.01 * idle_time)
    values = np.stack([signal, 1 - signal])
    return xr.Dataset(
        {variable: (("detuning_signs", "idle_time"), values)},
        coords={"detuning_signs": [1, -1], "idle_time": idle_time},
    )


def test_discrimination_plot_has_probability_limits_and_smooth_fits():
    fig, axis = plt.subplots()
    data = _ramsey_dataset()
    fit_with_raw_coordinates = xr.merge([data, _fit_dataset()])

    plot_individual_data_with_fit(axis, data, {"qubit": "q1"}, fit_with_raw_coordinates)

    assert axis.get_ylim() == (-0.1, 1.1)
    fit_lines = [line for line in axis.lines if line.get_linestyle() == "-"]
    assert len(fit_lines) == 2
    assert all(len(line.get_xdata()) >= 1000 for line in fit_lines)
    assert all(np.ptp(line.get_ydata()) > 0.5 for line in fit_lines)
    assert axis.get_legend()._loc == 3
    assert axis.texts[0].get_position() == (1.0, 1.02)
    plt.close(fig)


def test_analog_plot_keeps_automatic_limits_and_smooth_fits():
    fig, axis = plt.subplots()

    plot_individual_data_with_fit(axis, _ramsey_dataset("I"), {"qubit": "q1"}, _fit_dataset())

    assert axis.get_ylim() != (-0.1, 1.1)
    fit_lines = [line for line in axis.lines if line.get_linestyle() == "-"]
    assert all(len(line.get_xdata()) >= 1000 for line in fit_lines)
    plt.close(fig)


def test_fft_spectrum_excludes_zero_frequency():
    frequency_mhz, amplitude = ramsey_fft_spectrum(_ramsey_dataset(), detuning_sign=1)

    assert frequency_mhz.size == amplitude.size
    assert frequency_mhz.size > 0
    assert np.all(frequency_mhz > 0)


def test_fft_view_excludes_zero_and_spectrum_is_offset_invariant():
    from calibration_utils.ramsey.plotting import plot_fft_peak
    time = np.arange(1024) * 4.
    signal = np.cos(2 * np.pi * 32 * np.arange(1024) / 1024)
    ds = xr.Dataset({"state": (("detuning_signs", "idle_time"), [signal, signal])},
                    coords={"detuning_signs": [1, -1], "idle_time": time})
    shifted = ds.assign(state=ds.state + 100.)
    f, a = ramsey_fft_spectrum(ds, 1)
    fs, amps = ramsey_fft_spectrum(shifted, 1)
    np.testing.assert_allclose(fs, f)
    np.testing.assert_allclose(amps, a, atol=1e-10)
    fig, ax = plt.subplots()
    plot_fft_peak(ax, shifted)
    fig.canvas.draw()
    assert ax.get_xlim()[0] == f[0]
    assert ax.get_xlim()[0] > 0
    for line in ax.lines:
        assert np.all(np.asarray(line.get_xdata()) > 0)
    plt.close(fig)


def test_fft_removes_linear_baseline_in_analysis_and_plot():
    from calibration_utils.ramsey.analysis import _fft_frequency_guess
    time = np.arange(1000) * 4.
    oscillation = .2 * np.cos(2 * np.pi * .001 * time)
    baseline = 10. + .001 * time
    signals = np.stack([oscillation, oscillation + baseline])
    original = signals.copy()
    ds = xr.Dataset({"state": (("detuning_signs", "idle_time"), signals)},
                    coords={"detuning_signs": [1, -1], "idle_time": time})
    f, clean = ramsey_fft_spectrum(ds, 1)
    fd, drifted = ramsey_fft_spectrum(ds, -1)
    np.testing.assert_allclose(fd, f)
    np.testing.assert_allclose(drifted, clean, atol=1e-10)
    assert f[np.argmax(drifted)] == 1.
    assert _fft_frequency_guess(time, signals[1]) * 1000 == f[np.argmax(drifted)]
    np.testing.assert_array_equal(signals, original)
