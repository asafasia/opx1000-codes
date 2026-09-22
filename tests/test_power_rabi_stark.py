"""Offline synthetic checks of spectral-ridge extraction and power-law fits."""

from types import SimpleNamespace
import json

import numpy as np
import xarray as xr

from calibration_utils.power_rabi_chevron.stark import fit_stark_shift, _spectral_peak


def make_spectrum(exponent=2.0, noise=0.001, iq=False, theory=True):
    frequency = np.linspace(-40e6, 40e6, 321)
    amplitude = np.linspace(0, 0.4, 25)
    peak = 1.3e6 + 4e6 * (amplitude / 0.4) ** exponent
    signal = 0.1 + 0.75 * np.exp(-0.5 * ((frequency[:, None] - peak) / 3e6) ** 2)
    signal[:, 0] = 0.1
    rng = np.random.default_rng(8831)
    signal += rng.normal(0, noise, signal.shape)
    dims = ("qubit", "detuning", "amp_prefactor")
    if iq:
        # A rotated readout dip: neither I nor Q maximum is the spectral peak.
        complex_signal = (0.4 + 0.2j) - np.exp(0.7j) * (signal - 0.1)
        variables = {"I": (dims, complex_signal.real[None]), "Q": (dims, complex_signal.imag[None])}
    else:
        variables = {"state": (dims, signal[None])}
    ds = xr.Dataset(
        variables,
        coords={
            "qubit": ["q1"], "detuning": frequency,
            "amp_prefactor": amplitude / 0.2,
            "full_amp": (("qubit", "amp_prefactor"), amplitude[None]),
            "full_freq": (("qubit", "detuning"), (5e9 + frequency)[None]),
            "rabi_frequency_hz": (("qubit", "amp_prefactor"), (100e6 * amplitude)[None]),
            "signed_anharmonicity_hz": ("qubit", [-200e6]),
            "stark_theory_applicable": ("qubit", [theory]),
        },
    )
    node = SimpleNamespace(parameters=SimpleNamespace(use_state_discrimination=not iq))
    return ds, node, peak


def test_quadratic_center_and_prefactor_recovery():
    ds, node, expected = make_spectrum()
    fitted, reports = fit_stark_shift(ds, node)
    result = reports["q1"]
    assert result["success"]
    assert fitted.peak_status.values[0, 0] == "zero_amplitude"
    np.testing.assert_allclose(fitted.peak_detuning_hz.values[0, 1:], expected[1:], atol=0.08e6)
    assert abs(result["intercept_hz"] - 1.3e6) < 0.08e6
    assert abs(result["quadratic_coefficient_hz_per_amplitude2"] / 25e6 - 1) < 0.03
    assert abs(result["theory_coefficient"] - 0.5) < 0.02
    assert abs(result["exponent"] - 2) < 0.08
    assert result["quadratic_consistency"] == "consistent"
    assert result["r_squared"] > 0.995
    json.dumps(reports, allow_nan=False)


def test_rotated_iq_dip_uses_complex_displacement():
    ds, node, expected = make_spectrum(iq=True)
    fitted, reports = fit_stark_shift(ds, node)
    assert reports["q1"]["success"]
    np.testing.assert_allclose(fitted.peak_detuning_hz.values[0, 1:], expected[1:], atol=0.08e6)
    assert abs(reports["q1"]["theory_coefficient"] - 0.5) < 0.02


def test_flat_edge_ambiguous_and_nan_spectra_are_rejected():
    ds, node, _ = make_spectrum()
    frequency = ds.detuning.values
    ds.state.values[0, :, 1] = 0.1
    ds.state.values[0, :, 2] = np.exp(-0.5 * ((frequency - 40e6) / 3e6) ** 2)
    ds.state.values[0, :, 3] = (
        np.exp(-0.5 * ((frequency - 10e6) / 2e6) ** 2)
        + np.exp(-0.5 * ((frequency + 10e6) / 2e6) ** 2)
    )
    ds.state.values[0, :, 4] = np.nan
    fitted, reports = fit_stark_shift(ds, node)
    assert not fitted.peak_valid.values[0, 1:5].any()
    assert fitted.peak_status.values[0, 2] == "edge"
    assert fitted.peak_status.values[0, 3] == "ambiguous"
    assert fitted.peak_status.values[0, 4] == "nonfinite"
    assert reports["q1"]["success"]


def test_nonquadratic_data_are_not_declared_quadratic():
    ds, node, _ = make_spectrum(exponent=3.5, noise=0.0003)
    _, reports = fit_stark_shift(ds, node)
    result = reports["q1"]
    assert result["success"]
    assert abs(result["exponent"] - 3.5) < 0.1
    assert result["quadratic_consistency"] == "inconsistent"


def test_unresolved_shift_is_inconclusive():
    ds, node, _ = make_spectrum()
    ds.state.values[0, :, 1:] = ds.state.values[0, :, 1, None]
    _, reports = fit_stark_shift(ds, node)
    assert reports["q1"]["success"]
    assert reports["q1"]["quadratic_consistency"] == "inconclusive"


def test_missing_anharmonicity_and_rabi_do_not_block_empirical_fit():
    ds, node, _ = make_spectrum()
    ds = ds.drop_vars(["signed_anharmonicity_hz", "rabi_frequency_hz"])
    fitted, reports = fit_stark_shift(ds, node)
    result = reports["q1"]
    assert result["success"]
    assert not result["theory_applicable"]
    assert result["theory_coefficient"] is None
    assert np.isnan(fitted.theory_shift_hz).all()
    json.dumps(reports, allow_nan=False)


def test_shaped_pulse_reports_empirical_coefficient_without_half_claim():
    ds, node, _ = make_spectrum(theory=False)
    fitted, reports = fit_stark_shift(ds, node)
    result = reports["q1"]
    assert result["success"]
    assert result["theory_coefficient"] is not None
    assert result["ratio_to_ideal_half"] is None
    assert not result["theory_applicable"]
    assert np.isnan(fitted.theory_shift_hz).all()


def test_amplitude_and_weak_drive_limits_control_fit_selection():
    ds, node, _ = make_spectrum()
    node.parameters.stark_min_amp_factor = 0.5
    node.parameters.stark_max_amp_factor = 1.5
    node.parameters.stark_max_rabi_to_anharmonicity = 0.1
    fitted, reports = fit_stark_shift(ds, node)
    selected = fitted.peak_fit_used.values[0]
    assert reports["q1"]["success"]
    assert np.all(ds.amp_prefactor.values[selected] >= 0.5)
    assert np.all(ds.rabi_frequency_hz.values[0, selected] <= 20e6)
    assert fitted.peak_valid.values[0].sum() > selected.sum()


def test_nan_amplitudes_and_partial_trace_nans_are_handled():
    ds, node, _ = make_spectrum()
    ds.full_amp.values[0, 1] = np.nan
    ds.state.values[0, :4, 2] = np.nan
    fitted, reports = fit_stark_shift(ds, node)
    assert reports["q1"]["success"]
    assert fitted.peak_status.values[0, 1] == "nonfinite_amplitude"
    assert fitted.peak_valid.values[0, 2]
    json.dumps(reports, allow_nan=False)


def test_too_few_valid_points_does_not_claim_fit_success():
    ds, node, _ = make_spectrum()
    ds.state.values[0, :, 4:] = np.nan
    _, reports = fit_stark_shift(ds, node)
    assert not reports["q1"]["success"]
    assert reports["q1"]["quadratic_consistency"] == "inconclusive"


def test_disabled_drive_cut_does_not_claim_weak_drive_benchmark():
    ds, node, _ = make_spectrum()
    ds = ds.assign_coords(rabi_frequency_hz=ds.rabi_frequency_hz * 2)
    node.parameters.stark_max_rabi_to_anharmonicity = None
    fitted, reports = fit_stark_shift(ds, node)
    result = reports["q1"]
    assert result["success"]
    assert result["theory_coefficient"] is not None
    assert not result["theory_applicable"]
    assert result["ratio_to_ideal_half"] is None
    assert np.isnan(fitted.theory_shift_hz).all()


def test_gaussian_center_recovers_noisy_broad_peak_instead_of_sample_maximum():
    frequency = np.linspace(-50e6, 50e6, 101)
    expected = 3.7e6
    rng = np.random.default_rng(56)
    signal = (0.2 + 0.8 * np.exp(-0.5 * ((frequency - expected) / 12e6) ** 2)
              + rng.normal(0, 0.06, frequency.size))
    center, error, status = _spectral_peak(frequency, signal, 5, 5)
    assert status == "valid"
    assert abs(frequency[np.argmax(signal)] - expected) > 5e6
    assert abs(center - expected) < 0.7e6
    assert 0 < error < 1e6
    # Physical signal units must not change the center or uncertainty.
    scaled_center, scaled_error, scaled_status = _spectral_peak(
        frequency, signal * 1e-4 + 0.003, 5, 5,
    )
    assert scaled_status == "valid"
    np.testing.assert_allclose([scaled_center, scaled_error], [center, error], rtol=1e-4)


def test_gaussian_center_rejects_duplicate_frequencies():
    frequency = np.linspace(-20e6, 20e6, 41)
    signal = 0.2 + np.exp(-0.5 * ((frequency - 2e6) / 4e6) ** 2)
    frequency[10] = frequency[9]
    center, error, status = _spectral_peak(frequency, signal, 5, 5)
    assert status == "invalid_frequency_axis"
    assert np.isnan(center) and np.isnan(error)


def test_robust_gaussian_handles_outliers_and_sloping_background():
    frequency = np.linspace(-100e6, 100e6, 201)
    expected = 8.2e6
    rng = np.random.default_rng(640)
    signal = (0.3 + 0.001 * frequency / 1e6
              + 0.4 * np.exp(-0.5 * ((frequency - expected) / 18e6) ** 2)
              + rng.normal(0, 0.025, frequency.size))
    signal[[35, 96, 123, 172]] += [1.2, -0.8, 1.0, -1.0]
    center, error, status, diagnostics = _spectral_peak(
        frequency, signal, 5, 5, center_bounds=(-50e6, 50e6), return_diagnostics=True,
    )
    assert status == "valid"
    assert abs(center - expected) < 1e6
    assert abs(diagnostics["gaussian_sigma_hz"] / 18e6 - 1) < 0.1
    assert error > 0
    assert diagnostics["gaussian_height_snr"] > 5


def test_full_spectrum_supports_broad_peak_with_bounded_center():
    frequency = np.linspace(-100e6, 100e6, 101)
    expected = 12e6
    rng = np.random.default_rng(82)
    signal = 0.35 + 0.3 * np.exp(-0.5 * ((frequency - expected) / 28e6) ** 2)
    signal += rng.normal(0, 0.015, frequency.size)
    center, error, status = _spectral_peak(
        frequency, signal, 5, 5, center_bounds=(-20e6, 20e6),
    )
    assert status == "valid"
    assert abs(center - expected) < 2e6
    assert error < 2e6
    # A center outside the allowed range must not be reported as a valid edge fit.
    _, _, restricted_status = _spectral_peak(
        frequency, signal, 5, 5, center_bounds=(-20e6, 0),
    )
    assert restricted_status != "valid"


def test_neighbor_seed_cannot_force_center_to_neighbor_peak():
    frequency = np.linspace(-50e6, 50e6, 101)
    signal = 0.1 + 0.7 * np.exp(-0.5 * ((frequency - 9e6) / 6e6) ** 2)
    wrong_seed = 0.1 + 0.7 * np.exp(-0.5 * ((frequency + 18e6) / 12e6) ** 2)
    center, _, status = _spectral_peak(frequency, signal, 5, 5, seed_signal=wrong_seed)
    assert status == "valid"
    assert abs(center - 9e6) < 0.01e6


def test_neighbor_seed_does_not_create_peaks_in_noise():
    frequency = np.linspace(-50e6, 50e6, 101)
    seed_signal = 0.3 + 0.5 * np.exp(-0.5 * (frequency / 10e6) ** 2)
    for seed in range(5):
        signal = 0.3 + np.random.default_rng(seed).normal(0, 0.05, frequency.size)
        _, _, status = _spectral_peak(frequency, signal, 5, 5, seed_signal=seed_signal)
        assert status != "valid"
