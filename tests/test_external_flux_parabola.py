import numpy as np
import pytest

from calibration_utils.qubit_spectroscopy_vs_flux.external_analysis import robust_parabola


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("bad_indices", [[0], [10], [20], [0, 10, 19]])
def test_vertex_resists_false_peaks_and_endpoint_outliers(sign, bad_indices):
    rng = np.random.default_rng(123)
    voltage = np.linspace(0.002, 0.004, 21)
    vertex = 0.00313
    freq = 4e6 + sign * 5e12 * (voltage - vertex)**2 + rng.normal(0, 2e4, len(voltage))
    freq[bad_indices] += 40e6
    result, curve, mask = robust_parabola(voltage, freq, 1e5)
    assert result["success"], result
    assert result["idle_offset"] == pytest.approx(vertex, abs=1e-5)
    assert result["frequency_shift"] == pytest.approx(4e6, abs=5e4)
    assert result["extremum_type"] == ("maximum" if sign < 0 else "minimum")
    assert not mask[bad_indices].any()
    assert result["num_outliers"] == len(bad_indices)
    assert np.isfinite(curve).all()


def test_missing_peaks_and_narrow_scan_at_large_absolute_bias():
    voltage = 3.5 + np.linspace(-1e-5, 1e-5, 21)
    vertex = 3.5 + 1e-6
    frequency = 5e9 - 1e15 * (voltage - vertex)**2
    frequency[[2, 7]] = np.nan
    frequency[-1] += 1e8
    result, _, mask = robust_parabola(voltage, frequency, 1e3)
    assert result["success"], result
    assert result["idle_offset"] == pytest.approx(vertex, abs=1e-9)
    assert result["frequency_shift"] == pytest.approx(5e9, abs=1)
    assert not mask[[2, 7, 20]].any()
    assert result["num_missing"] == 2


@pytest.mark.parametrize("slope", [0, 1e6])
def test_flat_or_linear_data_has_no_reliable_vertex(slope):
    voltage = np.linspace(-1, 1, 11)
    with pytest.raises(ValueError, match="Curvature"):
        robust_parabola(voltage, 2e6 + slope * voltage, 1e4)


def test_outside_vertex_is_not_successful():
    voltage = np.linspace(-1, 1, 11)
    result, _, _ = robust_parabola(voltage, -1e6 * (voltage - 2)**2, 1e3)
    assert not result["success"]
    assert result["idle_offset"] == pytest.approx(2)


def test_too_few_distinct_peaks_fail():
    with pytest.raises(ValueError, match="five distinct"):
        robust_parabola([0, 0, 1, 1, 2], [1, 1, 2, 2, 3], 1)


@pytest.mark.parametrize("better", ["I", "Q"])
@pytest.mark.parametrize("invalid_best", [False, True])
def test_selects_higher_r_squared_and_retains_both_candidates(better, invalid_best):
    from unittest.mock import patch
    import xarray as xr
    from calibration_utils.qubit_spectroscopy_vs_flux.external_analysis import fit_external_flux
    voltage = np.linspace(-1, 1, 21)
    ideal = 4e6 - 3e6 * (voltage - 0.1)**2
    noisy = ideal + np.random.default_rng(11).normal(0, 1e5, len(voltage))
    worse = "Q" if better == "I" else "I"
    tracks = {better: 4e6 - 3e6 * (voltage - 1.1)**2 if invalid_best else ideal, worse: noisy}
    ds = xr.Dataset(
        {ch: (("qubit", "detuning", "flux_bias"), np.zeros((1, 3, 21))) for ch in ("I", "Q")},
        coords={"qubit": ["q6"], "detuning": [-1e7, 0, 1e7], "flux_bias": voltage,
                "drive_frequency_hz": ("qubit", [5e9])},
    )
    # Use a realistic frequency bin size without building a large signal map.
    ds = ds.reindex(detuning=np.arange(-1e7, 1e7, 1e5))
    def peaks(signal, **kwargs):
        return xr.Dataset({"position": (("qubit", "flux_bias"), tracks[signal.name][None, :])},
                          coords={"qubit": ds.qubit, "flux_bias": ds.flux_bias})
    with patch("calibration_utils.qubit_spectroscopy_vs_flux.external_analysis.peaks_dips", side_effect=peaks):
        fits, results = fit_external_flux(ds)
    result = results["q6"]
    expected = worse if invalid_best else better
    assert result["selected_quadrature"] == expected
    assert result["quadrature_fit_results"][better]["r_squared"] == pytest.approx(1)
    assert result["quadrature_fit_results"][better]["r_squared"] > result["quadrature_fit_results"][worse]["r_squared"]
    assert result["idle_offset"] == pytest.approx(0.1, abs=0.03)
    if invalid_best:
        assert not result["quadrature_fit_results"][better]["success"]
    np.testing.assert_allclose(fits.peak_freq.sel(qubit="q6"), tracks[expected])
    assert set(fits.quadrature.values) == {"I", "Q"}


def test_r_squared_excludes_robustly_rejected_outlier():
    voltage = np.linspace(-1, 1, 21)
    frequency = 4e6 - 3e6 * (voltage - 0.1)**2
    frequency[0] += 4e7
    result, _, mask = robust_parabola(voltage, frequency, 1e5)
    assert not mask[0]
    assert result["r_squared"] == pytest.approx(1)
    assert result["r_squared_all_peaks"] < 0
