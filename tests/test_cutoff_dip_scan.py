import json
from dataclasses import replace
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import xarray as xr

from Projects.shaped_pulse_spectroscopy.scripts import run_cutoff_dip_scan as scan


def gaussian(sigma=40_000, center=25_000, contrast=0.3):
    x = np.linspace(-500_000, 500_000, 201)
    y = 0.55 + 1e-7*x - contrast*np.exp(-0.5*((x-center)/sigma)**2)
    return x, y


def test_width_and_contrast_with_sloping_background():
    x, y = gaussian()
    result = scan.measure_dip(x, y)
    assert result["accepted"]
    assert result["center_hz"] == 25_000
    assert result["contrast"] == pytest.approx(0.3, abs=1e-8)
    assert result["fwhm_hz"] == pytest.approx(2*np.sqrt(2*np.log(2))*40_000, rel=0.002)
    assert result["right_half_hz"]-result["left_half_hz"] == result["fwhm_hz"]


def test_rejections_retain_contrast_but_mask_width():
    x, y = gaussian(sigma=5_000)
    result = scan.measure_dip(x, y)
    assert result["status"] == "undersampled_dip"
    assert np.isnan(result["fwhm_hz"])
    assert result["candidate_fwhm_hz"] > 0 and result["contrast"] > 0.29
    assert scan.measure_dip(x, np.full_like(x, .5))["status"] == "low_contrast"
    _, y = gaussian(center=250_000)
    assert scan.measure_dip(x, y)["status"] == "center_at_search_boundary"
    y[0] = np.nan
    assert scan.measure_dip(x, y)["status"] == "invalid_population"
    with pytest.raises(ValueError, match="spanning"):
        scan.measure_dip(x[1:], y[1:])


def test_multiple_dips_are_counted_and_nearest_crossings_used():
    x, y = gaussian(center=0)
    y -= .15*np.exp(-0.5*((x-180_000)/20_000)**2)
    result = scan.measure_dip(x, y)
    assert result["accepted"] and result["prominent_minima"] == 2
    assert result["fwhm_hz"] < 100_000


@pytest.mark.parametrize("changes", [
    dict(cutoff_min=1e-5), dict(cutoff_max=1.1), dict(pulse_length_ns=30001),
    dict(amplitude_points=1), dict(amplitude_max=2), dict(frequency_span_mhz=.5),
    dict(transition_ns=0), dict(num_shots=0), dict(drag_beta=float("nan")),
    dict(frequency_points=21, frequency_span_mhz=100),
])
def test_invalid_settings(changes):
    with pytest.raises(ValueError):
        replace(scan.ScanSettings(), **changes)


def test_pulse_and_safe_defaults():
    settings = scan.ScanSettings()
    p = scan.parameters_for_cutoff(settings, 1.)
    assert p.waveform_template_length_in_ns == p.lorentzian_length_in_ns == 30_000
    assert p.edge_smoothing == "polynomial" and p.edge_smoothing_power == 5
    assert p.echo and not p.ac_stark_correction and not p.simulate
    assert not p.use_readout_mitigation and not p.use_three_state_discrimination
    assert np.isfinite(scan.build_waveform(p)).all()
    assert not scan.build_parser().parse_args([]).execute
    for acquire in (False, True):
        opts = scan.calibration_options(acquire=acquire)
        assert not opts.update_state and not opts.apply_profile_update
        assert not opts.propose_profile_update and not opts.plot_data


def test_paper_zero_drive_reference_is_not_accepted_as_a_dip():
    settings = scan.ScanSettings(amplitude_min=0, amplitude_max=1, amplitude_points=100)
    np.testing.assert_allclose(scan.amplitude_prefactors(scan.parameters_for_cutoff(settings, .001)),
                               np.linspace(0, 1, 100))
    # Even a spurious dip from drift/noise cannot produce a zero-drive linewidth.
    x, y = gaussian()
    data = xr.Dataset({"state": (("qubit", "amp_prefactor", "detuning"),
                                 np.tile(y, (1, 100, 1)))},
                      coords={"qubit": [settings.qubit], "detuning": x,
                              "amp_prefactor": settings.amplitudes})
    rows = scan.analyze_dataset(data, settings, dict(cutoff=.001,
        nominal_rabi_mhz_per_factor=100., peak_rabi_mhz_per_factor=90., peak_complex_amplitude_v=.4))
    assert rows[0]["status"] == "zero_drive_reference" and not rows[0]["accepted"]
    assert np.isnan(rows[0]["fwhm_hz"]) and rows[1]["accepted"]


def test_dataset_metrics_and_summary(tmp_path):
    settings = scan.ScanSettings(cutoff_points=2, amplitude_points=2, t2_reference_us=10)
    x, y = gaussian()
    data = xr.Dataset({"state": (("qubit", "detuning", "amp_prefactor"),
                                 np.array([y, y]).T[None])},
                      coords={"qubit": [settings.qubit], "detuning": x,
                              "amp_prefactor": settings.amplitudes})
    rows = []
    for cutoff in settings.cutoffs:
        rows.extend(scan.analyze_dataset(data, settings, dict(
            cutoff=cutoff, nominal_rabi_mhz_per_factor=50., peak_rabi_mhz_per_factor=40.,
            peak_complex_amplitude_v=.5)))
    assert all(r["accepted"] for r in rows)
    assert rows[0]["fwhm_over_t2_limit"] == pytest.approx(rows[0]["fwhm_hz"]*np.pi*10e-6)
    assert rows[0]["nominal_rabi_mhz"] != rows[0]["peak_rabi_mhz"]
    scan.save_summary(tmp_path, rows, settings, complete=False)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert not saved["complete"] and saved["accepted_widths"] == 4
    assert len((tmp_path / "metrics.csv").read_text().splitlines()) == 5
    assert (tmp_path / "cutoff_fwhm_contrast.png").stat().st_size > 1000


def test_interrupted_acquisition_keeps_completed_cutoff(tmp_path, monkeypatch):
    settings = scan.ScanSettings(cutoff_points=2, amplitude_points=1,
                                 amplitude_min=.5, amplitude_max=.5)
    x, y = gaussian()
    dataset = xr.Dataset({"state": (("qubit", "amp_prefactor", "detuning"), y[None, None])},
                         coords={"qubit": [settings.qubit], "amp_prefactor": [.5], "detuning": x})
    checks = [dict(cutoff=float(c), nominal_rabi_mhz_per_factor=50.,
                   peak_rabi_mhz_per_factor=40., peak_complex_amplitude_v=.5) for c in settings.cutoffs]
    calls = []
    class FakeCalibration:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.results = {"ds_raw": dataset}
            self.namespace = {"calibration_run_directory": str(tmp_path / "raw")}

        def run(self):
            return SimpleNamespace(interrupted=len(calls) == 2)

    monkeypatch.setattr(scan, "EchoLorentzian", FakeCalibration)
    with pytest.raises(RuntimeError, match="interrupted"):
        scan.run_scan(settings, object(), checks, tmp_path)
    output = next(tmp_path.glob("*/summary.json"))
    summary = json.loads(output.read_text())
    assert not summary["complete"] and summary["measured_spectra"] == 1
    with np.load(output.parent / "cutoff_000_scan.npz") as saved:
        np.testing.assert_allclose(saved["state"][0], y)
