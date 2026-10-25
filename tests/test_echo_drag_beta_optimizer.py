import json
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import xarray as xr

from Projects.shaped_pulse_spectroscopy.scripts import optimize_echo_drag_beta as optimizer


def synthetic_slices(beta, *, optimum=-0.3, flat_slice=None, amplitudes=None):
    detuning = np.linspace(-500e3, 500e3, 201)
    amps = np.asarray(optimizer.AMPLITUDE_FACTORS if amplitudes is None else amplitudes)
    centers = 25e3 + 150e3*(beta-optimum)*amps**2
    rng = np.random.default_rng(123)
    traces = 0.5 - 0.3*np.exp(-0.5*((detuning[None, :]-centers[:, None])/40e3)**2)
    traces += rng.normal(0, 0.0005, traces.shape)
    if flat_slice is not None:
        traces[flat_slice] = 0.5
    return xr.Dataset({"state": (("qubit", "amp_prefactor", "detuning"), traces[None, :, :])},
                      coords={"qubit": ["q1"], "amp_prefactor": amps, "detuning": detuning})


def test_defaults_and_exact_three_amplitudes():
    s = optimizer.SweepSettings()
    assert len(s.betas) == 40
    np.testing.assert_allclose(s.betas[[0, -1]], [-0.7, 0.1])
    for beta in s.betas:
        p = optimizer.parameters_for_beta(s, beta)
        np.testing.assert_array_equal(optimizer.amplitude_prefactors(p), [0.25, 0.5, 1.])
        assert p.drag_beta == beta and p.echo
        assert p.edge_smoothing == "polynomial" and p.edge_smoothing_power == 5
        assert p.echo_transition_time_ns == 16
        assert not p.ac_stark_correction
    args = optimizer.build_parser().parse_args([])
    assert not args.execute
    options = optimizer.calibration_options(acquire=True)
    assert options.save_raw_data and options.analyse_data
    assert not any((options.update_state, options.apply_profile_update,
                    options.propose_profile_update, options.plot_data))


def test_existing_fit_recovers_optimum_despite_common_frequency_offset():
    settings = optimizer.SweepSettings(beta_points=9)
    records = [optimizer.fit_and_score(synthetic_slices(beta), beta=beta, settings=settings)[0]
               for beta in settings.betas]
    assert all(r["valid"] for r in records)
    best = optimizer.select_best(records)
    assert best["beta"] == pytest.approx(-0.3)
    assert best["center_rms_hz"] < 100
    assert best["weighted_center_hz"] == pytest.approx(25e3, abs=100)
    # A common detuning offset is not penalized; amplitude dependence is.
    assert records[0]["center_rms_hz"] > 10e3


def test_one_bad_slice_rejects_entire_beta():
    record, _ = optimizer.fit_and_score(synthetic_slices(-0.3, flat_slice=2),
                                        beta=-0.3, settings=optimizer.SweepSettings())
    assert record["accepted"] == [True, True, False]
    assert not record["valid"] and record["center_rms_hz"] is None
    assert optimizer.select_best([record]) is None


def test_requested_descending_grid_and_five_amplitude_fits(tmp_path):
    args = vars(optimizer.build_parser().parse_args([
        "--beta-min", "-1.5", "--beta-max", "0", "--beta-points", "200",
        "--beta-order", "descending", "--amplitude-min", "0.2",
        "--amplitude-max", "1", "--amplitude-points", "5",
        "--amplitude-spacing", "linear", "--num-shots", "1000",
    ]))
    for name in ("execute", "dry_run", "output_root"):
        args.pop(name)
    settings = optimizer.SweepSettings(**args)
    assert settings.num_shots == 1000 and len(settings.betas) == 200
    np.testing.assert_allclose(settings.betas[[0, -1]], [0, -1.5])
    assert np.all(np.diff(settings.betas) < 0)
    expected = [.2, .4, .6, .8, 1.]
    np.testing.assert_allclose(settings.amplitude_factors, expected)
    p = optimizer.parameters_for_beta(settings, -1.0)
    np.testing.assert_allclose(optimizer.amplitude_prefactors(p), expected)
    records = []
    for beta in [-1.2, -1., -.8]:
        record, fitted = optimizer.fit_and_score(
            synthetic_slices(beta, optimum=-1., amplitudes=expected),
            beta=beta, settings=settings)
        assert len(record["center_hz"]) == 5 and record["valid"]
        records.append(record)
    assert optimizer.select_best(records)["beta"] == -1.
    rejected, _ = optimizer.fit_and_score(
        synthetic_slices(-1., optimum=-1., flat_slice=4, amplitudes=expected),
        beta=-1., settings=settings)
    assert rejected["accepted"] == [True, True, True, True, False]
    assert not rejected["valid"] and rejected["center_rms_hz"] is None
    summary = optimizer.save_summary(tmp_path, records, settings, complete=False)
    np.testing.assert_allclose(summary["amplitude_factors"], expected)
    assert len((tmp_path / "fits.csv").read_text().splitlines()) == 16
    optimizer.save_trace_plot(fitted, records[-1], settings, tmp_path / "five_slices.png")
    assert (tmp_path / "five_slices.png").stat().st_size > 1000


def test_low_contrast_fits_are_not_optima():
    record, _ = optimizer.fit_and_score(synthetic_slices(-0.3), beta=-0.3,
        settings=optimizer.SweepSettings(min_contrast=0.4))
    assert not record["valid"]


def test_all_waveforms_checked_and_voltage_limit_enforced(monkeypatch):
    machine = SimpleNamespace(qubits={"q1": SimpleNamespace(anharmonicity=-216e6)})
    settings = optimizer.SweepSettings(peak_amplitude_v=0.9)
    checked = optimizer.check_waveforms(settings, machine)
    assert checked["max_complex_amplitude_v"] > 0.7
    assert 0 < checked["max_complex_amplitude_v"] <= 1.0
    count = []
    def unsafe_i(p):
        count.append(p.drag_beta)
        return np.ones(100)*1.01
    monkeypatch.setattr(optimizer, "build_waveform", unsafe_i)
    with pytest.raises(ValueError, match="exceeds 1"):
        optimizer.check_waveforms(settings, machine)
    assert count == [settings.beta_min]


def test_partial_summary_does_not_declare_final_optimum_and_exports_plots(tmp_path):
    settings = optimizer.SweepSettings()
    record, fitted = optimizer.fit_and_score(synthetic_slices(-0.3), beta=-0.3, settings=settings)
    summary = optimizer.save_summary(tmp_path, [record], settings, complete=False)
    assert summary["best_beta"] is None and not summary["complete"]
    optimizer.save_trace_plot(fitted, record, settings, tmp_path / "slices.png")
    assert (tmp_path / "slices.png").stat().st_size > 1000
    assert (tmp_path / "beta_summary.png").stat().st_size > 1000
    assert len((tmp_path / "fits.csv").read_text().splitlines()) == 4
    failed, _ = optimizer.fit_and_score(synthetic_slices(0, flat_slice=0),
                                       beta=0, settings=settings)
    optimizer.write_json(tmp_path / "failed.json", failed)
    assert json.loads((tmp_path / "failed.json").read_text())["center_hz"][0] is None


def test_interrupted_acquisition_saves_partial_state_without_selecting_beta(tmp_path, monkeypatch):
    class FakeCalibration:
        def __init__(self, **kwargs):
            assert kwargs["auto_connect"] is False
        def run(self):
            return SimpleNamespace(interrupted=True)
    monkeypatch.setattr(optimizer, "check_waveforms", lambda *args: {})
    monkeypatch.setattr(optimizer, "EchoLorentzian", FakeCalibration)
    with pytest.raises(RuntimeError, match="interrupted"):
        optimizer.run_sweep(optimizer.SweepSettings(beta_points=2), object(), tmp_path)
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert summary["best_beta"] is None and summary["measured_betas"] == 0
    assert not summary["complete"]


def test_successful_sweep_visits_every_beta_and_saves_best(tmp_path, monkeypatch):
    seen = []
    class FakeCalibration:
        def __init__(self, **kwargs):
            p = kwargs["parameters"]
            seen.append(p.drag_beta)
            np.testing.assert_array_equal(optimizer.amplitude_prefactors(p), [0.25, 0.5, 1.])
            self.results = {"ds_raw": synthetic_slices(p.drag_beta)}
            self.namespace = {"calibration_run_directory": tmp_path / "raw"}
        def run(self):
            return SimpleNamespace(interrupted=False)
    monkeypatch.setattr(optimizer, "check_waveforms", lambda *args: {})
    monkeypatch.setattr(optimizer, "EchoLorentzian", FakeCalibration)
    monkeypatch.setattr(optimizer, "save_trace_plot", lambda *args: None)
    s = optimizer.SweepSettings(beta_points=3, beta_min=-0.7, beta_max=0.1)
    output, summary = optimizer.run_sweep(s, object(), tmp_path)
    np.testing.assert_allclose(seen, [-0.7, -0.3, 0.1])
    assert summary["complete"] and summary["best_beta"] == pytest.approx(-0.3)
    assert len(json.loads((output / "records.json").read_text())) == 3


@pytest.mark.parametrize("kwargs", [dict(beta_min=1), dict(beta_points=1),
    dict(peak_amplitude_v=1.01), dict(pulse_length_ns=21), dict(frequency_points=4),
    dict(transition_ns=0), dict(beta_max=float("nan")), dict(beta_order="random"),
    dict(amplitude_min=0), dict(amplitude_max=2), dict(amplitude_points=1),
    dict(amplitude_min=1), dict(amplitude_spacing="invalid")])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        optimizer.SweepSettings(**kwargs)
