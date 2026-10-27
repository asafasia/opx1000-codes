from types import SimpleNamespace
import json

import numpy as np
import pytest
import xarray as xr

from Projects.shaped_pulse_spectroscopy.scripts import optimize_echo_drag_kappa as opt


def dataset(settings, kappa=0.0, *, validation=False, flat=False):
    s = opt.scoring_settings(settings, validation=validation)
    amps = s.amplitude_factors
    det = np.linspace(-500e3, 500e3, s.frequency_points)
    centers = 30e3 + (kappa-0.0001)*1e8*amps**2
    values = .55-.3*np.exp(-.5*((det[None, :]-centers[:, None])/68e3)**2)
    values += np.random.default_rng(7).normal(0, .0004, values.shape)
    if flat:
        values[-1] = .5
    return xr.Dataset({"state": (("qubit", "amp_prefactor", "detuning"), values[None])},
        coords={"qubit": [settings.qubit], "amp_prefactor": amps, "detuning": det})


def test_wide_dips_score_and_bad_amplitude_rejects_whole_candidate():
    s = opt.Settings()
    record, _ = opt.score_dataset(dataset(s, .0001), s, s.beta, .0001)
    assert record["valid"] and min(record["fwhm_hz"]) > 120e3
    assert record["center_rms_hz"] < 500
    assert not record["leakage_available"]
    bad, _ = opt.score_dataset(dataset(s, flat=True), s, s.beta, 0)
    assert not bad["valid"] and bad["center_rms_hz"] is None


def test_pair_parameters_do_not_replace_selected_beta_or_global_defaults():
    s = opt.Settings(beta=-1.097)
    before = opt.ECHO_PULSE
    p = opt.parameters_for_pair(s, s.beta, -.0002)
    assert p.drag_beta == s.beta and p.ac_stark_correction
    assert p.stark_kappa_mhz_inv == -.0002
    assert p.lorentzian_length_in_ns == 20000 and p.echo_transition_time_ns == 16
    assert not opt.parameters_for_pair(s, s.beta, 0).ac_stark_correction
    assert opt.ECHO_PULSE == before and not opt.ECHO_PULSE.ac_stark_correction
    options = opt.beta_tools.calibration_options(acquire=True)
    assert not any((options.update_state, options.apply_profile_update, options.propose_profile_update))


def test_symmetric_bounded_scan_and_refinement():
    s = opt.Settings()
    assert s.kappas[0] == 0
    np.testing.assert_allclose(sorted(s.kappas), np.linspace(-.0003, .0003, 7))
    np.testing.assert_allclose(opt.local_kappas(.0001, s), [.00005, .00015])
    assert opt.local_kappas(.0003, s) == pytest.approx([.00025])


def test_quantized_chirp_tracks_both_signs_and_square_amplitude():
    machine = SimpleNamespace(qubits={"q1": SimpleNamespace(xy=SimpleNamespace(
        operations={"x180": SimpleNamespace(amplitude=.309527255, length=40)}))})
    for kappa in [-.0003, 0, .0003]:
        metrics = opt.check_chirp(opt.Settings(), machine, kappa)
        assert metrics["max_chirp_error_hz"] < 100
        assert metrics["peak_drive_correction_hz"] < 200e3


def test_interrupted_run_retains_partial_status(tmp_path, monkeypatch):
    class Interrupted:
        def __init__(self, **kwargs):
            assert kwargs["auto_connect"] is False
        def run(self):
            return SimpleNamespace(interrupted=True)
    monkeypatch.setattr(opt, "assert_outputs_allowed", lambda: None)
    monkeypatch.setattr(opt, "EchoLorentzian", Interrupted)
    monkeypatch.setattr(opt.beta_tools, "check_waveforms", lambda *a: {})
    monkeypatch.setattr(opt, "check_chirp", lambda *a: {})
    monkeypatch.setattr(opt, "profile_hashes", lambda *a: {"profile": "unchanged"})
    with pytest.raises(RuntimeError, match="interrupted"):
        opt.run(opt.Settings(), object(), tmp_path / "run")
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert not summary["complete"] and summary["selected_pair"] is None
    assert summary["profile_files_unchanged"]


def test_stages_hold_beta_then_recheck_and_validate_with_fresh_baselines(tmp_path, monkeypatch):
    s = opt.Settings()
    seen = []
    class Fake:
        def __init__(self, **kw):
            p = kw["parameters"]
            seen.append((p.drag_beta, p.stark_kappa_mhz_inv))
            assert not kw["auto_connect"]
            self.results = {"ds_raw": dataset(s, p.stark_kappa_mhz_inv,
                validation=p.amp_factor_points == s.validation_amplitude_points)}
            self.namespace = {"calibration_run_directory": tmp_path / "raw"}
        def run(self):
            return SimpleNamespace(interrupted=False)
    monkeypatch.setattr(opt, "assert_outputs_allowed", lambda: None)
    monkeypatch.setattr(opt, "EchoLorentzian", Fake)
    monkeypatch.setattr(opt.beta_tools, "check_waveforms", lambda *a: {})
    monkeypatch.setattr(opt, "check_chirp", lambda *a: {})
    monkeypatch.setattr(opt, "profile_hashes", lambda *a: {"profile": "unchanged"})
    monkeypatch.setattr(opt, "plot_summary", lambda *a: None)
    monkeypatch.setattr(opt.beta_tools, "save_scan", lambda *a: None)
    monkeypatch.setattr(opt.beta_tools, "save_2d_plot", lambda *a: None)
    summary = opt.run(s, object(), tmp_path / "run")
    assert summary["complete"] and summary["measurements"] == 14
    assert all(beta == s.beta for beta, _ in seen[:9])
    assert seen[-3] == seen[-1] == (s.beta, 0)
    assert summary["selected_pair"]["kappa_mhz_inv"] == pytest.approx(.0001)
    assert summary["candidate_better_than_both_baselines"]
