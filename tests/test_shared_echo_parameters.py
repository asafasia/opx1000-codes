"""Shared pulse propagation and fast beta-map acquisition, without hardware."""

import ast
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import xarray as xr

from Projects.shaped_pulse_spectroscopy.scripts import optimize_echo_drag_beta_2d as fast
from shaped_pulse_spectroscopy import echo_parameters as shared

optimizer = fast.optimizer
PROJECT = Path(__file__).resolve().parents[1] / "Projects" / "shaped_pulse_spectroscopy"


def test_current_pulse_is_shared_and_instances_are_independent(monkeypatch):
    changed = replace(shared.ECHO_PULSE, drag_beta=-0.8, echo_transition_time_ns=24,
                      cutoff=.002, edge_smoothing_power=3)
    monkeypatch.setattr(shared, "ECHO_PULSE", changed)
    first = shared.fast_echo_parameters()
    for key, value in asdict(changed).items():
        assert getattr(first, key) == value
    first.drag_beta = 0
    assert shared.echo_parameters().drag_beta == -0.8
    assert shared.echo_parameters(drag_beta=0).drag_beta == 0
    assert shared.fast_echo_parameters(frequency_points=101).frequency_points == 101
    assert shared.fast_echo_parameters().frequency_points == 81


@pytest.mark.parametrize("overrides", [
    {"lorentzian_peak_amplitude": 1.01}, {"cutoff": 0},
    {"echo_transition_time_ns": -1}, {"edge_smoothing_power": 0},
    {"lorentzian_length_in_ns": 21}, {"drag_beta": float("nan")},
    {"waveform_template_length_in_ns": 24000},
    {"pulse_shape": "gaussian", "edge_smoothing": "polynomial"},
])
def test_shared_pulse_rejects_invalid_overrides(overrides):
    with pytest.raises(ValueError):
        shared.echo_parameters(**overrides)


@pytest.mark.parametrize("amplitude", [0.71, 0.8, 1.0])
def test_shared_pulse_accepts_amplitude_above_former_ceiling(amplitude):
    assert shared.echo_parameters(lorentzian_peak_amplitude=amplitude).lorentzian_peak_amplitude == amplitude


@pytest.mark.parametrize("name", ["detuning_amplitude_sweep", "amplitude_sweep",
                                 "fixed_amplitude_spectroscopy", "cutoff_amp_fwhm_map"])
def test_general_entrypoints_take_pulse_from_shared_file(name):
    module = importlib.import_module(f"experiments.{name}")
    tree = ast.parse((PROJECT / "experiments" / f"{name}.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.If)
                and ast.unparse(node.test) == "__name__ == '__main__'")
    # Evaluate only parameter assignments, never options, connections, or run().
    assignments = [node for node in main.body if isinstance(node, ast.Assign)
                   and any(ast.unparse(target).startswith("parameters") for target in node.targets)]
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "<parameters only>", "exec"), namespace)
    for key, value in asdict(shared.ECHO_PULSE).items():
        assert getattr(namespace["parameters"], key) == value
    if name == "cutoff_amp_fwhm_map":
        assert namespace["parameters"].fit_fwhm


def test_other_builders_inherit_shared_pulse():
    from experiments.fixed_amplitude_batch import build_parser, parameters_for
    from experiments.stark_kappa_sweep import _experiment_parameters
    from experiments.drag_beta_kappa_calibration import WaveformSpec
    from Projects.shaped_pulse_spectroscopy.scripts import run_pulse_length_sweeps as lengths
    fixed = parameters_for(build_parser().parse_args([]), echo=True, rabi_mhz=2.32)
    length_args = lengths.build_parser().parse_args([])
    lengths.validate_args(length_args)
    length = lengths.parameters_for(length_args, shared.ECHO_PULSE.lorentzian_length_in_ns/1000)
    for p in (fixed, length):
        for key, value in asdict(shared.ECHO_PULSE).items():
            assert getattr(p, key) == value
    kappa = _experiment_parameters(.001)
    assert kappa.drag_beta == shared.ECHO_PULSE.drag_beta
    assert kappa.ac_stark_correction and kappa.stark_kappa_mhz_inv == .001
    assert WaveformSpec().cutoff == shared.ECHO_PULSE.cutoff
    assert WaveformSpec().peak_amplitude_v == shared.ECHO_PULSE.lorentzian_peak_amplitude


def synthetic_map(settings, beta, *, flat_index=None):
    frequencies = np.linspace(-500e3, 500e3, settings.frequency_points)
    amplitudes = settings.amplitude_factors
    centers = 20e3 + 100e3*(beta + .3)*amplitudes**2
    state = .5 - .3*np.exp(-.5*((frequencies[None, :]-centers[:, None])/40e3)**2)
    state += np.random.default_rng(12).normal(0, .001, state.shape)
    if flat_index is not None:
        state[flat_index] = .5
    return xr.Dataset({"state": (("qubit", "amp_prefactor", "detuning"), state[None])},
                      coords={"qubit": ["q1"], "amp_prefactor": amplitudes, "detuning": frequencies})


def test_fast_defaults_and_explicit_cli_overrides():
    settings = fast.default_settings()
    assert settings.amplitude_points == 13 and settings.frequency_points == 81
    assert settings.num_shots == 200 and settings.beta_points == 40
    np.testing.assert_allclose(settings.betas[[0, -1]], [0, -1.5])
    np.testing.assert_allclose(settings.amplitude_factors[[0, -1]], [.2, 1])
    assert settings.amplitude_points * settings.frequency_points < 31*200/5
    args = optimizer.build_parser(defaults=settings).parse_args([
        "--beta-points", "25", "--frequency-points", "101", "--amplitude-points", "17"])
    assert not args.execute
    assert (args.beta_points, args.frequency_points, args.amplitude_points) == (25, 101, 17)


def test_fast_map_fits_every_amplitude_and_rejects_one_failed_slice():
    settings = fast.default_settings()
    good, _ = optimizer.fit_and_score(synthetic_map(settings, -.3), beta=-.3, settings=settings)
    off, _ = optimizer.fit_and_score(synthetic_map(settings, -.7), beta=-.7, settings=settings)
    assert good["valid"] and len(good["accepted"]) == 13
    assert good["center_rms_hz"] < 1000 < off["center_rms_hz"]
    bad, _ = optimizer.fit_and_score(synthetic_map(settings, -.3, flat_index=6), beta=-.3, settings=settings)
    assert not bad["valid"] and bad["center_rms_hz"] is None
    assert not bad["accepted"][6]


def test_fast_sweep_saves_every_map_fits_and_minimum(tmp_path, monkeypatch):
    settings = replace(fast.default_settings(), beta_min=-.7, beta_max=.1, beta_points=3)
    seen = []
    class FakeCalibration:
        def __init__(self, **kwargs):
            assert kwargs["auto_connect"] is False
            assert not kwargs["options"].update_state
            p = kwargs["parameters"]
            seen.append(p.drag_beta)
            self.results = {"ds_raw": synthetic_map(settings, p.drag_beta)}
            self.namespace = {"calibration_run_directory": tmp_path / "raw"}
        def run(self):
            return SimpleNamespace(interrupted=False)
    monkeypatch.setattr(optimizer, "check_waveforms", lambda *args: {})
    monkeypatch.setattr(optimizer, "EchoLorentzian", FakeCalibration)
    output, summary = optimizer.run_sweep(settings, object(), tmp_path, plot_mode="2d")
    np.testing.assert_allclose(seen, [.1, -.3, -.7])
    assert summary["complete"] and summary["best_beta"] == pytest.approx(-.3)
    assert len(list(output.glob("beta_*_map.png"))) == 3
    for path in output.glob("beta_*_scan.npz"):
        with np.load(path) as scan:
            assert scan["state"].shape == (13, 81)
            assert scan["accepted"].all()
    assert len((output / "fits.csv").read_text().splitlines()) == 1+3*13
    snapshot = json.loads((output / "settings.json").read_text())
    assert snapshot["parameters_at_first_beta"]["cutoff"] == shared.ECHO_PULSE.cutoff
    assert snapshot["parameters_at_first_beta"]["drag_beta"] == pytest.approx(.1)
