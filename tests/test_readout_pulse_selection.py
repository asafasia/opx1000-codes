"""Manifest readout selection reaches QUA and calibration update targets."""

from copy import deepcopy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from qm.qua import program
from profiles import load_profile, ProfileError
from profiles.loader import validate_profile
from quam_config.readout_pulses import FlatTopGaussianReadoutPulse
from quam_config.populate_quam_lf_mw_fems import _create_pulse
from utils.readout_macro import measure_readout


def selected_profile():
    profile = deepcopy(load_profile("single_qubit", qubit="q6"))
    q = profile["qubits"]["qubits"]["q6"]
    q["readout"]["use_kernel"] = False
    q["readout_gef"]["use_kernel"] = False
    return profile


@pytest.mark.parametrize("ge", ["readout", "readout_flattop"])
@pytest.mark.parametrize("gef", ["readout_GEF", "readout_GEF_flattop"])
def test_manifest_selects_independent_pulses_and_power_update_targets(ge, gef, monkeypatch):
    profile = selected_profile()
    profile["manifest"].update(readout_pulse=ge, readout_gef_pulse=gef)
    definitions = profile["pulses"]["pulses"]["q6"]
    for name, amplitude in [("readout", 0.1), ("readout_flattop", 0.2),
                            ("readout_GEF", 0.15), ("readout_GEF_flattop", 0.25)]:
        definitions[name]["amplitude"] = amplitude
    validate_profile(profile)
    builder = importlib.import_module("quam_config.create_machine_from_profile")
    monkeypatch.setattr(builder, "load_profile", lambda *a, **kw: profile)
    machine = builder.create_machine_from_profile("single_qubit", save=False, qubit="q6")
    q = machine.qubits["q6"]
    assert q.resonator.readout_discriminator == profile["manifest"]["readout_discriminator"]
    config = machine.generate_config()
    for operation, selected, states in [("readout", ge, ["g", "e"]), ("readout_GEF", gef, ["g", "e", "f"])]:
        pulse = q.resonator.operations[operation]
        assert pulse.amplitude == definitions[selected]["amplitude"]
        assert isinstance(pulse, FlatTopGaussianReadoutPulse) == selected.endswith("_flattop")
        assert q.resonator.readout_pulse_names[operation] == selected
        configured = config["pulses"][config["elements"][q.resonator.name]["operations"][operation]]
        assert configured["operation"] == "measurement"
        waveforms = [config["waveforms"][name] for name in configured["waveforms"].values()]
        assert any(w["type"] == "arbitrary" for w in waveforms) == selected.endswith("_flattop")
        with program():
            measure_readout(q, operation)
        module = importlib.import_module("calibrations.08b_readout_power_optimization")
        node = module.ReadoutPowerOptimization(parameters=module.Parameters(readout_states=states), machine=machine)
        node.namespace["qubits"] = [q]
        node.outcomes = {"q6": "successful"}
        node.results["fit_results"] = {"q6": {"optimal_amplitude": .12, "readout_fidelity": 95}}
        updates = node.profile_updates()
        amplitude_updates = {k: v for k, v in updates.items() if k.startswith("pulses.json.")}
        assert amplitude_updates == {f"pulses.json.pulses.q6.{selected}.amplitude": .12}


@pytest.mark.parametrize("field,value", [("readout_pulse", "missing"), ("readout_pulse", "readout_GEF"),
                                         ("readout_gef_pulse", "readout_flattop"), ("readout_pulse", [])])
def test_manifest_rejects_wrong_pulse_choices(field, value):
    profile = selected_profile()
    profile["manifest"][field] = value
    with pytest.raises(ProfileError, match=field):
        validate_profile(profile)


def test_manifest_requires_selected_definition():
    profile = selected_profile()
    profile["manifest"]["readout_pulse"] = "readout_flattop"
    del profile["pulses"]["pulses"]["q6"]["readout_flattop"]
    with pytest.raises(ProfileError, match="missing selected readout pulse"):
        validate_profile(profile)


def test_selected_shape_keeps_operation_kernel_filename(tmp_path):
    folder = tmp_path / "test" / "kernels"
    folder.mkdir(parents=True)
    np.savez(folder / "q6_readout_kernel.npz", time_ns=[1000, 2000], profile_kernel=[1, -0.5])
    definition = dict(type="flat_top_gaussian", target="resonator", length_ns=2000, amplitude=.1)
    settings = dict(use_kernel=True, threshold=0, rus_exit_threshold=0, integration_weights_angle_rad=0)
    pulse = _create_pulse("readout_flattop", definition, SimpleNamespace(name="q6"), settings, "test", tmp_path,
                          kernel_pulse_name="readout")
    assert [list(row) for row in pulse.integration_weights] == [[1, 1000], [-0.5, 1000]]
