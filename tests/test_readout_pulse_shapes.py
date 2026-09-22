"""Profile-to-config checks for optional Gaussian readout edges."""

from copy import deepcopy
import importlib

import numpy as np
import pytest

from profiles import load_profile, ProfileError
from profiles.loader import _validate_pulse, validate_profile
from quam_config.readout_pulses import (
    FlatTopGaussianReadoutPulse, with_gaussian_edges, with_square_envelope,
)
from utils.readout_macro import readout_signature, validate_readout_calibration, readout_settings


def pulse_definition(**changes):
    return dict(target="resonator", type="flat_top_gaussian", amplitude=0.1,
                length_ns=2000, edge_length_ns=100,
                **changes)


@pytest.mark.parametrize("field,value", [
    ("edge_length_ns", None), ("edge_length_ns", 0), ("edge_length_ns", -4),
    ("edge_length_ns", True), ("edge_length_ns", 10), ("edge_length_ns", 1000),
    ("length_ns", 2002), ("amplitude", 0.71), ("target", "qubit"),
])
def test_reject_invalid_gaussian_readout_profile(field, value):
    pulse = pulse_definition()
    pulse[field] = value
    with pytest.raises(ProfileError):
        _validate_pulse("q6.readout", pulse)


@pytest.mark.parametrize("operation", ["readout", "readout_GEF"])
def test_selected_shape_reaches_measurement_config(operation, monkeypatch):
    profile = deepcopy(load_profile("single_qubit", qubit="q6"))
    profile["manifest"].pop("readout_pulse", None)
    profile["manifest"].pop("readout_gef_pulse", None)
    for entry in profile["pulses"]["pulses"]["q6"].values():
        if entry["target"] == "resonator":
            entry["type"] = "constant"
    selected = profile["pulses"]["pulses"]["q6"][operation]
    selected.update(type="flat_top_gaussian", edge_length_ns=100, amplitude=0.1)
    validate_profile(profile)
    module = importlib.import_module("quam_config.create_machine_from_profile")
    monkeypatch.setattr(module, "load_profile", lambda *a, **kw: profile)
    machine = module.create_machine_from_profile("single_qubit", save=False, qubit="q6")
    q = machine.qubits["q6"]
    pulse = q.resonator.operations[operation]
    assert isinstance(pulse, FlatTopGaussianReadoutPulse)
    other = "readout_GEF" if operation == "readout" else "readout"
    assert not isinstance(q.resonator.operations[other], FlatTopGaussianReadoutPulse)
    config = machine.generate_config()
    entry = config["pulses"][config["elements"][q.resonator.name]["operations"][operation]]
    assert entry["operation"] == "measurement"
    assert entry["length"] == selected["length_ns"]
    assert entry["integration_weights"]
    waveforms = [config["waveforms"][name] for name in entry["waveforms"].values()]
    samples = next(np.asarray(w["samples"]) for w in waveforms if w["type"] == "arbitrary" and max(np.abs(w["samples"])) > 0)
    assert len(samples) == selected["length_ns"]
    assert abs(samples[0]) < 1e-4
    np.testing.assert_allclose(samples[100:-100], 0.1)
    assert pulse.integration_weights == [(1, selected["length_ns"]) ]

    shaped_signature = readout_signature(q, operation)
    readout_settings(q, operation)["calibration_signature"] = shaped_signature
    validate_readout_calibration(q, operation)
    pulse.edge_length_ns = 200
    with pytest.raises(ValueError, match="recalibrate IQ blobs"):
        validate_readout_calibration(q, operation)
    q.resonator.operations[operation] = with_square_envelope(pulse)
    square_signature = readout_signature(q, operation)
    assert square_signature != shaped_signature
    q.resonator.operations[operation] = with_gaussian_edges(q.resonator.operations[operation], 100)
    assert readout_signature(q, operation) == shaped_signature


def test_edges_stay_fixed_when_length_changes():
    pulse = FlatTopGaussianReadoutPulse(length=2000, amplitude=0.1, edge_length_ns=100)
    pulse.length = 1000
    waveform = pulse.waveform_function()
    assert len(waveform) == 1000
    assert pulse.flat_length == 800
    np.testing.assert_allclose(waveform[100:900], 0.1)
    pulse.length = 200
    with pytest.raises(ValueError, match="positive flat-top"):
        pulse.waveform_function()
