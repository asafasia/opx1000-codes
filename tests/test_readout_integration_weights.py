"""Verify automatic constant weights and independent file-backed readout kernels."""

from types import SimpleNamespace
import numpy as np
import pytest

from profiles import ProfileError
from profiles.loader import _validate_pulse
from quam_config import create_machine
from quam_config.populate_quam_lf_mw_fems import _create_pulse
from quam_config.readout_pulses import with_gaussian_edges, with_square_envelope


def definition(shape):
    pulse = dict(target="resonator", type=shape, length_ns=2000, amplitude=0.1)
    if shape == "flat_top_gaussian":
        pulse["edge_length_ns"] = 100
    return pulse


def settings(use_kernel=False):
    return dict(use_kernel=use_kernel, threshold=0.01, rus_exit_threshold=0.02,
                integration_weights_angle_rad=0.3)


@pytest.mark.parametrize("shape", ["constant", "flat_top_gaussian"])
@pytest.mark.parametrize("operation", ["readout", "readout_GEF"])
def test_constant_weights_follow_length_in_generated_config(shape, operation, tmp_path):
    profile = definition(shape)
    _validate_pulse("q6." + operation, profile)
    # Old inline weights must neither override the constant weights nor require migration.
    profile["integration_weights"] = [[0.2, 16]]
    _validate_pulse("q6." + operation, profile)
    machine = create_machine(profile_name="single_qubit", qubit="q6")
    q = machine.qubits["q6"]
    pulse = _create_pulse(operation, profile, q, settings(), "test", tmp_path)
    q.resonator.operations[operation] = pulse
    assert pulse.integration_weights == [(1, 2000)]
    pulse.length = 1200
    assert pulse.integration_weights == [(1, 1200)]
    config = machine.generate_config()
    entry = config["pulses"][config["elements"][q.resonator.name]["operations"][operation]]
    weights = config["integration_weights"][entry["integration_weights"]["iw1"]]
    assert len(weights["cosine"]) == 1
    assert weights["cosine"][0][1] == 1200
    assert weights["cosine"][0][0] == pytest.approx(np.cos(0.3))


@pytest.mark.parametrize("operation", ["readout", "readout_GEF"])
def test_optimized_weights_use_selected_kernel_file(operation, tmp_path):
    folder = tmp_path / "test" / "kernels"
    folder.mkdir(parents=True)
    for name, values in [("readout", [0.25, -0.5]), ("readout_GEF", [0.75, -1.0])]:
        np.savez(folder / f"q6_{name}_kernel.npz", time_ns=[1000, 2000], profile_kernel=values)
    profile = definition("flat_top_gaussian")
    _validate_pulse("q6." + operation, profile)
    pulse = _create_pulse(operation, profile, SimpleNamespace(name="q6"), settings(True), "test", tmp_path)
    expected = [0.25, -0.5] if operation == "readout" else [0.75, -1.0]
    assert [list(row) for row in pulse.integration_weights] == [[v, 1000] for v in expected]
    profile["length_ns"] = 1200
    with pytest.raises(ProfileError, match="spans 2000 ns"):
        _create_pulse(operation, profile, SimpleNamespace(name="q6"), settings(True), "test", tmp_path)


def test_missing_optimized_kernel_is_not_replaced_by_constant_weights(tmp_path):
    with pytest.raises(ProfileError, match="does not exist"):
        _create_pulse("readout", definition("constant"), SimpleNamespace(name="q6"), settings(True), "test", tmp_path)


def test_shape_conversion_keeps_weights_automatic(tmp_path):
    pulse = _create_pulse("readout", definition("constant"), SimpleNamespace(name="q6"), settings(), "test", tmp_path)
    shaped = with_gaussian_edges(pulse, 100)
    shaped.length = 1200
    assert shaped.integration_weights == [(1, 1200)]
    square = with_square_envelope(shaped)
    square.length = 800
    assert square.integration_weights == [(1, 800)]


def test_gaussian_edge_duration_defaults_to_100_ns(tmp_path):
    profile = definition("flat_top_gaussian")
    profile.pop("edge_length_ns")
    _validate_pulse("q6.readout", profile)
    pulse = _create_pulse("readout", profile, SimpleNamespace(name="q6"), settings(), "test", tmp_path)
    assert pulse.edge_length_ns == 100
    assert pulse.integration_weights == [(1, 2000)]
