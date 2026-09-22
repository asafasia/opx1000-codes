"""Offline behavioral checks for independent GE/GEF readout and automatic reset."""
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch
import copy

import numpy as np
import pytest
import xarray as xr
from qm.qua import declare, program, stream_processing
from calibration_utils.T1.parameters import Parameters
from calibrations import CalibrationOptions
from profiles import load_profile
from profiles.loader import validate_profile, ProfileError
from quam_config import create_machine
from utils.experiment_readout import PopulationStreams, reset_qubit, convert_IQ_to_V
from utils.readout_macro import readout_state_configured, measure_readout

GE = ["g", "e"]
GEF = ["g", "e", "f"]
CENTERS = [[-0.001, 0.0], [0.001, 0.0], [0.0, 0.002]]

@pytest.mark.parametrize("states", [["e", "g"], ["g", "f"], ["g"], ["g", "e", "e"]])
def test_reject_invalid_readout_bases(states):
    with pytest.raises(ValueError, match="readout_states"):
        Parameters(readout_states=states)

@pytest.mark.parametrize("name", ["q1", "q6"])
def test_second_pulse_is_independent_and_uncalibrated(name):
    machine = create_machine(profile_name="single_qubit", qubit=name)
    q = machine.qubits[name]
    original = q.resonator.operations["readout"].length
    q.resonator.operations["readout_GEF"].length = original + 40
    assert q.resonator.operations["readout"].length == original
    assert q.resonator.readout_gef["gef_centers"] is None
    with program():
        state = declare(int)
        with pytest.raises(ValueError, match="readout_GEF has no calibrated IQ centers"):
            readout_state_configured(q, state, num_states=3)

@pytest.mark.parametrize("states", [GE, GEF])
@pytest.mark.parametrize("alias", ["active", "active_gef"])
def test_active_reset_follows_readout_basis(states, alias):
    parameters = Parameters(readout_states=states, reset_type=alias, active_reset_max_attempts=4)
    q = object()
    with patch("utils.experiment_readout.active_reset_configured") as reset:
        reset_qubit(q, parameters)
    assert reset.call_args.kwargs["num_states"] == len(states)
    assert reset.call_args.kwargs["pulse_name"] == ("readout_GEF" if len(states)==3 else "readout")
    assert reset.call_args.kwargs["max_attempts"] == 4


def test_gef_frequency_is_selected_and_restored():
    calls = []
    rr = SimpleNamespace(operations={"readout_GEF": object()}, RF_frequency=6e9,
        intermediate_frequency=100_000_000, readout_gef={"frequency_hz": 6.002e9},
        update_frequency=lambda f: calls.append(("frequency", f)),
        measure=lambda name, **kw: calls.append(("measure", name)))
    measure_readout(SimpleNamespace(name="q", resonator=rr), "readout_GEF", frequency_offset=100)
    assert calls == [("frequency", 102000100), ("measure", "readout_GEF"), ("frequency", 100000000)]


def test_iq_conversion_uses_selected_pulse_length():
    rr = SimpleNamespace(operations={"readout": SimpleNamespace(length=1000),
        "readout_GEF": SimpleNamespace(length=2000)}, selected_readout_operation="readout_GEF")
    ds = xr.Dataset({"I": ("qubit", [1.0]), "Q": ("qubit", [2.0])}, coords={"qubit": ["q"]})
    converted = convert_IQ_to_V(ds, [SimpleNamespace(name="q", resonator=rr)])
    assert converted.I.item() == 4096 / 2000


def test_population_streams_record_indicators_not_integer_state_labels():
    fake_streams = {label: object() for label in GEF}
    bundle = PopulationStreams(GEF, fake_streams)
    with patch("utils.experiment_readout.declare"), patch("utils.experiment_readout.assign") as assign, patch("utils.experiment_readout.Cast.to_int", side_effect=int), patch("utils.experiment_readout.save") as save:
        bundle.save_shot(2)
    assert [call.args[1] for call in assign.call_args_list] == [0, 0, 1]
    assert [call.args[1] for call in save.call_args_list] == list(fake_streams.values())


@pytest.mark.parametrize("states", [GE, GEF])
@pytest.mark.parametrize("name", ["q1", "q6"])
def test_t1_builds_with_mode_specific_readout_and_active_reset(name, states):
    machine = create_machine(profile_name="single_qubit", qubit=name)
    q = machine.qubits[name]
    q.resonator.readout_gef["gef_centers"] = CENTERS
    params = Parameters(qubits=[name], readout_states=states, reset_type="active", use_state_discrimination=True)
    cls = import_module("calibrations.05_T1").T1
    node = cls(parameters=params, machine=machine, options=CalibrationOptions(save_raw_data=False, save_figures=False,
        plot_data=False, update_state=False, propose_profile_update=False, apply_profile_update=False))
    qua = node.create_qua_program()
    assert qua is not None
    config = machine.generate_config()
    assert "readout_GEF" in config["elements"][q.resonator.name]["operations"]


def test_iq_gef_bootstraps_without_centers_and_updates_only_gef():
    module = import_module("calibrations.07_iq_blobs")
    machine = create_machine(profile_name="single_qubit", qubit="q1")
    q = machine.qubits["q1"]
    ge_before = copy.deepcopy(q.resonator.readout_ge.to_dict())
    p = module.Parameters(qubits=["q1"], readout_states=GEF, reset_type="thermal", num_shots=3)
    node = module.IqBlobs(parameters=p, machine=machine)
    assert node.create_qua_program() is not None
    assert p.operation == "readout_GEF" and node.namespace["prepared_states"] == GEF
    node.results["fit_results"] = {"q1": {"state_labels": GEF, "center_matrix": CENTERS,
        "confusion_matrix": np.eye(3).tolist()}}
    node.outcomes = {"q1": "successful"}
    node.update_state()
    updates = node.profile_updates()
    assert all(".readout_gef." in key for key in updates)
    assert q.resonator.readout_ge.to_dict() == ge_before
    assert np.asarray(q.resonator.readout_gef["gef_centers"]).shape == (3, 2)


def test_three_state_mitigation_uses_full_matrix():
    module = import_module("calibrations.05_T1")
    machine = create_machine(profile_name="single_qubit", qubit="q1")
    matrix = np.array([[.9,.08,.02],[.1,.8,.1],[.05,.15,.8]])
    machine.qubits["q1"].resonator.readout_gef["confusion_matrix"] = matrix.tolist()
    node = module.T1(parameters=Parameters(readout_states=GEF, use_state_discrimination=True,
        use_readout_mitigation=True, qubits=["q1"]), machine=machine)
    node.get_qubits()
    truth = np.array([.2,.3,.5]); measured = truth @ matrix
    ds = xr.Dataset({f"population_{s}": ("qubit", [measured[i]]) for i,s in enumerate(GEF)}, coords={"qubit":["q1"]})
    ds["state"] = ds.population_e.copy()
    node.results["ds_raw"] = ds
    node.apply_readout_mitigation()
    result = node.results["ds_raw"]
    np.testing.assert_allclose([result[f"population_{s}"].item() for s in GEF], truth)
    assert result.state.item() == pytest.approx(.3)
    assert result.state_unmitigated.item() == pytest.approx(measured[1])


def test_profile_rejects_invalid_gef_frequency_and_centers():
    for key, value in (("frequency_hz", 1e12), ("gef_centers", [[0,0],[1,1]])):
        profile = load_profile("single_qubit", qubit="q1")
        profile["qubits"]["qubits"]["q1"]["readout_gef"][key] = value
        with pytest.raises(ProfileError, match="readout_gef"):
            validate_profile(profile)


def test_changed_pulse_rejects_stale_centers():
    from utils.readout_macro import readout_signature
    q = create_machine(profile_name="single_qubit", qubit="q1").qubits["q1"]
    q.resonator.readout_gef["gef_centers"] = CENTERS
    q.resonator.readout_gef["calibration_signature"] = readout_signature(q, "readout_GEF")
    q.resonator.operations["readout_GEF"].amplitude *= 0.9
    with program():
        with pytest.raises(ValueError, match="changed after IQ calibration"):
            readout_state_configured(q, declare(int), num_states=3)


def test_ge_and_gef_switch_on_same_experiment_does_not_reuse_old_operation():
    module = import_module("calibrations.07_iq_blobs")
    machine = create_machine(profile_name="single_qubit", qubit="q1")
    node = module.IqBlobs(parameters=module.Parameters(readout_states=GEF, num_shots=2), machine=machine)
    node.create_qua_program()
    assert node.parameters.operation == "readout_GEF"
    node.parameters.readout_states = GE
    node.create_qua_program()
    assert node.parameters.operation == "readout"
    assert node.namespace["prepared_states"] == GE
    assert machine.qubits["q1"].resonator.selected_readout_operation == "readout"


def test_rb_ground_population_excludes_f_leakage():
    from utils.experiment_readout import ground_population
    ds = xr.Dataset({"state": ("shot", [.1]), "population_g": ("shot", [.6]),
                     "population_f": ("shot", [.3])})
    assert ground_population(ds).item() == .6
    assert ground_population(ds.drop_vars("population_g")).item() == pytest.approx(.6)


def test_gef_iq_calibration_signature_accepts_same_pulse_after_update():
    module = import_module("calibrations.07_iq_blobs")
    machine = create_machine(profile_name="single_qubit", qubit="q1")
    node = module.IqBlobs(parameters=module.Parameters(readout_states=GEF), machine=machine)
    node.namespace["qubits"] = [machine.qubits["q1"]]
    node.results["fit_results"] = {"q1":{"state_labels":GEF,"center_matrix":CENTERS,
        "confusion_matrix": np.eye(3).tolist()}}
    node.update_state()
    with program():
        readout_state_configured(machine.qubits["q1"], declare(int), num_states=3)


def test_ge_iq_update_preserves_gef_and_rotates_centers_only_once():
    module = import_module("calibrations.07_iq_blobs")
    machine = create_machine(profile_name="single_qubit", qubit="q1")
    q = machine.qubits["q1"]
    gef_before = q.resonator.readout_gef.to_dict()
    angle_before = q.resonator.operations["readout"].integration_weights_angle
    node = module.IqBlobs(parameters=module.Parameters(readout_states=GE), machine=machine)
    node.namespace["qubits"] = [q]
    node.results["fit_results"] = {"q1":{"state_labels":GE,"center_matrix":CENTERS[:2],
        "iw_angle": .25, "ge_threshold": 0.0, "rus_threshold": 0.0,
        "confusion_matrix": np.eye(2).tolist(), "fidelity_matrix":np.eye(2).tolist()}}
    node.update_state()
    updates = node.profile_updates()
    assert updates["qubits.json.qubits.q1.readout.integration_weights_angle_rad"] == pytest.approx(angle_before-.25)
    assert q.resonator.readout_gef.to_dict() == gef_before
    with program():
        readout_state_configured(q, declare(int), num_states=2)
