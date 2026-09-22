from importlib import import_module
from types import SimpleNamespace
import pytest
from quam_config import create_machine

@pytest.mark.parametrize("states, section, pulse", [(["g","e"],"readout","readout"),(["g","e","f"],"readout_gef","readout_GEF")])
def test_power_update_targets_only_selected_pulse_and_invalidates_centers(states, section, pulse):
    module = import_module("calibrations.08b_readout_power_optimization")
    machine = create_machine(profile_name="single_qubit", qubit="q1")
    node = module.ReadoutPowerOptimization(parameters=module.Parameters(readout_states=states), machine=machine)
    node.namespace["qubits"] = [machine.qubits["q1"]]
    node.outcomes = {"q1":"successful"}
    node.results["fit_results"] = {"q1":{"optimal_amplitude":.12,"readout_fidelity":95}}
    updates = node.profile_updates()
    selected_pulse = machine.qubits["q1"].resonator.readout_pulse_names[pulse]
    assert updates[f"pulses.json.pulses.q1.{selected_pulse}.amplitude"] == .12
    assert updates[f"qubits.json.qubits.q1.{section}.gef_centers"] is None
    assert updates[f"qubits.json.qubits.q1.{section}.confusion_matrix"] is None
    if len(states)==2:
        assert updates["metrics.json.qubits.q1.readout.fidelity_percent.thermal"] == 95
