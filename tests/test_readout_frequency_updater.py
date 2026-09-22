from importlib import import_module
import pytest
from quam_config import create_machine

@pytest.mark.parametrize("states, field", [(["g","e"],"frequencies_hz.resonator"),(["g","e","f"],"readout_gef.frequency_hz")])
def test_frequency_proposal_targets_selected_mode(states, field):
    module = import_module("calibrations.08a_readout_frequency_optimization")
    machine = create_machine(profile_name="single_qubit", qubit="q1")
    node = module.ReadoutFrequencyOptimization(parameters=module.Parameters(readout_states=states), machine=machine)
    node.namespace["qubits"] = [machine.qubits["q1"]]
    node.results["fit_results"] = {"q1":{"success":True,"optimal_frequency":6.7e9}}
    updates = node.profile_updates()
    assert updates[f"qubits.json.qubits.q1.{field}"] == 6.7e9
    assert machine.qubits["q1"].resonator.RF_frequency != 6.7e9
