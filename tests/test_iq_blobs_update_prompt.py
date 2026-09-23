from importlib import import_module
from unittest.mock import Mock
import numpy as np
import pytest
from calibrations import CalibrationOptions
from quam_config.create_machine_from_profile import create_machine_from_profile


@pytest.mark.parametrize("matched", [True, False])
def test_gef_update_prompts_only_for_matching_readout_and_explains_skips(matched):
    module = import_module("calibrations.07_iq_blobs")
    machine = create_machine_from_profile("single_qubit", qubit="q6", save=False)
    updater, logger = Mock(), Mock()
    node = module.IqBlobs(
        parameters=module.Parameters(readout_states=["g", "e", "f"] if matched else ["g", "e"]),
        machine=machine, profile_name="single_qubit", profile_updater=updater, logger=logger,
        options=CalibrationOptions(propose_profile_update=True, apply_profile_update=True))
    node.namespace["qubits"] = [machine.qubits["q6"]]
    node.results["fit_results"] = {"q6": {
        "state_labels": ["g", "e", "f"],
        "center_matrix": [[-.001, 0], [.001, 0], [0, .002]],
        "confusion_matrix": np.eye(3).tolist(),
    }}
    assert node._propose_profile_update_from_options() is matched
    if matched:
        updater.stage.assert_called_once()
        assert all(".readout_gef." in key for key in updater.stage.call_args.args[1])
        updater.confirm_and_apply.assert_called_once_with(updater.stage.return_value)
    else:
        updater.stage.assert_not_called()
        updater.confirm_and_apply.assert_not_called()
        assert any("do not match readout_states" in call.args[0] for call in logger.call_args_list)
