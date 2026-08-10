from calibration_utils.T1.parameters import Parameters


def test_qubit_experiment_parameters_include_readout_mitigation_switch():
    parameters = Parameters()

    assert parameters.use_readout_mitigation is False
    parameters.use_readout_mitigation = True
    assert parameters.use_readout_mitigation is True
