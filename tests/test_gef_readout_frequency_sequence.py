from quam_config import create_machine

def test_gef_readout_pulse_has_independent_integration_weights():
    q = create_machine(profile_name="single_qubit", qubit="q1").qubits["q1"]
    ge = q.resonator.operations["readout"]
    gef = q.resonator.operations["readout_GEF"]
    before = [list(row) for row in ge.integration_weights]
    gef.integration_weights = None  # Clear the automatic QuAM reference before overriding it.
    gef.integration_weights = [[.5, gef.length]]
    assert [list(row) for row in ge.integration_weights] == before
    assert gef.integration_weights_angle == 0
