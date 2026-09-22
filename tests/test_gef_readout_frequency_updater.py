from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def test_gef_readout_frequency_stages_profile_shift_update():
    source = (
        REPOSITORY_ROOT
        / "calibrations"
        / "14_gef_readout_frequency_optimization.py"
    ).read_text()

    assert "def profile_updates" in source
    assert "readout_gef.frequency_hz" in source
    assert "resonator.GEF_frequency_shift =" in source


def test_profile_loader_applies_gef_readout_shift():
    source = (REPOSITORY_ROOT / "quam_config" / "populate_quam_lf_mw_fems.py").read_text()
    validation_source = (REPOSITORY_ROOT / "profiles" / "loader.py").read_text()

    assert '"gef_frequency_shift_hz"' in source
    assert "qubit.resonator.GEF_frequency_shift" in source
    assert "readout.gef_frequency_shift_hz must be numeric" in validation_source


def test_proposal_does_not_depend_on_update_state_and_does_not_double_shift():
    from importlib import import_module
    from quam_config import create_machine
    module = import_module("calibrations.14_gef_readout_frequency_optimization")
    machine = create_machine(profile_name="single_qubit", qubit="q1")
    q = machine.qubits["q1"]
    original = q.resonator.readout_gef["frequency_hz"]
    ge = q.resonator.RF_frequency
    node = module.GefReadoutFrequencyOptimization(parameters=module.Parameters(), machine=machine)
    node.namespace["qubits"] = [q]
    node.results["fit_results"] = {"q1":{"optimal_detuning":100000}}
    node.outcomes = {"q1":"successful"}
    before = node.profile_updates()
    node.update_state()
    assert node.profile_updates() == before
    assert q.resonator.readout_gef["frequency_hz"] == original + 100000
    assert q.resonator.RF_frequency == ge
