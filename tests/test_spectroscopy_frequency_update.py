"""Spectroscopy persistence checks using isolated profiles and no hardware."""
import json
from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace

import pytest

from calibrations import CalibrationOptions
from profiles import ProfileUpdater


module = import_module("calibrations.03a_qubit_spectroscopy")


def make_node(transition="ge", *, updater=None, apply=True, success=True):
    qubit = SimpleNamespace(name="q1", f_01=4.4e9, f_12=4.1e9,
                            anharmonicity=3e8, xy=SimpleNamespace(RF_frequency=4.4e9))
    node = module.QubitSpectroscopy(
        parameters=module.Parameters(transition=transition), machine=object(),
        profile_name="example", profile_updater=updater,
        options=CalibrationOptions(apply_profile_update=apply), logger=lambda _: None,
    )
    node.namespace["qubits"] = [qubit]
    node.outcomes = {"q1": "successful" if success else "failed"}
    node.results["fit_results"] = {"q1": {"success": success, "frequency": 4.2e9}}
    node.record_state_updates = nullcontext
    return node, qubit


@pytest.mark.parametrize("transition", ["ge", "ef"])
@pytest.mark.parametrize("success", [True, False])
def test_memory_updates_only_successful_transition(transition, success):
    node, qubit = make_node(transition, success=success)
    node.update_state()
    assert qubit.f_01 == (4.2e9 if success and transition == "ge" else 4.4e9)
    assert qubit.xy.RF_frequency == qubit.f_01
    assert qubit.f_12 == (4.2e9 if success and transition == "ef" else 4.1e9)
    assert qubit.anharmonicity == (2e8 if success and transition == "ef" else 3e8)


def test_ef_updates_preserve_negative_anharmonicity():
    node, qubit = make_node("ef")
    qubit.anharmonicity = -3e8
    updates = node.profile_updates()
    assert updates["qubits.json.qubits.q1.transmon.anharmonicity_hz"] == -2e8
    node.update_state()
    assert qubit.f_12 == 4.2e9
    assert qubit.anharmonicity == -2e8


@pytest.mark.parametrize("apply", [True, False])
@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("interrupted", [True, False])
@pytest.mark.parametrize("answer", ["y", "n", ""])
def test_ge_profile_persistence_respects_options_and_fit(tmp_path, monkeypatch, apply, success, interrupted, answer):
    root = tmp_path / "profiles" / "example"
    root.mkdir(parents=True)
    qubits = root / "qubits.json"
    original = {"qubits": {"q1": {"frequencies_hz": {"qubit_f01": 4.4e9, "qubit_f12": 4.1e9}}}}
    qubits.write_text(json.dumps(original))
    updater = ProfileUpdater(output_root=tmp_path / "proposals", profiles_root=root.parent)
    node, _ = make_node(updater=updater, apply=apply, success=success)

    prompts = []

    def respond(prompt):
        prompts.append(prompt)
        # The profile must still contain the original value at confirmation.
        assert json.loads(qubits.read_text()) == original
        return answer

    monkeypatch.setattr("builtins.input", respond)
    # Exercise the shared run lifecycle after a completed/recovered acquisition,
    # with only hardware, processing, and artifact generation stubbed out.
    node.options.save_raw_data = False
    node.options.save_analysis_result = False
    node.options.save_figures = False
    node.options.plot_data = False
    node.options.report_runtime_estimate = False
    node.create_qua_program = lambda: "program"
    node.automatic_dc_bias = nullcontext
    node.execute_qua_program = lambda: node.namespace.update(acquisition_interrupted=interrupted)
    node.prepare_acquisition_results = lambda: None
    node.analyse_data = lambda: None
    monkeypatch.setattr("calibrations.core.base.assert_outputs_allowed", lambda: None)
    status = node.run()
    assert status.interrupted == interrupted
    assert status.profile_update_proposed == success
    assert node.namespace["qubits"][0].f_01 == (4.2e9 if success else 4.4e9)
    assert len(prompts) == int(success and apply)
    actual = json.loads(qubits.read_text())
    accepted = apply and answer == "y"
    if success and accepted:
        original["qubits"]["q1"]["frequencies_hz"]["qubit_f01"] = 4.2e9
    assert actual == original
    if success:
        proposal = node.namespace["profile_update_proposal"]
        assert json.loads((proposal / "proposal.json").read_text())["status"] == (
            "applied" if accepted else "pending")
        if accepted:
            backup = json.loads((proposal / "profile_before_update" / "qubits.json").read_text())
            assert backup["qubits"]["q1"]["frequencies_hz"]["qubit_f01"] == 4.4e9
