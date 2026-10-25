"""Exercise each registered profile-proposal entry point with isolated files."""
import json
from types import SimpleNamespace

import pytest

from calibrations.core import BaseCalibration, CalibrationOptions
from calibrations.registry import CALIBRATIONS
from profiles import ProfileUpdater


PROFILE_KEYS = [key for key, entry in CALIBRATIONS.items()
                if entry.load_class().profile_updates is not BaseCalibration.profile_updates]


@pytest.mark.parametrize("key", PROFILE_KEYS)
@pytest.mark.parametrize("answer", ["y", "n", "", "EOF", "interrupt"])
def test_recovered_fit_proposals_write_only_after_confirmation(tmp_path, monkeypatch, key, answer):
    entry = CALIBRATIONS[key]
    profile = tmp_path / "profiles" / "example"
    profile.mkdir(parents=True)
    path = profile / "qubits.json"
    original = {"qubits": {
        name: {"frequencies_hz": {"qubit_f01": 5e9}} for name in ("q1", "q2")
    }}
    path.write_text(json.dumps(original))
    node = entry.load_class()(
        parameters=entry.load_parameters_class()(), machine=object(),
        profile_name="example", logger=lambda _: None,
        options=CalibrationOptions(apply_profile_update=True),
        profile_updater=ProfileUpdater(tmp_path / "updates", profile.parent),
    )
    node.namespace.update(acquisition_interrupted=True,
                          qubits=[SimpleNamespace(name=name) for name in ("q1", "q2")])
    node.results["fit_results"] = {
        "q1": {"success": True, "frequency": 4.9e9},
        "q2": {"success": False, "frequency": 4.8e9},
    }
    node.outcomes = {"q1": "successful", "q2": "failed"}
    # Use one common field to isolate the real proposal/confirmation override
    # from each experiment's independently tested fit-to-profile field mapping.
    node.profile_updates = lambda: node.qubit_profile_updates({"frequencies_hz.qubit_f01": "frequency"})
    prompts = []

    def respond(prompt):
        prompts.append(prompt)
        assert json.loads(path.read_text()) == original
        proposal = json.loads((node.namespace["profile_update_proposal"] / "proposal.json").read_text())
        assert [change["path"] for change in proposal["changes"]] == [
            "qubits.json.qubits.q1.frequencies_hz.qubit_f01"]
        if answer == "EOF":
            raise EOFError()
        if answer == "interrupt":
            raise KeyboardInterrupt()
        return answer

    monkeypatch.setattr("builtins.input", respond)
    if answer == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            node._propose_profile_update_from_options()
    else:
        assert node._propose_profile_update_from_options()
    assert len(prompts) == 1
    if answer == "y":
        original["qubits"]["q1"]["frequencies_hz"]["qubit_f01"] = 4.9e9
    assert json.loads(path.read_text()) == original
    proposal = json.loads((node.namespace["profile_update_proposal"] / "proposal.json").read_text())
    assert proposal["status"] == ("applied" if answer == "y" else "pending")
