import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from profiles import ProfileUpdater

module = import_module("calibrations.06a_ramsey")


def make_node(offset=2e5, reference=5e9, updater=None):
    q = SimpleNamespace(name="q6", f_01=5.1e9, xy=SimpleNamespace(RF_frequency=5.1e9), T2ramsey=None)
    node = module.Ramsey(parameters=module.Parameters(), machine=SimpleNamespace(),
                         profile_name="single_qubit", profile_updater=updater, logger=lambda _: None)
    node.namespace["qubits"] = [q]
    node.results["ds_raw"] = xr.Dataset(coords={"qubit": ["q6"], "drive_frequency_hz": ("qubit", [reference])})
    node.results["fit_results"] = {"q6": {"success": True, "freq_offset": offset, "decay": 20e-6}}
    node._add_frequency_corrections()
    return node, q


@pytest.mark.parametrize("offset", [-2e5, 2e5])
def test_signed_correction_uses_acquisition_frequency_only_once(offset):
    node, q = make_node(offset)
    expected = 5e9 - offset
    node.update_state()
    node.update_state()
    assert q.f_01 == expected
    assert q.xy.RF_frequency == expected
    assert node.profile_updates()["qubits.json.qubits.q6.frequencies_hz.qubit_f01"] == expected
    assert node.results["fit_results"]["q6"]["frequency_correction_hz"] == -offset


def test_staging_only_does_not_ask_to_apply():
    updater = MagicMock()
    node, _ = make_node(updater=updater)
    assert node.propose_profile_update(apply=False)
    updater.stage.assert_called_once()
    updater.confirm_and_apply.assert_not_called()


@pytest.mark.parametrize("answer, expected", [("no", 5e9), ("yes", 5e9 - 2e5)])
def test_profile_frequency_changes_only_after_yes(tmp_path, answer, expected):
    root = tmp_path / "profiles" / "single_qubit"
    root.mkdir(parents=True)
    qubits = root / "qubits.json"
    qubits.write_text(json.dumps({"qubits": {"q6": {"frequencies_hz": {"qubit_f01": 5e9}}}}))
    (root / "metrics.json").write_text(json.dumps({"qubits": {"q6": {"coherence": {"t2_ramsey_ns": 10e-6}}}}))
    updater = ProfileUpdater(output_root=tmp_path / "proposals", profiles_root=root.parent)
    node, _ = make_node(updater=updater)
    with patch("builtins.input", return_value=answer) as prompt:
        assert node.propose_profile_update()
    prompt.assert_called_once()
    assert json.loads(qubits.read_text())["qubits"]["q6"]["frequencies_hz"]["qubit_f01"] == expected


@pytest.mark.parametrize("offset", [float("nan"), float("inf"), 6e9])
def test_invalid_frequency_correction_not_proposed(offset):
    node, _ = make_node(offset)
    assert not any("qubit_f01" in key for key in node.profile_updates())


def test_failed_fit_not_proposed():
    node, _ = make_node()
    node.results["fit_results"]["q6"]["success"] = False
    assert node.profile_updates() == {}


def test_legacy_data_without_reference_does_not_guess_from_current_profile():
    node, _ = make_node()
    node.results["ds_raw"] = node.results["ds_raw"].drop_vars("drive_frequency_hz")
    node.results["fit_results"] = {"q6": {"success": True, "freq_offset": 2e5, "decay": 20e-6}}
    node._add_frequency_corrections()
    assert not any("qubit_f01" in key for key in node.profile_updates())


@pytest.mark.parametrize("offset", [-5e4, 5e4])
def test_synthetic_ramsey_detuning_gives_correct_frequency_proposal(offset):
    node, _ = make_node()
    node.parameters.use_state_discrimination = True
    node.parameters.frequency_detuning_in_mhz = 0.5
    times = np.linspace(16, 6000, 151)
    signals = [0.5 + 0.35 * np.exp(-times / 10000) * np.cos(2*np.pi*(5e5 + sign*offset)*1e-9*times + 0.1)
               for sign in (-1, 1)]
    node.results["ds_raw"] = xr.Dataset(
        {"state": (("qubit", "detuning_signs", "idle_time"), np.asarray([signals]))},
        coords={"qubit": ["q6"], "detuning_signs": [-1, 1], "idle_time": times,
                "drive_frequency_hz": ("qubit", [5e9])},
    )
    node.analyse_data()
    result = node.results["fit_results"]["q6"]
    assert result["success"]
    assert result["freq_offset"] == pytest.approx(offset, abs=100)
    assert node.profile_updates()["qubits.json.qubits.q6.frequencies_hz.qubit_f01"] == pytest.approx(5e9-offset, abs=100)
