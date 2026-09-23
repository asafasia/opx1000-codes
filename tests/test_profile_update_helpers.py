"""Shared profile update behavior; no controller access or real profile writes."""
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from calibrations import CalibrationOptions


PULSE_CASES = [
    ("04b_power_rabi", "PowerRabi", {"transition": "ge"}, {"opt_amp": .23}, "ge", "amplitude", .23),
    ("04b_power_rabi", "PowerRabi", {"transition": "ef"}, {"opt_amp": .13}, "ef", "amplitude", .13),
    ("04e_fine_rabi_calibration", "FineRabiCalibration", {}, {"optimal_amp_prefactor": 1.1}, "ge", "amplitude", .22),
    ("10b_drag_calibration_180_minus_180", "DragCalibration180Minus180", {}, {"alpha": .31}, "drag", "beta", .31),
]


@pytest.fixture
def pulse_profile(monkeypatch):
    data = {
        "qubits": {"qubits": {name: {"operations": {
            "x180": "ge", "EF_x180": "ef", "x180_drag": "drag",
        } if name != "q3" else {}} for name in ("q1", "q2", "q3")}},
        "pulses": {"pulses": {name: {
            "ge": {"type": "gaussian", "amplitude": .2},
            "ef": {"type": "gaussian", "amplitude": .3},
            "drag": {"type": "drag", "amplitude": .2, "beta": .1},
        } for name in ("q1", "q2", "q3")}},
    }
    loader = Mock(return_value=data)
    monkeypatch.setattr("profiles.load_profile", loader)
    return data, loader


def make_node(module_name, class_name, parameters=None, fit=None):
    module = import_module(f"calibrations.{module_name}")
    node = getattr(module, class_name)(parameters=module.Parameters(**(parameters or {})),
        machine=object(), profile_name="example", profile_updater=Mock(), logger=Mock(),
        options=CalibrationOptions(apply_profile_update=False))
    node.namespace["qubits"] = [SimpleNamespace(name=name, f_01=5e9) for name in ("q1", "q2", "q3")]
    node.outcomes = {"q1": "successful", "q2": "failed", "q3": "successful"}
    node.results["fit_results"] = {name: dict(fit or {}) for name in ("q1", "q2", "q3")}
    return node


@pytest.mark.parametrize("module, cls, params, fit, pulse, field, expected", PULSE_CASES)
def test_pulse_updates_resolve_dedicated_pulses_and_only_stage_successful_fits(
    pulse_profile, module, cls, params, fit, pulse, field, expected,
):
    profile, loader = pulse_profile
    before = deepcopy(profile)
    node = make_node(module, cls, params, fit)
    updates = node.profile_updates()
    assert updates == {f"pulses.json.pulses.q1.{pulse}.{field}": pytest.approx(expected)}
    loader.assert_called_once_with("example")
    assert profile == before
    assert node._propose_profile_update_from_options()
    node.profile_updater.stage.assert_called_once_with(node.name, updates, profile_name="example")
    node.profile_updater.confirm_and_apply.assert_not_called()
    assert node.namespace["profile_update_proposal"] is node.profile_updater.stage.return_value


@pytest.mark.parametrize("amplitude", [np.nan, np.inf, .701, -.701])
def test_pulse_updates_reject_invalid_amplitudes(pulse_profile, amplitude):
    node = make_node("04b_power_rabi", "PowerRabi", fit={"opt_amp": amplitude})
    assert node.profile_updates() == {}


@pytest.mark.parametrize("fit", [{}, {"optimal_amp_prefactor": -1}, {"optimal_amp_prefactor": 0},
                                      {"optimal_amp_prefactor": np.nan}, {"optimal_amp_prefactor": 4}])
def test_fine_rabi_skips_missing_or_invalid_scale(pulse_profile, fit):
    node = make_node("04e_fine_rabi_calibration", "FineRabiCalibration", fit=fit)
    assert node.profile_updates() == {}


def test_fine_rabi_scales_current_profile_not_mutated_machine(pulse_profile):
    profile, _ = pulse_profile
    node = make_node("04e_fine_rabi_calibration", "FineRabiCalibration", fit={"optimal_amp_prefactor": 1.1})
    assert node.profile_updates()["pulses.json.pulses.q1.ge.amplitude"] == pytest.approx(.22)
    profile["pulses"]["pulses"]["q1"]["ge"]["amplitude"] = .1
    assert node.profile_updates()["pulses.json.pulses.q1.ge.amplitude"] == pytest.approx(.11)


def test_drag_requires_drag_pulse_type(pulse_profile):
    profile, _ = pulse_profile
    profile["pulses"]["pulses"]["q1"]["drag"]["type"] = "gaussian"
    node = make_node("10b_drag_calibration_180_minus_180", "DragCalibration180Minus180", fit={"alpha": .31})
    assert node.profile_updates() == {}


@pytest.mark.parametrize("transition", ["ge", "ef"])
def test_spectroscopy_preserves_transition_specific_paths(transition):
    node = make_node("03a_qubit_spectroscopy", "QubitSpectroscopy", {"transition": transition}, {"frequency": 4.8e9})
    node.outcomes["q3"] = "failed"
    if transition == "ge":
        expected = {"qubits.json.qubits.q1.frequencies_hz.qubit_f01": 4.8e9}
    else:
        expected = {"qubits.json.qubits.q1.frequencies_hz.qubit_f12": 4.8e9,
                    "qubits.json.qubits.q1.transmon.anharmonicity_hz": .2e9}
    assert node.profile_updates() == expected
    assert node._propose_profile_update_from_options()
    node.profile_updater.confirm_and_apply.assert_not_called()


def test_echo_metric_uses_fitted_dataset_and_skips_failed_qubits():
    node = make_node("06b_echo", "Echo")
    node.outcomes["q3"] = "failed"
    node.results["ds_fit"] = xr.Dataset({"T2_echo": ("qubit", [12000, 15000, 18000])},
                                          coords={"qubit": ["q1", "q2", "q3"]})
    assert node.profile_updates() == {"metrics.json.qubits.q1.coherence.t2_echo_ns": 12000.0}


@pytest.mark.parametrize("states, section", [(["g", "e"], "readout"), (["g", "e", "f"], "readout_gef")])
def test_readout_weights_invalidate_only_selected_discriminator(states, section):
    node = make_node("10d_readout_weights_optimization", "ReadoutWeightsOptimization", {"readout_states": states})
    node.outcomes["q3"] = "failed"
    assert node.profile_updates() == {
        f"qubits.json.qubits.q1.{section}.use_kernel": True,
        f"qubits.json.qubits.q1.{section}.gef_centers": None,
        f"qubits.json.qubits.q1.{section}.confusion_matrix": None,
    }


def test_explicit_failed_outcome_wins_over_fit_success():
    node = make_node("03a_qubit_spectroscopy", "QubitSpectroscopy", fit={"success": True, "frequency": 5e9})
    node.outcomes = {name: "failed" for name in node.outcomes}
    assert node.profile_updates() == {}
