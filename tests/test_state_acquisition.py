from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from calibration_utils.state_acquisition import calculate_populations
from quam_config import create_machine


CASES = [
    ("qubit_spectroscopy", "03a_qubit_spectroscopy", "QubitSpectroscopy"),
    ("rabi_chevron", "04a_rabi_chevron", "RabiChevron"),
]


@pytest.mark.parametrize("states", [["g", "e"], ["g", "e", "f"]])
@pytest.mark.parametrize("chevron", [False, True])
def test_population_reduction_preserves_shots_and_sweep_axes(states, chevron):
    dims = ("qubit", "shot", "detuning") + (("pulse_duration",) if chevron else ())
    shape = (2, 6, 3) + ((4,) if chevron else ())
    labels = np.arange(np.prod(shape)).reshape(shape) % len(states)
    labels[:, 1::2] = 0
    ds = xr.Dataset({"state": (dims, labels)}, attrs={"readout_states": states})
    params = SimpleNamespace(readout_states=states)
    result = calculate_populations(ds, params)
    np.testing.assert_array_equal(result.state_shots, labels)
    np.testing.assert_array_equal(ds.state, labels)
    for i, label in enumerate(states):
        np.testing.assert_allclose(result[f"population_{label}"], (labels == i).mean(axis=1))
    np.testing.assert_allclose(sum(result[f"population_{s}"] for s in states), 1)
    xr.testing.assert_equal(result.state, result.population_e.rename("state"))
    assert "shot" not in result.state.dims
    # Re-analysis must not replace populations after mitigation.
    result["state"] = result.state + 0.01
    assert calculate_populations(result, params) is result


@pytest.mark.parametrize("labels", [[0, 2], [0, 0.5], [0, np.nan], []])
def test_invalid_binary_labels_are_rejected(labels):
    ds = xr.Dataset({"state": ("shot", labels)})
    with pytest.raises(ValueError, match="integer state labels"):
        calculate_populations(ds, SimpleNamespace(readout_states=["g", "e"]))


@pytest.mark.parametrize("utils_name,module_name,class_name", CASES)
@pytest.mark.parametrize("acquisition", ["averaged", "single_shot"])
@pytest.mark.parametrize("states", [["g", "e"], ["g", "e", "f"]])
def test_program_builds_with_correct_shot_axes(utils_name, module_name, class_name, acquisition, states):
    params_cls = import_module(f"calibration_utils.{utils_name}.parameters").Parameters
    params = params_cls(acquisition=acquisition, num_shots=3, readout_states=states,
                        frequency_span_in_mhz=6, frequency_step_in_mhz=3,
                        use_state_discrimination=True, reset_type="thermal")
    node = getattr(import_module(f"calibrations.{module_name}"), class_name)(
        parameters=params, machine=create_machine(qubit="q6"))
    if len(states) == 3:
        # Synthetic IQ centers for offline program generation only.
        node.machine.qubits["q6"].resonator.readout_gef["gef_centers"] = [
            [0.0, 0.0], [0.01, 0.0], [0.0, 0.01],
        ]
        node.machine.qubits["q6"].resonator.readout_gef["calibration_signature"] = None
    try:
        program = node.create_qua_program()
        assert program is not None
        axes = node.namespace["sweep_axes"]
        assert ("shot" in axes) == (acquisition == "single_shot")
        if acquisition == "single_shot":
            assert list(axes)[:3] == ["qubit", "shot", "detuning"]
            assert len(axes["shot"]) == 3
            ds = xr.Dataset({"state": (("qubit", "shot", "detuning"), np.zeros((1, 3, 2), dtype=int))})
            annotated = node.annotate_readout_dataset(ds)
            assert "population_state" not in annotated.state.attrs
            assert annotated.attrs["acquisition"] == "single_shot"
    finally:
        for q in node.namespace.get("tracked_qubits", []):
            q.revert_changes()


@pytest.mark.parametrize("utils_name,module_name,class_name", CASES)
def test_single_shot_supports_analog_acquisition(utils_name, module_name, class_name):
    params_cls = import_module(f"calibration_utils.{utils_name}.parameters").Parameters
    assert params_cls().acquisition == "averaged"
    node = getattr(import_module(f"calibrations.{module_name}"), class_name)(
        parameters=params_cls(acquisition="single_shot", use_state_discrimination=False),
        machine=create_machine(qubit="q6"))
    try:
        from qm import generate_qua_script
        script = generate_qua_script(node.create_qua_program())
        assert 'save_all("I1")' in script and 'save_all("Q1")' in script
        assert 'save("state1")' not in script
        assert "shot" in node.namespace["sweep_axes"]
    finally:
        for q in node.namespace.get("tracked_qubits", []):
            q.revert_changes()


@pytest.mark.parametrize("utils_name,module_name,class_name", CASES)
@pytest.mark.parametrize("states", [["g", "e"], ["g", "e", "f"]])
def test_mitigation_receives_populations_and_keeps_labels(utils_name, module_name, class_name, states, monkeypatch):
    from calibrations.core import BaseCalibration
    params_cls = import_module(f"calibration_utils.{utils_name}.parameters").Parameters
    node = getattr(import_module(f"calibrations.{module_name}"), class_name)(
        parameters=params_cls(acquisition="single_shot", use_state_discrimination=True,
                              readout_states=states, use_readout_mitigation=True), machine=create_machine(qubit="q6"))
    labels = np.array([0, 1, len(states)-1, 0]).reshape(1, 4, 1)
    node.results["ds_raw"] = xr.Dataset(
        {"state": (("qubit", "shot", "detuning"), labels)})
    def mitigate(self, ds, qubits, strength):
        assert "shot" not in ds.state.dims
        np.testing.assert_array_equal(ds.state_shots, labels)
        for label in states:
            assert f"population_{label}" in ds
        self.results["ds_raw"]["state"] = ds.state + 0.02
    monkeypatch.setattr(BaseCalibration, "_mitigate_population_dataset", mitigate)
    node.apply_readout_mitigation()
    ds = node.results["ds_raw"]
    np.testing.assert_allclose(ds.state, (labels == 1).mean(axis=1) + 0.02)
    assert calculate_populations(ds, node.parameters) is ds
