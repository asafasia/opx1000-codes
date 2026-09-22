from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from calibration_utils.qubit_spectroscopy.parameters import Parameters
from calibration_utils.qubit_spectroscopy.analysis import process_raw_dataset, fit_raw_data
from quam_config import create_machine

module = import_module("calibrations.03a_qubit_spectroscopy")


@pytest.mark.parametrize("transition", ["ge", "ef"])
def test_target_centers_program_without_changing_profile_or_ge_preparation(transition):
    machine = create_machine(qubit="q6")
    q = machine.qubits["q6"]
    original = (q.xy.RF_frequency, q.xy.intermediate_frequency, q.f_01, q.f_12)
    node = module.QubitSpectroscopy(
        parameters=Parameters(transition=transition, target_frequency_in_mhz=4000,
                              frequency_span_in_mhz=10, frequency_step_in_mhz=1,
                              num_shots=2, use_state_discrimination=False), machine=machine)
    node.create_qua_program()
    assert node.namespace["scan_centers_hz"]["q6"] == pytest.approx(4e9)
    offset = module.transition_frequency_offset(q, transition, 4000)
    assert q.xy.RF_frequency + offset == pytest.approx(4e9)
    assert (q.xy.RF_frequency, q.xy.intermediate_frequency, q.f_01, q.f_12) == original


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_target_is_rejected(value):
    with pytest.raises(ValidationError):
        Parameters(target_frequency_in_mhz=value)


@pytest.mark.parametrize("transition", ["ge", "ef"])
@pytest.mark.parametrize("saved_center", [None, 4.2e9])
def test_analysis_and_fit_use_target_or_saved_center(transition, saved_center):
    q = SimpleNamespace(name="q6", f_01=4.8e9, f_12=4.5e9, anharmonicity=3e8,
        xy=SimpleNamespace(RF_frequency=4.8e9, operations={
            "saturation": SimpleNamespace(amplitude=0.2), "x180": SimpleNamespace(length=40)}),
        resonator=SimpleNamespace(operations={"readout": SimpleNamespace(integration_weights_angle=0)}))
    node = SimpleNamespace(namespace={"qubits": [q]}, parameters=Parameters(
        transition=transition, target_frequency_in_mhz=4000, use_state_discrimination=True,
        frequency_span_in_mhz=20, frequency_step_in_mhz=0.5))
    detuning = np.linspace(-10e6, 10e6, 41)
    signal = 0.1 + 0.8 * (1.8e6)**2 / ((detuning-1.37e6)**2 + (1.8e6)**2)
    ds = xr.Dataset({"state": (("qubit", "detuning"), signal[None, :])},
                    coords={"qubit": ["q6"], "detuning": detuning})
    if saved_center is not None:
        ds = ds.assign_coords(scan_center_frequency_hz=("qubit", [saved_center]))
    processed = process_raw_dataset(ds, node)
    center = 4e9 if saved_center is None else saved_center
    np.testing.assert_allclose(processed.full_freq.values, center+detuning[None, :])
    _, results = fit_raw_data(processed, node)
    assert results["q6"].frequency == pytest.approx(center+1.37e6, abs=1e3)


def test_none_keeps_existing_transition_offsets():
    q = SimpleNamespace(anharmonicity=3e8, xy=SimpleNamespace(RF_frequency=4.8e9))
    assert module.transition_frequency_offset(q, "ge", None) == 0
    assert module.transition_frequency_offset(q, "ef", None) == -3e8
