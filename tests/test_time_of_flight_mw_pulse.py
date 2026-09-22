import importlib

import numpy as np
import pytest
from pydantic import ValidationError

from calibration_utils.time_of_flight_mw.parameters import Parameters
from calibration_utils.time_of_flight_mw.pulses import with_gaussian_edges
from quam.components.pulses import SquareReadoutPulse
from quam_config import create_machine
from calibrations.core import CalibrationOptions


def test_gaussian_readout_preserves_settings_and_peak():
    original = SquareReadoutPulse(
        length=2000, amplitude=0.6, axis_angle=0.3,
        threshold=0.01, integration_weights=[(1, 2000)],
        integration_weights_angle=0.4,
    )
    shaped = with_gaussian_edges(original, 100)
    waveform = np.asarray(shaped.waveform_function())
    assert waveform.shape == (2000,)
    np.testing.assert_allclose(waveform, waveform[::-1])
    np.testing.assert_allclose(waveform[100:1900], 0.6 * np.exp(0.3j))
    assert abs(waveform[0]) < 1e-4
    assert np.max(abs(waveform)) <= 0.7
    assert shaped.integration_weights_function() == original.integration_weights_function()
    assert shaped.threshold == original.threshold
    assert original.waveform_function() == 0.6 * np.exp(0.3j)


def test_gaussian_edges_require_a_flat_top():
    with pytest.raises(ValidationError, match="positive flat-top"):
        Parameters(readout_length_in_ns=200, readout_edge_length_in_ns=100)
    Parameters(readout_pulse_shape="square", readout_length_in_ns=200)


@pytest.mark.parametrize("shape", ["flat_top_gaussian", "square"])
def test_time_of_flight_builds_config_without_hardware(shape, tmp_path, monkeypatch):
    module = importlib.import_module("calibrations.01b_time_of_flight_mw_fem")
    # Keep the generated debug script out of the user's existing debug directory.
    monkeypatch.setattr(module, "Path", lambda *_: tmp_path / "calibrations" / "tof.py")
    machine = create_machine(profile_name="single_qubit", qubit="q6")
    original_tof = machine.qubits["q6"].resonator.time_of_flight
    original = machine.qubits["q6"].resonator.operations["readout"].to_dict()
    calibration = module.TimeOfFlightMwFem(
        parameters=Parameters(readout_pulse_shape=shape), machine=machine,
        options=CalibrationOptions(
            save_raw_data=False, save_figures=False, plot_data=False,
            update_state=False, propose_profile_update=False, apply_profile_update=False,
        ),
    )
    calibration.create_qua_program()
    config = machine.generate_config()
    pulse = machine.qubits["q6"].resonator.operations["readout"]
    assert abs(pulse.amplitude) <= 0.7
    assert pulse.length == 2000
    element = config["elements"][machine.qubits["q6"].resonator.name]
    configured_pulse = config["pulses"][element["operations"]["readout"]]
    assert configured_pulse["operation"] == "measurement"
    if shape == "flat_top_gaussian":
        configured_waveforms = [config["waveforms"][name] for name in configured_pulse["waveforms"].values()]
        assert any(waveform["type"] == "arbitrary" for waveform in configured_waveforms)
    if shape == "flat_top_gaussian":
        assert pulse.flat_length == 1800
        waveform = np.asarray(pulse.waveform_function())
        assert len(waveform) == 2000
        # Even with a successful fit, a shaped onset must not update the TOF.
        calibration.results["fit_results"] = {"q6": {"success": True, "tof_to_add": 100}}
        calibration.update_state()
        assert machine.qubits["q6"].resonator.time_of_flight == original_tof
        assert machine.qubits["q6"].resonator.operations["readout"].to_dict() == original
    calibration.cleanup()
    assert machine.qubits["q6"].resonator.operations["readout"].to_dict() == original
