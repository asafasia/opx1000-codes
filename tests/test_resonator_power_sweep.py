"""Power sweeps preserve the configured MW-FEM gain; no hardware connections."""

from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from qm import generate_qua_script
from quam_builder.tools.power_tools import get_output_power_mw_channel

from quam_config.create_machine_from_profile import create_machine_from_profile


module = import_module("calibrations.02b_resonator_spectroscopy_vs_power")


@pytest.mark.parametrize("gain", [-2, 4, 10])
def test_default_sweep_reaches_current_gain_and_fit_keeps_it(gain):
    machine = create_machine_from_profile("Or_profile", qubit="q1a", save=False)
    q = machine.qubits["q1a"]
    q.resonator.opx_output.full_scale_power_dbm = gain
    parameters = module.Parameters(num_shots=2, num_power_points=3,
                                   frequency_span_in_mhz=2, frequency_step_in_mhz=1)
    node = module.ResonatorSpectroscopyVsPower(parameters=parameters, machine=machine)
    try:
        program = node.create_qua_program()
        assert "stream_processing" in generate_qua_script(program)
        powers = node.namespace["sweep_axes"]["power"].values
        np.testing.assert_allclose(powers, [-50, (-50 + gain) / 2, gain])
        assert q.resonator.opx_output.full_scale_power_dbm == gain
        assert q.resonator.operations["readout"].amplitude == 1
        assert get_output_power_mw_channel(q.resonator, "readout") == pytest.approx(gain)
        node.record_state_updates = nullcontext
        node.outcomes = {q.name: "successful"}
        node.results["fit_results"] = {
            q.name: {"optimal_power": gain - 6.020599913279624, "frequency_shift": 0}
        }
        node.update_state()
        assert q.resonator.opx_output.full_scale_power_dbm == gain
        assert q.resonator.operations["readout"].amplitude == pytest.approx(0.5)
    finally:
        for resonator in node.namespace.get("tracked_resonators", []):
            resonator.revert_changes()


def qubit(name, gain):
    return SimpleNamespace(name=name, resonator=SimpleNamespace(
        opx_output=SimpleNamespace(full_scale_power_dbm=gain)))


def test_explicit_lower_maximum_preserves_different_gains():
    maximum, gains, amplitudes = module.resolve_power_sweep(
        [qubit("q1", 10), qubit("q2", 4)], module.Parameters(max_power_dbm=-5)
    )
    assert maximum == -5
    assert gains == {"q1": 10, "q2": 4}
    for name in gains:
        assert gains[name] + 20 * np.log10(amplitudes[name]) == pytest.approx(-5)


@pytest.mark.parametrize("parameters, message", [
    ({"max_power_dbm": 11}, "gain will not be changed"),
    ({"max_amp": 0.8}, "gain will not be changed"),
    ({"min_power_dbm": 10}, "min_power_dbm"),
])
def test_unreachable_range_is_rejected(parameters, message):
    with pytest.raises(ValueError, match=message):
        module.resolve_power_sweep([qubit("q1", 10)], module.Parameters(**parameters))


def test_different_gains_require_separate_default_sweeps():
    with pytest.raises(ValueError, match="Run them separately"):
        module.resolve_power_sweep([qubit("q1", 10), qubit("q2", 4)], module.Parameters())


@pytest.mark.parametrize("parameters", [{"max_amp": 0}, {"max_amp": 1.1},
    {"num_power_points": 1}, {"max_power_dbm": float("nan")}])
def test_invalid_sweep_parameters_are_rejected(parameters):
    with pytest.raises(ValidationError):
        module.Parameters(**parameters)
