from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from qm import generate_qua_script

from quam_config.components import ArduinoDCBias
from quam_config.create_machine_from_profile import create_machine_from_profile


module = import_module("calibrations.03d_qubit_spectroscopy_vs_external_flux")


@pytest.fixture
def calibration():
    machine = create_machine_from_profile("single_qubit", save=False, qubit="q3")
    machine.dc_bias.qubit_biases_v["q3"] = 0.003
    calibration = module.QubitSpectroscopyVsExternalFlux(
        parameters=module.Parameters(num_shots=2, num_flux_points=3,
                                    frequency_span_in_mhz=2, frequency_step_in_mhz=0.5,
                                    bias_settle_time_s=0),
        machine=machine,
    )
    calibration.namespace["qua_program"] = calibration.create_qua_program()
    return calibration


def test_program_has_external_pauses_and_separate_voltage_averages(calibration):
    script = generate_qua_script(calibration.namespace["qua_program"])
    assert script.count("pause()") == 2  # loop start and final measurement-free pause
    assert "FUNCTIONS.average(0)" in script
    assert 'save_all("I")' in script
    assert 'save_all("Q")' in script
    assert ".z" not in script
    np.testing.assert_allclose(calibration.namespace["sweep_axes"]["flux_bias"], [0.002, 0.003, 0.004])
    assert not calibration.options.apply_profile_update


def test_simulation_build_omits_pauses_and_never_connects_source(calibration):
    calibration.parameters.simulate = True
    with patch.object(ArduinoDCBias, "connect") as connect:
        script = generate_qua_script(calibration.create_qua_program())
    assert "pause()" not in script
    connect.assert_not_called()


def test_unsafe_sweep_fails_before_hardware_access(calibration):
    calibration.parameters.flux_bias_center_in_v = calibration.machine.dc_bias.max_abs_voltage_v
    with patch.object(ArduinoDCBias, "connect") as connect:
        with pytest.raises(ValueError, match="Every swept voltage"):
            calibration.create_qua_program()
    connect.assert_not_called()


def test_sweep_obeys_profile_limit_without_another_ceiling(calibration):
    calibration.machine.dc_bias.max_abs_voltage_v = 0.2
    calibration.parameters.flux_bias_center_in_v = 0.15
    with patch.object(ArduinoDCBias, "connect") as connect:
        calibration.create_qua_program()
    np.testing.assert_allclose(calibration.namespace["sweep_axes"]["flux_bias"], [0.149, 0.15, 0.151])
    connect.assert_not_called()


def mock_execution(calibration, events, *, fail_resume=None):
    job = MagicMock()
    state = {"row": -1, "paused": True}

    def resume():
        state["row"] += 1
        events.append(("resume", state["row"]))
        if state["row"] == fail_resume:
            raise RuntimeError("acquisition failed")
        state["paused"] = False

    def is_paused():
        events.append(("pause", state["row"]))
        # A result becomes available while running, then we reach the next pause.
        state["paused"] = True
        return True

    job.resume.side_effect = resume
    job.is_paused.side_effect = is_paused
    job.halt.side_effect = lambda: events.append(("halt",))
    handles = {}
    for name, offset in (("I", 0), ("Q", 100)):
        handle = MagicMock()
        handle.count_so_far.side_effect = lambda: state["row"] + 1
        handle.fetch.side_effect = lambda index, timeout, offset=offset: {
            "value": np.arange(4) + 10 * index + offset
        }
        handles[name] = handle
    job.result_handles.get.side_effect = handles.get

    @contextmanager
    def session(*args, **kwargs):
        yield SimpleNamespace(execute=lambda program: job)

    controller = MagicMock()

    def set_voltage(channel, voltage):
        assert state["paused"] or voltage == 0
        events.append(("voltage", voltage))

    controller.set_voltage.side_effect = set_voltage
    controller.close.side_effect = lambda: events.append(("close",))
    calibration.machine = SimpleNamespace(
        dc_bias=calibration.machine.dc_bias,
        connect=lambda: None, generate_config=lambda: {},
    )
    return job, handles, session, controller


def test_execution_orders_voltage_pause_rows_and_cleanup(calibration):
    events = []
    job, handles, session, controller = mock_execution(calibration, events)
    with patch.object(module, "qm_session", session), \
         patch.object(module, "assert_outputs_allowed"), \
         patch("quam_config.components.dc_bias.open_controller", return_value=controller):
        calibration.execute_qua_program()
    assert events[0] == ("pause", -1)
    assert [event for event in events if event[0] == "voltage"] == [
        ("voltage", 0.002), ("voltage", 0.003), ("voltage", 0.004), ("voltage", 0.0),
    ]
    assert events[-3:] == [("halt",), ("voltage", 0.0), ("close",)]
    assert job.resume.call_count == 3
    for handle in handles.values():
        assert [call.args[0] for call in handle.fetch.call_args_list] == [0, 1, 2]
    ds = calibration.results["ds_raw"]
    assert ds.I.dims == ("qubit", "detuning", "flux_bias")
    np.testing.assert_array_equal(ds.I[0, :, 2], [20, 21, 22, 23])
    np.testing.assert_array_equal(ds.Q[0, :, 1], [110, 111, 112, 113])
    assert ds.flux_bias.attrs["units"] == "V"


def test_failed_resume_halts_and_zeros_source(calibration):
    events = []
    job, _, session, controller = mock_execution(calibration, events, fail_resume=1)
    with patch.object(module, "qm_session", session), \
         patch.object(module, "assert_outputs_allowed"), \
         patch("quam_config.components.dc_bias.open_controller", return_value=controller):
        with pytest.raises(RuntimeError, match="acquisition failed"):
            calibration.execute_qua_program()
    assert ("voltage", 0.0) in events
    assert events.index(("halt",)) < events.index(("voltage", 0.0))
    controller.close.assert_called_once()
    assert calibration.results["ds_raw"].sizes["flux_bias"] == 1


def test_wait_timeout_is_bounded(calibration):
    with patch.object(module, "assert_outputs_allowed"), \
         patch.object(module.time, "monotonic", side_effect=[0, 301]):
        with pytest.raises(TimeoutError, match="initial pause"):
            calibration._wait_for(lambda: False, "initial pause")


def test_small_scan_still_processes_and_plots_without_proposal(calibration):
    rows = {"I": [[1, 2, 5, 1]] * 3, "Q": [[1, 2, 3, 1]] * 3}
    calibration.results["ds_raw"] = calibration._dataset(rows, 3)
    calibration.analyse_data()
    with patch.object(module.plt, "show"):
        calibration.plot_data()
    assert "IQ_abs" in calibration.results["ds_raw"]
    assert calibration.outcomes == {"q3": "failed"}
    assert calibration.profile_updates() == {}
    module.plt.close("all")


def test_timeout_during_sweep_halts_and_zeros(calibration):
    events = []
    job, handles, session, controller = mock_execution(calibration, events)
    handles["Q"].count_so_far.side_effect = lambda: 0
    calibration.parameters.pause_timeout_s = 0.001
    with patch.object(module, "qm_session", session), \
         patch.object(module, "assert_outputs_allowed"), \
         patch("quam_config.components.dc_bias.open_controller", return_value=controller):
        with pytest.raises(TimeoutError, match="I/Q results"):
            calibration.execute_qua_program()
    assert job.resume.call_count == 1
    assert ("voltage", 0.0) in events
    controller.close.assert_called_once()


def test_saved_map_round_trip_keeps_dimensions_and_frequency(calibration, tmp_path):
    rows = {"I": [[1, 2, 5, 1]] * 3, "Q": [[1, 2, 3, 1]] * 3}
    expected = calibration._dataset(rows, 3)
    np.savez(tmp_path / "sweep.npz", **{name: value.values for name, value in expected.coords.items()})
    np.savez(tmp_path / "results.npz", **{name: value.values for name, value in expected.data_vars.items()})
    calibration.machine.dc_bias.qubit_biases_v["q3"] = 0.007
    calibration.load_data(tmp_path)
    actual = calibration.results["ds_raw"]
    assert actual.I.dims == expected.I.dims
    assert actual.drive_frequency_hz.dims == ("qubit",)
    assert float(actual.configured_bias_v.sel(qubit="q3")) == pytest.approx(0.003)
    np.testing.assert_array_equal(actual.I, expected.I)
    calibration.analyse_data()
    assert calibration.results["ds_raw"].full_freq.dims == ("qubit", "detuning")


@pytest.mark.parametrize("quadrature", ["I", "Q"])
def test_parabolic_map_rejects_false_high_peak_and_fits_absolute_bias(calibration, quadrature):
    calibration.parameters.num_flux_points = 31
    calibration.parameters.frequency_span_in_mhz = 20
    calibration.parameters.frequency_step_in_mhz = 0.1
    calibration.create_qua_program()
    axes = calibration.namespace["sweep_axes"]
    voltage = axes["flux_bias"].values
    detuning = axes["detuning"].values
    sweet_spot = 0.00315
    peak = 4e6 - 6e12 * (voltage - sweet_spot)**2
    peak[4] = 8e6  # Strong false peak, above the true maximum.
    signal = 0.01 + 0.1 * np.exp(-((detuning[:, None] - peak) / 2e5) ** 2)
    calibration.results["ds_raw"] = calibration._dataset(
        {"I": signal.T if quadrature == "I" else np.zeros_like(signal.T),
         "Q": -signal.T if quadrature == "Q" else np.zeros_like(signal.T)}, len(voltage),
    )
    calibration.analyse_data()
    result = calibration.results["fit_results"]["q3"]
    assert result["success"], result
    assert result["selected_quadrature"] == quadrature
    assert result["r_squared"] > 0.99
    assert result["num_outliers"] >= 1
    assert not calibration.results["ds_fit"].fit_inlier.sel(qubit="q3").values[4]
    assert result["idle_offset"] == pytest.approx(sweet_spot, abs=5e-5)
    updates = calibration.profile_updates()
    assert updates["qubits.json.qubits.q3.dc_bias_v"] == result["idle_offset"]
    assert result["qubit_frequency"] == pytest.approx(
        calibration.namespace["drive_frequency_hz"] + 4e6, abs=3e5,
    )
    with patch.object(module.plt, "show"):
        calibration.plot_data()
    module.plt.close("all")


@pytest.mark.parametrize("shape", ["flat", "outside"])
def test_unreliable_parabola_does_not_propose_profile_updates(calibration, shape):
    calibration.parameters.num_flux_points = 11
    calibration.parameters.frequency_span_in_mhz = 30
    calibration.parameters.frequency_step_in_mhz = 0.1
    calibration.create_qua_program()
    axes = calibration.namespace["sweep_axes"]
    voltage, detuning = axes["flux_bias"].values, axes["detuning"].values
    peak = np.full(len(voltage), 2e6) if shape == "flat" else -1e12 * (voltage - 0.005)**2
    signal = 0.01 + 0.1 * np.exp(-((detuning[:, None] - peak) / 2e5)**2)
    calibration.results["ds_raw"] = calibration._dataset(
        {"I": signal.T, "Q": np.zeros_like(signal.T)}, len(voltage),
    )
    calibration.analyse_data()
    assert not calibration.results["fit_results"]["q3"]["success"]
    assert calibration.profile_updates() == {}


@pytest.mark.parametrize("scale", [-1.6, 1.6])
def test_mw_drive_allows_unit_amplitude(calibration, scale):
    q = calibration.namespace["qubits"][0]
    q.xy.operations[calibration.parameters.operation].amplitude = 0.625
    calibration.parameters.operation_amplitude_factor = scale
    calibration.create_qua_program()


@pytest.mark.parametrize("scale", [-1.7, 1.7])
def test_mw_drive_rejects_amplitude_above_one(calibration, scale):
    q = calibration.namespace["qubits"][0]
    q.xy.operations[calibration.parameters.operation].amplitude = 0.625
    calibration.parameters.operation_amplitude_factor = scale
    with pytest.raises(ValueError, match="Scaled pulse amplitude must not exceed 1"):
        calibration.create_qua_program()
