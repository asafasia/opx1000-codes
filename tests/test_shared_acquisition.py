"""Offline acquisition coverage: no connections to the QOP or lab instruments."""
from types import SimpleNamespace
from pathlib import Path
from importlib import import_module

import numpy as np
import pytest
import xarray as xr
from qm import generate_qua_script
from qm.qua import declare_stream, program, stream_processing

from calibration_utils.state_acquisition import AcquisitionLayout, prepare_acquisition_dataset
from calibrations.registry import CALIBRATIONS
from quam_config import create_machine


CLOUD_KEYS = {"resonator", "iq-blobs", "iq-blobs-gef", "readout-power", "readout-frequency-amplitude"}
KEYS = sorted(set(CALIBRATIONS) - {"hello"})


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("acquisition", ["averaged", "single_shot"])
def test_all_registered_measurement_programs_build(key, acquisition, tmp_path, monkeypatch):
    entry = CALIBRATIONS[key]
    cls = entry.load_parameters_class()
    if key in CLOUD_KEYS and acquisition == "averaged":
        with pytest.raises(ValueError, match="single_shot"):
            cls(acquisition=acquisition)
        return
    if key == "tof":
        module = import_module(entry.module)
        monkeypatch.setattr(module, "Path", lambda *_: tmp_path / "calibrations" / "tof.py")
    params = cls(acquisition=acquisition, num_shots=3, reset_type="thermal", simulate=True)
    if key.startswith("rb"):
        params.num_random_sequences = 2
        params.max_circuit_depth = 5
    machine = create_machine(qubit="q6")
    if key == "qubit-external-flux":
        machine.dc_bias.qubit_biases_v["q6"] = 0.0
    node = entry.load_class()(parameters=params, machine=machine)
    try:
        qua = node.create_qua_program()
        script = generate_qua_script(qua)
        assert "stream_processing" in script
        layout = node.namespace["acquisition_layout"]
        axes = node.namespace["sweep_axes"]
        assert layout.single_shot == (acquisition == "single_shot")
        assert (layout.shot_axis in axes) == layout.single_shot
        expected = [name for name, _ in layout.loops
                    if name != layout.shot_axis or layout.single_shot]
        assert [k for k in axes if k not in ("qubit", "readout_time")] == expected
        if key.startswith("rb"):
            assert layout.loops[-1] == ("shot", 3)
            assert "nb_of_sequences" in axes
        elif key not in {"qubit-external-flux"}:
            assert layout.loops[0][0] in ("shot", "n_runs")
    finally:
        node.cleanup()
        for q in node.namespace.get("tracked_qubits", []):
            q.revert_changes()
        for r in node.namespace.get("tracked_resonators", []):
            r.revert_changes()


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("single_shot", [False, True])
def test_real_qua_reduces_only_the_shot_axis(position, single_shot):
    loops = [("sequence", 2), ("sweep", 5)]
    loops.insert(position, ("shot", 3))
    layout = AcquisitionLayout(tuple(loops), single_shot=single_shot)
    with program() as qua:
        stream = declare_stream()
        with stream_processing():
            layout.save(stream, "result")
    script = generate_qua_script(qua)
    if single_shot:
        expected = "".join(f".buffer({n})" for _, n in reversed(loops[1:]))
    elif position == 0:
        expected = '.buffer(5).buffer(2).average()'
    elif position == 1:
        expected = '.buffer(5).buffer(3).map(FUNCTIONS.average(0))'
    else:
        expected = '.buffer(3).map(FUNCTIONS.average(0)).buffer(5)'
    terminal = 'save_all' if single_shot or position != 0 else 'save'
    assert expected + f'.{terminal}("result")' in script


@pytest.mark.parametrize("position", [0, 1, 2])
def test_iq_reduction_preserves_each_sequence_and_sweep(position):
    dims = ["sequence", "sweep"]
    sizes = [2, 5]
    dims.insert(position, "shot")
    sizes.insert(position, 3)
    values = np.arange(np.prod(sizes), dtype=float).reshape(sizes)
    ds = xr.Dataset({"I": (dims, values), "Q": (dims, -values)})
    result = prepare_acquisition_dataset(ds, SimpleNamespace())
    np.testing.assert_equal(result.I, values.mean(axis=position))
    np.testing.assert_equal(result.Q, -values.mean(axis=position))
    xr.testing.assert_equal(result.I_shots, ds.I.rename("I_shots"))
    xr.testing.assert_equal(result.Q_shots, ds.Q.rename("Q_shots"))
    assert result.I.dims == ("sequence", "sweep")
    assert prepare_acquisition_dataset(result, SimpleNamespace()) is result


def test_cloud_data_is_not_reduced():
    ds = xr.Dataset({"Ig": (("n_runs", "sweep"), np.arange(15).reshape(3, 5))})
    assert prepare_acquisition_dataset(ds, SimpleNamespace()) is ds


@pytest.mark.parametrize("loops", [(("sweep", 3),), (("shot", 0),), (("shot", 2), ("shot", 3))])
def test_invalid_acquisition_layouts_fail_before_build(loops):
    with pytest.raises(ValueError):
        AcquisitionLayout(loops)


def test_single_shot_round_trip_preserves_equal_length_axes_and_metadata(tmp_path):
    from calibration_io import CalibrationSaver
    from calibrations.core import BaseCalibration

    profile = tmp_path / "profiles" / "main"
    profile.mkdir(parents=True)
    (profile / "profile.json").write_text("{}")
    saver = CalibrationSaver(tmp_path / "runs", tmp_path / "profiles")
    # Equal dimension lengths and auxiliary coordinates defeat shape guessing.
    ds = xr.Dataset(
        {"state": (("qubit", "depths", "shot"), np.arange(9).reshape(1, 3, 3) % 3)},
        coords={"qubit": ["q6"], "shot": np.arange(3), "depths": [1, 4, 8],
                "frequency": ("qubit", [5e9])},
        attrs={"acquisition": "single_shot", "readout_states": ["g", "e", "f"]},
    )
    ds.state.attrs["long_name"] = "discriminated state label"
    directory = saver.save_xarray("test", ds, profile_name="main")
    loaded = BaseCalibration.load_saved_run(SimpleNamespace(), directory)
    xr.testing.assert_identical(loaded, ds)
    processed = prepare_acquisition_dataset(loaded, SimpleNamespace())
    np.testing.assert_allclose(processed.population_f, 1 / 3)


@pytest.mark.parametrize("key", ["power-rabi", "t1", "ramsey", "echo", "drag", "rb", "rb-interleaved"])
@pytest.mark.parametrize("readout", ["IQ", "GE", "GEF"])
@pytest.mark.parametrize("acquisition", ["averaged", "single_shot"])
def test_readout_modes_share_streams_without_averaging_integer_labels(key, readout, acquisition):
    entry = CALIBRATIONS[key]
    params = entry.load_parameters_class()(
        num_shots=3, acquisition=acquisition, simulate=True, reset_type="thermal",
        use_state_discrimination=readout != "IQ",
        readout_states=["g", "e", "f"] if readout == "GEF" else ["g", "e"],
    )
    if key.startswith("rb"):
        params.num_random_sequences = 2
        params.max_circuit_depth = 5
    node = entry.load_class()(parameters=params, machine=create_machine(qubit="q6"))
    try:
        script = generate_qua_script(node.create_qua_program())
        terminal = 'save_all' if acquisition == 'single_shot' or key.startswith('rb') else 'save'
        if readout == "IQ":
            assert f'{terminal}("I1")' in script and f'{terminal}("Q1")' in script
            assert '"state1"' not in script
        else:
            assert f'{terminal}("state1")' in script
            assert '"I1"' not in script
            assert (f'{terminal}("population_g1")' in script) == (acquisition == "averaged")
            assert (f'{terminal}("population_f1")' in script) == (readout == "GEF" and acquisition == "averaged")
        if acquisition == "single_shot":
            assert '.average(' not in script
    finally:
        node.cleanup()


@pytest.mark.parametrize("acquisition", ["averaged", "single_shot"])
def test_standalone_resonator_scan_uses_shared_layout(acquisition):
    from calibrations.resonator_parameter_scan import Parameters, ResonatorParameterScan
    node = ResonatorParameterScan(parameters=Parameters(acquisition=acquisition, num_shots=3),
                                  machine=create_machine(qubit="q9"))
    script = generate_qua_script(node.create_qua_program())
    assert ('shot' in node.namespace['sweep_axes']) == (acquisition == 'single_shot')
    terminal = 'save_all' if acquisition == 'single_shot' else 'save'
    assert f'{terminal}("Ig1")' in script
