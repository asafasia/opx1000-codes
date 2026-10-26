"""Completed-buffer fetching and cancellation regressions, with no hardware."""
from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from calibrations.core import CalibrationError, CalibrationOptions
from utils import result_fetching
from utils.qm_session import qm_session
from calibrations.registry import CALIBRATIONS


class Handle:
    def __init__(self, job, name):
        self.job, self.name = job, name

    def count_so_far(self):
        if self.name == "n":
            return int(self.job.count is not None)
        return int(self.job.status in {"Done", "Canceled"} and self.name not in self.job.missing
                   and self.job.values[self.name].size > 0)

    def fetch_all(self, *, flat_struct=False):
        if self.name == "n":
            return self.job.count
        assert self.job.status in {"Done", "Canceled"}
        return self.job.values[self.name]

    def has_dataloss(self):
        return self.name in self.job.lost


class Handles:
    def __init__(self, job):
        self.job = job
        self.handles = {name: Handle(job, name) for name in ("n", *job.values)}
        self.waited = False

    def keys(self):
        return self.handles.keys()

    def get(self, name):
        return self.handles.get(name)

    def wait_for_all_values(self, timeout=None):
        assert self.job.status in {"Done", "Canceled", "Error"}
        self.waited = True
        return self.job.complete


class Job:
    id = "test-job"

    def __init__(self, *, single_shot=True, states=("g", "e", "f"), analog=False):
        self.timeline = iter([
            ("In queue", None), ("Running", 0), ("Running", 1),
            ("Processing", 2), ("Done", 2),
        ])
        self.status, self.count = "In queue", None
        self.missing, self.lost = set(), set()
        self.complete = True
        self.report_reads = 0
        self.cancel_calls = 0
        self.labels = np.array([[0, 1], [2, 0], [1, 2]]) % len(states)
        if analog:
            self.values = {"I1": self.labels.astype(float), "Q1": -self.labels.astype(float)}
            if not single_shot:
                self.values = {key: val.mean(axis=0) for key, val in self.values.items()}
        elif single_shot:
            self.values = {"state1": self.labels}
        else:
            self.values = {f"population_{label}1": (self.labels == i).mean(axis=0)
                           for i, label in enumerate(states)}
            self.values["state1"] = self.values["population_e1"]
        self.result_handles = Handles(self)

    def get_status(self):
        self.status, self.count = next(self.timeline, (self.status, self.count))
        return self.status

    def cancel(self):
        self.cancel_calls += 1
        self.status = "Canceled"
        self.timeline = iter([])
        self.complete = False

    def execution_report(self):
        self.report_reads += 1
        return "No execution errors"


class LiveHandle(Handle):
    """QOP serves completed live buffers but loses headers after cancellation."""
    def count_so_far(self):
        if self.name == "n":
            return super().count_so_far()
        if self.job.status == "Canceled":
            raise result_fetching.QMConnectionError("Unexpected error in RPC handling")
        return int(self.job.values[self.name].size > 0)

    def fetch_all(self, *, flat_struct=False):
        if self.name == "n":
            return super().fetch_all(flat_struct=flat_struct)
        assert self.job.status == "Running", "Buffers must be captured before cancellation"
        return self.job.values[self.name].copy()


@pytest.mark.parametrize("single_shot", [False, True])
@pytest.mark.parametrize("unavailable", ["headers", "wait", "status"])
def test_ctrl_c_captures_buffers_before_qop_loses_canceled_results(monkeypatch, single_shot, unavailable):
    job = Job(single_shot=single_shot, analog=True)
    job.timeline = iter([("Running", 1)])
    if single_shot:
        job.values = {name: value[:2] for name, value in job.values.items()}
    job.result_handles.handles = {name: LiveHandle(job, name) for name in ("n", *job.values)}
    original_status = job.get_status
    original_wait = job.result_handles.wait_for_all_values

    def status():
        if unavailable == "status" and job.status == "Canceled":
            raise result_fetching.QMConnectionError("Unexpected error in RPC handling")
        return original_status()

    def wait(**kwargs):
        if unavailable == "wait":
            raise result_fetching.QMConnectionError("Unexpected error in RPC handling")
        return original_wait(**kwargs)

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(job, "get_status", status)
    monkeypatch.setattr(job.result_handles, "wait_for_all_values", wait)
    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    ds = result_fetching.fetch_result_dataset(job, axes(single_shot))
    assert job.cancel_calls == 1
    assert ds.attrs["acquisition_interrupted"]
    np.testing.assert_array_equal(ds.I[0], job.values["I1"])
    np.testing.assert_array_equal(ds.Q[0], job.values["Q1"])
    if single_shot:
        assert ds.attrs["completed_iterations"] == 2


def test_ctrl_c_without_snapshot_reports_unavailable_qop_results(monkeypatch):
    job = Job(analog=True)
    job.timeline = iter([("Running", 0)])
    # Regular handles have no completed live buffers; canceled headers fail.
    original_count = job.result_handles.get("I1").count_so_far

    def count():
        if job.status == "Canceled":
            raise result_fetching.QMConnectionError("Unexpected error in RPC handling")
        return original_count()

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(job.result_handles.get("I1"), "count_so_far", count)
    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    with pytest.raises(result_fetching.AcquisitionError, match="no complete buffers were recovered"):
        result_fetching.fetch_result_dataset(job, axes())
    assert job.cancel_calls == 1


def test_second_ctrl_c_during_snapshot_still_stops_job(monkeypatch):
    job = Job(analog=True)
    job.timeline = iter([("Running", 0)])
    job.result_handles.handles = {name: LiveHandle(job, name) for name in ("n", *job.values)}

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    monkeypatch.setattr(job.result_handles.get("I1"), "fetch_all", interrupt)
    with pytest.raises(KeyboardInterrupt):
        result_fetching.fetch_result_dataset(job, axes())
    assert job.cancel_calls == 1


def axes(single_shot=True):
    return {"qubit": xr.DataArray(["q1"]),
            **({"shot": xr.DataArray(np.arange(3))} if single_shot else {}),
            "idle_time": xr.DataArray([16, 100])}


@pytest.mark.parametrize("key", sorted(set(CALIBRATIONS) - {"hello", "qubit-external-flux"}))
@pytest.mark.parametrize("acquisition", ["averaged", "single_shot"])
def test_registered_experiments_recover_actual_stream_layouts(monkeypatch, tmp_path, key, acquisition):
    import re
    from qm import generate_qua_script
    from quam_config import create_machine

    entry = CALIBRATIONS[key]
    cloud_keys = {"resonator", "iq-blobs", "iq-blobs-gef", "readout-power", "readout-frequency-amplitude"}
    if key in cloud_keys and acquisition == "averaged":
        with pytest.raises(ValueError, match="single_shot"):
            entry.load_parameters_class()(acquisition=acquisition)
        return
    if key == "tof":
        monkeypatch.setattr(import_module(entry.module), "Path", lambda *_: tmp_path / "calibrations" / "tof.py")
    params = entry.load_parameters_class()(
        acquisition=acquisition, num_shots=3, reset_type="thermal", simulate=True,
    )
    if key.startswith("rb"):
        params.num_random_sequences = 3
        params.max_circuit_depth = 5
    node = entry.load_class()(parameters=params, machine=create_machine(qubit="q6"))
    try:
        script = generate_qua_script(node.create_qua_program())
        layout = node.namespace["acquisition_layout"]
        result_axes = node.namespace["sweep_axes"]
        dims = [name for name in result_axes if name != "qubit"]
        shape = tuple(len(result_axes[name]) for name in dims)
        partial = layout.single_shot or layout.loops[0][0] != layout.shot_axis
        if partial:
            shape = (min(2, shape[0]), *shape[1:])
        names = set(re.findall(r'\.(?:save|save_all)\("([^"]+)"\)', script)) - {"n"}
        names -= set(result_fetching.XarrayDataFetcher.ignore_handles)
        assert names, "No measurement streams found in the built program"
        job = Job()
        job.timeline = iter([("Running", 0)])
        job.values = {}
        for name in names:
            value = np.zeros(shape)
            singleton_axes = tuple(i for i, size in enumerate(shape) if size == 1 and (i or not partial))
            job.values[name] = np.squeeze(value, axis=singleton_axes)
        job.result_handles = Handles(job)

        def interrupt(_):
            raise KeyboardInterrupt()

        monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
        ds = node.fetch_result_dataset(job)
        assert job.cancel_calls == 1
        assert ds.attrs["acquisition_interrupted"]
        assert node.namespace["acquisition_interrupted"]
        for name in dims:
            assert ds.sizes[name] == shape[dims.index(name)]
        if partial:
            assert ds.attrs["completed_iterations"] == shape[0]
    finally:
        node.cleanup()


@pytest.mark.parametrize("single_shot", [True, False])
@pytest.mark.parametrize("readout", ["IQ", "GE", "GEF"])
def test_progress_is_live_before_full_buffers_exist(monkeypatch, single_shot, readout):
    job = Job(single_shot=single_shot, states=("g", "e", "f") if readout == "GEF" else ("g", "e"),
              analog=readout == "IQ")
    monkeypatch.setattr(result_fetching.time, "sleep", lambda _: None)
    progress = []
    ds = result_fetching.fetch_result_dataset(
        job, axes(single_shot), on_progress=lambda count, started: progress.append((count, job.status)))
    assert progress == [(0, "In queue"), (1, "Running"), (2, "Processing")]
    assert job.result_handles.waited
    expected_dims = ("qubit", "shot", "idle_time") if single_shot else ("qubit", "idle_time")
    if readout == "IQ":
        assert ds.I.dims == ds.Q.dims == expected_dims
        np.testing.assert_array_equal(ds.I[0], job.values["I1"])
    else:
        assert ds.state.dims == expected_dims
        np.testing.assert_array_equal(ds.state[0], job.values["state1"])
        if not single_shot:
            np.testing.assert_allclose(ds.population_g + ds.population_e + ds.get("population_f", 0), 1)


def test_progress_appears_before_accessing_sdk_handles(monkeypatch):
    job = Job()
    progress = []
    monkeypatch.setattr(result_fetching.time, "sleep", lambda _: None)

    class CheckedJob:
        @property
        def result_handles(self):
            assert progress, "SDK access started before displaying initial progress"
            assert progress[0] == 0
            return job.result_handles

        def __getattr__(self, name):
            return getattr(job, name)

    result_fetching.fetch_result_dataset(
        CheckedJob(), axes(), on_progress=lambda count, started: progress.append(count))
    assert progress == [0, 1, 2]


def test_t1_progress_reaches_completion_after_fetching_results(monkeypatch):
    module = import_module("calibrations.05_T1")
    job = Job()
    node = module.T1(parameters=module.Parameters(acquisition="single_shot", num_shots=3),
                     machine=object())
    node.namespace["sweep_axes"] = axes()
    progress = []
    node.report_progress = lambda count, *, start_time: progress.append(
        (count, job.status, job.result_handles.waited, start_time))
    monkeypatch.setattr(result_fetching.time, "sleep", lambda _: None)

    dataset = node.fetch_result_dataset(job)

    assert dataset.state.shape == (1, 3, 2)
    assert [entry[:3] for entry in progress] == [
        (0, "In queue", False), (1, "Running", False),
        (2, "Processing", False), (3, "Done", True),
    ]
    assert len({entry[3] for entry in progress}) == 1


@pytest.mark.parametrize("status", ["Canceled", "Error"])
def test_failed_jobs_do_not_return_partial_data(monkeypatch, status):
    job = Job(single_shot=False)
    job.timeline = iter([(status, 0)])
    with pytest.raises(result_fetching.AcquisitionError, match=status):
        result_fetching.fetch_result_dataset(job, axes(False))
    assert not job.result_handles.waited


@pytest.mark.parametrize("failure, message", [
    ("missing", "without results for: state1"),
    ("lost", "data loss in: state1"),
    ("incomplete", "before completion"),
])
def test_missing_or_incomplete_results_fail_explicitly(failure, message):
    job = Job()
    job.timeline = iter([("Done", 2)])
    if failure == "incomplete":
        job.complete = False
    else:
        getattr(job, failure).add("state1")
    with pytest.raises(result_fetching.AcquisitionError, match=message):
        result_fetching.fetch_result_dataset(job, axes())


def test_counter_only_job_is_not_a_measurement_dataset():
    job = Job()
    job.values = {}
    job.result_handles = Handles(job)
    job.timeline = iter([("Done", 2)])
    with pytest.raises(result_fetching.AcquisitionError, match="no measurement"):
        result_fetching.fetch_result_dataset(job, axes())


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, RuntimeError])
def test_session_closes_and_preserves_cancellation(monkeypatch, error_type):
    module = import_module("utils.qm_session")
    events = []

    @contextmanager
    def swallowing_session(*args, **kwargs):
        try:
            yield "qm"
        except KeyboardInterrupt:
            pass  # Matches the installed qualang-tools context manager.
        finally:
            events.append("closed")

    monkeypatch.setattr(module, "_qm_session", swallowing_session)
    with pytest.raises(error_type):
        with qm_session(None, {}, timeout=100):
            raise error_type("stop")
    assert events == ["closed"]


@pytest.mark.parametrize("has_data", [False, True])
@pytest.mark.parametrize("fit_success", [False, True])
def test_t1_interrupt_recovers_data_and_restores_bias(monkeypatch, has_data, fit_success):
    module = import_module("calibrations.05_T1")
    session_module = import_module("utils.qm_session")
    events = []
    job = Job(states=("g", "e"))
    job.values["state1"] = job.values["state1"][:2 if has_data else 0]

    @contextmanager
    def swallowing_session(*args, **kwargs):
        try:
            yield SimpleNamespace(execute=lambda _: job)
        except KeyboardInterrupt:
            pass
        finally:
            events.append("qm_closed")

    @contextmanager
    def apply_bias(_):
        events.append("bias_applied")
        try:
            yield
        finally:
            events.append("bias_zeroed")

    machine = SimpleNamespace(connect=lambda: None, generate_config=lambda: {},
        dc_bias=SimpleNamespace(qubit_biases_v={"q1": -0.2}, output_channel=0,
            voltage_for_qubit=lambda _: -0.2, applied_for_qubit=apply_bias))
    node = module.T1(parameters=module.Parameters(acquisition="single_shot", num_shots=3),
        machine=machine, options=CalibrationOptions(save_raw_data=False, save_analysis_result=False,
            save_figures=False, plot_data=True, update_state=True, propose_profile_update=True,
            apply_profile_update=False, report_runtime_estimate=False))
    node.namespace["sweep_axes"] = axes()
    node.create_qua_program = lambda: "program"
    node.namespace["qubits"] = [SimpleNamespace(name="q1")]
    def analyse():
        events.append("analysed")
        node.outcomes = {"q1": "successful" if fit_success else "failed"}

    node.analyse_data = analyse
    node.plot_data = lambda: events.append("plotted")
    node.update_state = lambda: events.append("state_updated")
    node._propose_profile_update_from_options = lambda: events.append("profile_proposed")
    monkeypatch.setattr(session_module, "_qm_session", swallowing_session)
    monkeypatch.setattr("calibrations.core.base.assert_outputs_allowed", lambda: None)

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    status = node.run()
    assert status.interrupted
    assert job.cancel_calls == 1
    assert events == ["bias_applied", "qm_closed", "bias_zeroed"] + (
        ["analysed", "plotted"] + (
            ["state_updated", "profile_proposed"] if fit_success else []
        ) if has_data else [])
    assert ("ds_raw" in node.results) == has_data
    if has_data:
        ds = node.results["ds_raw"]
        assert ds.sizes["shot"] == 2
        np.testing.assert_array_equal(ds.state_shots[0], job.values["state1"])
        np.testing.assert_allclose(ds.population_e[0], (job.values["state1"] == 1).mean(axis=0))
        assert ds.attrs["acquisition_interrupted"]
        assert ds.attrs["requested_iterations"] == 3
    assert job.report_reads == 1


def test_t1_missing_results_show_job_report_without_unbound_local_error(monkeypatch):
    module = import_module("calibrations.05_T1")
    job = Job()
    job.timeline = iter([("Done", 2)])
    job.missing.add("state1")
    node = module.T1(parameters=module.Parameters(acquisition="single_shot", num_shots=3),
                     machine=object())
    node.namespace["sweep_axes"] = axes()
    progress = []
    node.report_progress = lambda count, *, start_time: progress.append(count)
    with pytest.raises(CalibrationError, match="test-job finished without results for: state1"):
        node.fetch_result_dataset(job)
    assert progress == [0, 2]  # Missing measurement results must never show 100%.
    assert job.report_reads == 1
    assert node.namespace["execution_report"] == "No execution errors"


def test_rb_progress_counts_sequences_instead_of_nested_shots():
    module = import_module("calibrations.11a_single_qubit_randomized_benchmarking")
    node = module.SingleQubitRandomizedBenchmarking(
        parameters=module.Parameters(num_random_sequences=7, num_shots=30), machine=object())
    assert node.progress_total() == 7


@pytest.mark.parametrize("single_shot", [False, True])
@pytest.mark.parametrize("readout", ["IQ", "GE", "GEF"])
def test_ctrl_c_recovers_averaged_and_single_shot_measurements(monkeypatch, single_shot, readout):
    from calibration_utils.state_acquisition import prepare_acquisition_dataset

    states = ("g", "e", "f") if readout == "GEF" else ("g", "e")
    job = Job(single_shot=single_shot, states=states, analog=readout == "IQ")
    job.timeline = iter([("Running", 1)])
    if single_shot:
        job.values = {name: value[:2] for name, value in job.values.items()}

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    ds = result_fetching.fetch_result_dataset(job, axes(single_shot))
    assert job.cancel_calls == 1
    assert ds.attrs["acquisition_interrupted"]
    if single_shot:
        assert ds.sizes["shot"] == 2
        assert ds.attrs["completed_iterations"] == 2
    result = prepare_acquisition_dataset(ds, SimpleNamespace(readout_states=states))
    if readout == "IQ":
        expected = job.values["I1"].mean(axis=0) if single_shot else job.values["I1"]
        np.testing.assert_allclose(result.I[0], expected)
    else:
        expected = (job.values["state1"] == 1).mean(axis=0) if single_shot else job.values["state1"]
        np.testing.assert_allclose(result.state[0], expected)
        np.testing.assert_allclose(sum(result[f"population_{state}"] for state in states), 1)


@pytest.mark.parametrize("single_shot", [False, True])
def test_stopped_rb_keeps_complete_sequences_and_common_stream_prefix(monkeypatch, single_shot):
    job = Job(analog=True)
    job.timeline = iter([("Running", 1)])
    shape = (3, 2, 4) if single_shot else (3, 2)
    job.values = {"I1": np.arange(np.prod(shape)).reshape(shape),
                  "Q1": -np.arange(np.prod(shape)).reshape(shape)[:2]}
    job.result_handles = Handles(job)
    rb_axes = {"qubit": np.array(["q6"]), "nb_of_sequences": np.arange(5), "depths": np.array([1, 4])}
    if single_shot:
        rb_axes["shot"] = np.arange(4)

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    ds = result_fetching.fetch_result_dataset(job, rb_axes, partial_axis="nb_of_sequences")
    assert ds.sizes["nb_of_sequences"] == 2
    np.testing.assert_array_equal(ds.I[0], job.values["I1"][:2])
    np.testing.assert_array_equal(ds.Q[0], job.values["Q1"])
    assert ("shot" in ds.dims) == single_shot


def test_sdk_structured_save_all_preserves_length_one_axes():
    job = Job()
    job.timeline = iter([("Done", 0)])
    structured = np.zeros(1, dtype=[("value", np.int64, (1, 2))])
    structured["value"] = [[[1, 0]]]
    job.values = {"state1": structured}
    ds = result_fetching.fetch_result_dataset(job, {
        "qubit": np.array(["q1"]), "shot": np.arange(1),
        "sweep": np.arange(1), "time": np.array([16, 100]),
    })
    assert ds.state.shape == (1, 1, 1, 2)
    np.testing.assert_array_equal(ds.state.values, [[[[1, 0]]]])


@pytest.mark.parametrize("label", ["I", "Q", "state", "population_e"])
@pytest.mark.parametrize("shape", [(1, 200), (200, 1), (1, 200, 1)])
@pytest.mark.parametrize("interrupted", [False, True])
def test_results_restore_omitted_singleton_sweep_axes(monkeypatch, label, shape, interrupted):
    job = Job()
    job.timeline = iter([("Running" if interrupted else "Done", 0)])
    job.values = {f"{label}1": np.arange(200)}
    job.result_handles = Handles(job)
    result_axes = {"qubit": ["q1"], **{
        f"sweep_{i}": np.arange(size) for i, size in enumerate(shape)
    }}

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    ds = result_fetching.fetch_result_dataset(job, result_axes)
    assert ds[label].shape == (1, *shape)
    assert job.cancel_calls == int(interrupted)
    assert bool(ds.attrs.get("acquisition_interrupted")) == interrupted
    np.testing.assert_array_equal(ds[label].values.reshape(-1), np.arange(200))


@pytest.mark.parametrize("shots", [1, 2])
def test_ctrl_c_restores_singleton_sweeps_without_padding_partial_shots(monkeypatch, shots):
    job = Job(analog=True)
    job.timeline = iter([("Running", 0)])
    job.values = {"I1": np.arange(shots * 200).reshape(shots, 200),
                  "Q1": -np.arange(600).reshape(3, 200)}
    job.result_handles = Handles(job)

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    ds = result_fetching.fetch_result_dataset(job, {
        "qubit": ["q1"], "shot": np.arange(5),
        "nb_of_pulses": [1], "amp_prefactor": np.arange(200),
    })
    assert ds.I.shape == ds.Q.shape == (1, shots, 1, 200)
    assert ds.attrs["completed_iterations"] == shots
    np.testing.assert_array_equal(ds.I.values[0, :, 0, :], job.values["I1"])
    np.testing.assert_array_equal(ds.Q.values[0, :, 0, :], job.values["Q1"][:shots])


@pytest.mark.parametrize("interrupted", [False, True])
def test_results_reject_flattened_non_singleton_axes(monkeypatch, interrupted):
    job = Job()
    job.timeline = iter([("Running" if interrupted else "Done", 0)])
    job.values = {"I1": np.arange(200)}
    job.result_handles = Handles(job)

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    with pytest.raises(result_fetching.AcquisitionError, match="Unexpected result shape"):
        result_fetching.fetch_result_dataset(job, {
            "qubit": ["q1"], "pulses": [1, 3], "amplitude": np.arange(100),
        })


def test_second_ctrl_c_aborts_recovery(monkeypatch):
    job = Job()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    monkeypatch.setattr(job, "cancel", interrupt)
    with pytest.raises(KeyboardInterrupt):
        result_fetching.fetch_result_dataset(job, axes())


def test_legacy_halt_recovers_results(monkeypatch):
    job = Job()
    job.halt, job.cancel = job.cancel, None

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    ds = result_fetching.fetch_result_dataset(job, axes())
    assert ds.attrs["acquisition_interrupted"]
    assert job.cancel_calls == 1


def test_stopped_data_loss_remains_an_error(monkeypatch):
    job = Job()
    job.lost.add("state1")

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    with pytest.raises(result_fetching.AcquisitionError, match="data loss"):
        result_fetching.fetch_result_dataset(job, axes())


def test_recovered_t1_data_can_be_fitted_and_plotted(monkeypatch):
    from quam_config import create_machine
    import matplotlib.pyplot as plt

    module = import_module("calibrations.05_T1")
    node = module.T1(parameters=module.Parameters(acquisition="single_shot", num_shots=300, use_state_discrimination=True),
                     machine=create_machine(qubit="q1b"))
    node.get_qubits()
    times = np.linspace(16, 60000, 41)
    node.namespace["sweep_axes"] = {"qubit": np.array(["q1b"]), "shot": np.arange(300),
                                      "idle_time": xr.DataArray(times, attrs={"units": "ns"})}
    job = Job(states=("g", "e"))
    probability = 0.8 * np.exp(-times / 20000) + 0.05
    job.values["state1"] = (np.random.default_rng(4).random((200, 41)) < probability).astype(int)

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(result_fetching.time, "sleep", interrupt)
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        node.results["ds_raw"] = node.annotate_readout_dataset(node.fetch_result_dataset(job))
        node.analyse_data()
        node.plot_data()
        assert node.results["ds_raw"].sizes["shot"] == 200
        assert node.results["ds_fit"].success.item()
        assert "raw_fit" in node.results["figures"]
    finally:
        plt.close("all")
        node.cleanup()
