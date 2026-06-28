"""Qiskit circuit runner backed by this repository's calibrated QuAM."""

from __future__ import annotations

import inspect
from datetime import datetime
from time import perf_counter
from collections.abc import Mapping
from typing import Any


_GATE_LEVEL_STATUS_UPDATES = True


def _status(message: str) -> None:
    if _GATE_LEVEL_STATUS_UPDATES:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[gate-level {timestamp}] {message}", flush=True)


def patch_qm_job_submit_for_current_sdk() -> None:
    """Keep qiskit-qm-provider compatible with this installed QM SDK."""
    from qm.api.v2.job_api.job_api import JobApiWithDeprecations
    from qm.jobs.running_qm_job import RunningQmJob
    from qm.simulate.interface import SimulationConfig
    from qiskit.primitives.containers import BitArray
    import qiskit_qm_provider.job.qm_job as qm_job_module

    QMJob = qm_job_module.QMJob

    qm_job_module.RunningQmJob = (RunningQmJob, JobApiWithDeprecations)
    patch_bitarray_from_samples_for_current_sdk(BitArray)

    if getattr(QMJob.submit, "_opx1000_compat", False):
        return

    def _accepted_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return kwargs
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return kwargs
        return {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }

    def submit(self):
        compiler_options = self.metadata.get("compiler_options", None)
        simulate = self.metadata.get("simulate", None)
        kwargs = {
            "simulate": simulate,
            "compiler_options": compiler_options,
            "terminal_output": True,
        }
        if "timeout" in self.metadata:
            kwargs["options"] = {"timeout": self.metadata["timeout"]}

        if isinstance(simulate, SimulationConfig):
            _status("QOP simulation compile/submit started.")
            started = perf_counter()
            simulate_kwargs = _accepted_kwargs(
                self.qm.simulate,
                {"simulate": simulate, "compiler_options": compiler_options},
            )
            self._qm_job = self.qm.simulate(self.program, **simulate_kwargs)
            _status(
                "QOP simulation compile/submit finished "
                f"after {perf_counter() - started:.1f} s."
            )
            return

        if isinstance(self.program, list):
            _status(f"QOP queue submission started for {len(self.program)} circuits.")
            started = perf_counter()
            self._job_id = ""
            self._qm_job = []
            for prog in self.program:
                queue_kwargs = _accepted_kwargs(self.qm.queue.add, kwargs)
                self._qm_job.append(self.qm.queue.add(prog, **queue_kwargs))
            self._job_id += ",".join([job.id for job in self._qm_job])
            _status(
                "QOP queue submission finished "
                f"after {perf_counter() - started:.1f} s; "
                f"job id(s): {self._job_id or 'unavailable'}."
            )
            return

        _status("QOP compile/execute started. Waiting for hardware job to be accepted.")
        started = perf_counter()
        execute_kwargs = _accepted_kwargs(self.qm.execute, kwargs)
        self._qm_job = self.qm.execute(self.program, **execute_kwargs)
        self._job_id = self._qm_job.id if hasattr(self._qm_job, "id") else ""
        _status(
            "QOP compile/execute finished "
            f"after {perf_counter() - started:.1f} s; "
            "circuit is now running or already completed on the OPX "
            f"(job id: {self._job_id or 'unavailable'})."
        )

    submit._opx1000_compat = True
    QMJob.submit = submit


def patch_bitarray_from_samples_for_current_sdk(BitArray) -> None:
    """Normalize current-QM-SDK stream samples before Qiskit builds counts."""
    if getattr(BitArray.from_samples, "_opx1000_compat", False):
        return

    original_from_samples = BitArray.from_samples

    def _sample_value(sample):
        if hasattr(sample, "dtype") and getattr(sample.dtype, "names", None):
            field = "value" if "value" in sample.dtype.names else sample.dtype.names[0]
            sample = sample[field]
        elif isinstance(sample, tuple):
            sample = sample[0]
        if hasattr(sample, "item"):
            sample = sample.item()
        return int(sample)

    def from_samples(samples, num_bits):
        return original_from_samples(
            [_sample_value(sample) for sample in samples], num_bits
        )

    from_samples._opx1000_compat = True
    BitArray.from_samples = staticmethod(from_samples)


def add_local_basic_macros(backend, reset_type: str) -> None:
    """Register minimal single-qubit macros for this repository's QuAM."""
    from qm.qua import assign, declare, fixed, wait
    from quam.components.macro import PulseMacro, QubitMacro
    from quam.core import quam_dataclass

    @quam_dataclass
    class LocalMeasureMacro(QubitMacro):
        pulse: str = "readout"

        def apply(self):
            state = declare(bool)
            i_value = declare(fixed)
            q_value = declare(fixed)
            self.qubit.resonator.measure(self.pulse, qua_vars=(i_value, q_value))
            threshold = self.qubit.resonator.operations[self.pulse].threshold
            assign(state, i_value > threshold)
            return state

    @quam_dataclass
    class LocalResetMacro(QubitMacro):
        reset_type: str = "active"

        def apply(self):
            quam_reset_type = (
                "thermal" if self.reset_type == "thermalize" else self.reset_type
            )
            self.qubit.reset(quam_reset_type)

    @quam_dataclass
    class LocalRzMacro(QubitMacro):
        def apply(self, phi):
            self.qubit.xy.frame_rotation(phi)

    @quam_dataclass
    class LocalDelayMacro(QubitMacro):
        def apply(self, duration):
            wait(duration, self.qubit.xy.name)

    @quam_dataclass
    class LocalIdMacro(QubitMacro):
        def apply(self):
            pass

    for qubit in backend.machine.active_qubits:
        qubit.macros["x"] = PulseMacro(pulse=qubit.get_pulse("x180").get_reference())
        qubit.macros["sx"] = PulseMacro(pulse=qubit.get_pulse("x90").get_reference())
        qubit.macros["sxdg"] = PulseMacro(pulse=qubit.get_pulse("-x90").get_reference())
        qubit.macros["sy"] = PulseMacro(pulse=qubit.get_pulse("y90").get_reference())
        qubit.macros["sydg"] = PulseMacro(pulse=qubit.get_pulse("-y90").get_reference())
        if "y180" in qubit.xy.operations:
            qubit.macros["y"] = PulseMacro(
                pulse=qubit.get_pulse("y180").get_reference()
            )
        qubit.macros["rz"] = LocalRzMacro()
        qubit.macros["measure"] = LocalMeasureMacro()
        qubit.macros["reset"] = LocalResetMacro(reset_type=reset_type)
        qubit.macros["delay"] = LocalDelayMacro()
        qubit.macros["id"] = LocalIdMacro()


def build_backend(profile: str, qubit: str | None, reset_type: str):
    from qiskit_qm_provider import QMBackend, QMProvider, add_basic_macros

    from quam_config import create_machine

    class LocalFixedFrequencyBackend(QMBackend):
        @property
        def qubit_mapping(self):
            return {
                index: (q.xy.name, q.resonator.name)
                for index, q in enumerate(self.machine.active_qubits)
            }

        @property
        def meas_map(self):
            return [[index] for index, _ in enumerate(self.machine.active_qubits)]

    machine = create_machine(profile_name=profile, qubit=qubit)
    provider = QMProvider()
    backend = provider.get_backend(
        machine=machine, backend_cls=LocalFixedFrequencyBackend
    )

    try:
        add_basic_macros(backend, reset_type=reset_type)
    except ModuleNotFoundError as exc:
        if exc.name != "iqcc_calibration_tools":
            raise
        print(
            "qiskit-qm-provider add_basic_macros requires iqcc_calibration_tools; "
            "using local single-qubit macros instead."
        )
        add_local_basic_macros(backend, reset_type=reset_type)

    backend.update_target()
    return backend


class Runner:
    """Small Qiskit circuit runner backed by this repository's calibrated QuAM."""

    def __init__(
        self,
        qubit: str = "q1",
        *,
        profile: str = "single_qubit",
        reset_type: str = "active",
        status_updates: bool = True,
    ) -> None:
        self.qubit = qubit
        self.profile = profile
        self.reset_type = reset_type
        self.status_updates = status_updates
        self.backend = build_backend(profile=profile, qubit=qubit, reset_type=reset_type)

    def transpile(self, circuits, **transpile_options):
        from qiskit import transpile

        global _GATE_LEVEL_STATUS_UPDATES
        previous_status_updates = _GATE_LEVEL_STATUS_UPDATES
        _GATE_LEVEL_STATUS_UPDATES = self.status_updates
        try:
            _status("Qiskit transpilation started.")
            started = perf_counter()
            transpiled = transpile(circuits, self.backend, **transpile_options)
            _status(
                "Qiskit transpilation finished "
                f"after {perf_counter() - started:.1f} s."
            )
            return transpiled
        finally:
            _GATE_LEVEL_STATUS_UPDATES = previous_status_updates

    def submit(
        self,
        circuits,
        *,
        shots: int = 1000,
        do_transpile: bool = True,
        transpile_options: dict[str, Any] | None = None,
        **run_options,
    ):
        global _GATE_LEVEL_STATUS_UPDATES
        previous_status_updates = _GATE_LEVEL_STATUS_UPDATES
        _GATE_LEVEL_STATUS_UPDATES = self.status_updates
        patch_qm_job_submit_for_current_sdk()
        try:
            self.close_all_qms()
            run_input = (
                self.transpile(circuits, **(transpile_options or {}))
                if do_transpile
                else circuits
            )
            _status(f"Backend run requested with {shots} shots.")
            job = self.backend.run(run_input, shots=shots, **run_options)
            _status("Backend returned a job object; waiting for result data.")
            started = perf_counter()
            result = job.result()
            _status(f"Result data received after {perf_counter() - started:.1f} s.")
            return result
        finally:
            _GATE_LEVEL_STATUS_UPDATES = previous_status_updates

    def close_all_qms(self) -> None:
        try:
            self.backend.close_all_qms()
        except Exception as exc:
            print(f"Could not close existing quantum machines before run: {exc}")


def result_counts(result: Any) -> Mapping[str, int] | None:
    results = getattr(result, "results", None)
    if results is not None and not isinstance(results, list):
        data = getattr(results, "data", None)
        counts = getattr(data, "counts", None)
        if counts is not None:
            return counts
    if hasattr(result, "get_counts"):
        try:
            return result.get_counts()
        except TypeError:
            pass
    if hasattr(result, "data"):
        return result.data
    return None


def result_counts_list(result: Any) -> list[Mapping[str, int]]:
    """Return counts for each circuit in a single- or multi-circuit result."""
    results = getattr(result, "results", None)
    if isinstance(results, list):
        counts = []
        for experiment_result in results:
            data = getattr(experiment_result, "data", None)
            experiment_counts = getattr(data, "counts", None)
            if experiment_counts is None:
                raise ValueError("Experiment result does not contain counts.")
            counts.append(experiment_counts)
        return counts

    counts = result_counts(result)
    if counts is None:
        raise ValueError("Result does not contain counts.")
    return [counts]
