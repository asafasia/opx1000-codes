"""Base class for class-oriented calibration experiments.

The class-based calibration shape keeps the useful parts of the existing nodes
(`parameters`, `machine`, `namespace`, `results`, `outcomes`, and `log`) while
moving the lifecycle into ordinary methods that can be overridden and tested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import inspect
import json
from pathlib import Path
import time
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar

import numpy as np
import matplotlib.pyplot as plt

from calibration_io import CalibrationSaver
from calibrations.output_safety import assert_outputs_allowed
from profiles import ProfileUpdater, current_profile_name
from qualibrate import NodeParameters
from quam_config import Quam


P = TypeVar("P", bound=NodeParameters)
M = TypeVar("M", bound=Quam)


class CalibrationError(RuntimeError):
    """Raised when a calibration lifecycle step cannot complete."""


@dataclass(frozen=True)
class CalibrationStatus:
    """Compact report returned by :meth:`BaseCalibration.run`."""

    name: str
    mode: str
    simulated: bool
    loaded: bool
    raw_data_saved: bool
    figures_saved: bool
    ai_review_saved: bool
    profile_update_proposed: bool
    outcomes: Mapping[str, str] = field(default_factory=dict)
    interrupted: bool = False


@dataclass
class CalibrationOptions:
    """Runtime switches for the shared calibration lifecycle."""

    save_raw_data: bool = True
    save_analysis_result: bool = True
    save_figures: bool = True
    analyse_data: bool = True
    plot_data: bool = True
    update_state: bool = True
    propose_profile_update: bool = True
    apply_profile_update: bool = True
    ai_review: bool = False
    report_runtime_estimate: bool = True


class BaseCalibration(ABC, Generic[P, M]):
    """Abstract base class for new calibration experiments.

    Subclasses usually override:
    - :meth:`create_qua_program` for the QUA sequence and sweep axes.
    - :meth:`analyse_data` for processing and fit results.
    - :meth:`plot_data` for figures.
    - :meth:`profile_updates` or :meth:`update_state` for accepted outcomes.

    The object itself is intentionally node-like so existing helper functions
    that expect ``node.parameters`` or ``node.results`` can be reused.
    """

    def __init__(
        self,
        *,
        name: str,
        parameters: P,
        machine: M | None = None,
        description: str = "",
        profile_name: str | None = None,
        qubit: str | None = None,
        auto_connect: bool = False,
        saver: CalibrationSaver | None = None,
        profile_updater: ProfileUpdater | None = None,
        machine_factory: Callable[..., M] | None = None,
        logger: Callable[[str], None] | None = None,
        options: CalibrationOptions | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.profile_name = profile_name
        self.qubit = qubit
        self.machine = machine if machine is not None else self.create_machine(machine_factory)
        self.saver = saver or CalibrationSaver()
        self.profile_updater = profile_updater or ProfileUpdater()
        self.options = options or CalibrationOptions()
        self.namespace: dict[str, Any] = {}
        self.results: dict[str, Any] = {}
        self.outcomes: dict[str, str] = {}
        self._logger = logger or print

        if auto_connect:
            if not self.simulate_requested:
                assert_outputs_allowed()
            self.connect_machine(close_existing_qms=True)

    @property
    def simulate_requested(self) -> bool:
        return bool(getattr(self.parameters, "simulate", False))

    @property
    def load_data_id(self) -> Any:
        return getattr(self.parameters, "load_data_id", None)

    @property
    def timeout(self) -> int | None:
        return getattr(self.parameters, "timeout", None)

    def log(self, message: str) -> None:
        self._logger(f"[{self.name}] {message}")

    def create_machine(self, machine_factory: Callable[..., M] | None = None) -> M:
        """Build the default machine lazily from the repository profile."""
        if machine_factory is None:
            from quam_config import create_machine

            machine_factory = create_machine
        keyword_args = {}
        if self.profile_name is not None:
            keyword_args["profile_name"] = self.profile_name
        if self.qubit is not None:
            keyword_args["qubit"] = self.qubit
        return machine_factory(**keyword_args)

    def connect_machine(self, *, close_existing_qms: bool = False) -> Any:
        """Connect the machine and optionally close already-open QMs."""
        if not hasattr(self.machine, "connect"):
            raise CalibrationError("Machine object does not expose connect().")
        qmm = self.machine.connect()
        if close_existing_qms and hasattr(self.machine, "qmm"):
            self.machine.qmm.close_all_qms()
        return qmm

    def get_qubits(self) -> Any:
        """Return selected qubits using the existing qualibration helper."""
        from qualibration_libs.parameters import get_qubits

        self.namespace["qubits"] = qubits = get_qubits(self)
        from utils.experiment_readout import readout_operation
        for qubit in qubits:
            qubit.resonator.selected_readout_operation = readout_operation(self.parameters)
        return qubits

    def reset_qubit(self, qubit):
        from utils.experiment_readout import reset_qubit
        return reset_qubit(qubit, self.parameters)

    def readout_state(self, qubit, state):
        from utils.experiment_readout import readout_operation, readout_states
        from utils.readout_macro import readout_state_configured
        return readout_state_configured(qubit, state,
            num_states=len(readout_states(self.parameters)),
            pulse_name=readout_operation(self.parameters))

    def measure_readout(self, qubit, **kwargs):
        from utils.experiment_readout import readout_operation
        from utils.readout_macro import measure_readout
        return measure_readout(qubit, readout_operation(self.parameters), **kwargs)

    @property
    def single_shot_acquisition(self) -> bool:
        from calibration_utils.state_acquisition import single_shot_acquisition
        return single_shot_acquisition(self.parameters)

    def configure_acquisition(
        self, *, loop_order=None, shot_axis="shot", preserve_shots=False, intrinsic_axes=()
    ):
        """Bind stream buffers and dataset axes to the QUA loop order.

        Call after defining sweep_axes and before declaring streams. Default
        order is shot, then the sweep axes in insertion order (excluding qubit).
        Supply loop_order explicitly for nested shots, e.g. RB. intrinsic_axes
        are vector dimensions already supplied by the hardware (ADC samples).
        Cloud analyses use preserve_shots=True and their existing n_runs axis.
        """
        import xarray as xr
        from calibration_utils.state_acquisition import AcquisitionLayout

        single_shot = self.single_shot_acquisition
        if preserve_shots and not single_shot:
            raise ValueError(f"{self.name} requires acquisition='single_shot' for cloud analysis")
        axes = self.namespace["sweep_axes"]
        sweep_names = [name for name in axes if name not in ("qubit", shot_axis, *intrinsic_axes)]
        order = tuple(loop_order) if loop_order is not None else (shot_axis, *sweep_names)
        if set(order) != {shot_axis, *sweep_names}:
            raise ValueError("loop_order must contain every sweep axis and the shot axis exactly once")
        count = self.parameters.num_shots
        loops = tuple((name, count if name == shot_axis else len(axes[name])) for name in order)
        layout = AcquisitionLayout(loops, shot_axis=shot_axis, single_shot=single_shot)
        self.namespace["acquisition_layout"] = layout
        result_axes = {"qubit": axes["qubit"]} if "qubit" in axes else {}
        for name in order:
            if name == shot_axis:
                if single_shot:
                    result_axes[name] = axes.get(name, xr.DataArray(
                        np.arange(count), dims=name, attrs={"long_name": "shot index"}))
            else:
                result_axes[name] = axes[name]
        result_axes.update({name: axes[name] for name in intrinsic_axes})
        self.namespace["sweep_axes"] = result_axes

    def save_acquisition_stream(self, stream, name, *, split_outer_axis=None):
        """Finalize one state-population or analog stream using the shared layout."""
        layout = self.namespace["acquisition_layout"]
        if split_outer_axis is not None:
            from dataclasses import replace
            if layout.loops[0][0] != split_outer_axis or split_outer_axis == layout.shot_axis:
                raise ValueError("split_outer_axis must name the outermost non-shot loop")
            layout = replace(layout, loops=layout.loops[1:])
        layout.save(stream, name, save_all=split_outer_axis is not None)

    def process_readout_streams(self, i, state_st=None, I_st=None, Q_st=None):
        """Finalize a qubit's discriminated-state streams or I/Q streams."""
        if state_st is not None:
            self.save_acquisition_stream(state_st[i], f"state{i + 1}")
        else:
            if I_st is None or Q_st is None:
                raise ValueError("Readout streams require state_st or both I_st and Q_st")
            self.save_acquisition_stream(I_st[i], f"I{i + 1}")
            self.save_acquisition_stream(Q_st[i], f"Q{i + 1}")

    def declare_state_stream(self):
        from qm.qua import declare_stream
        from utils.experiment_readout import PopulationStreams, readout_states
        if self.single_shot_acquisition:
            return declare_stream()
        return PopulationStreams(readout_states(self.parameters))

    def save_readout_state(self, state, streams):
        from qm.qua import save
        if self.single_shot_acquisition:
            save(state, streams)
        else:
            streams.save_shot(state)

    def prepare_acquisition_results(self):
        """Reduce shots for analysis after saving raw data, preserving every shot."""
        from calibration_utils.state_acquisition import prepare_acquisition_dataset
        if "ds_raw" in self.results:
            self.results["ds_raw"] = prepare_acquisition_dataset(
                self.results["ds_raw"], self.parameters)

    def should_load_data(self) -> bool:
        return self.load_data_id is not None

    def should_simulate(self) -> bool:
        return self.simulate_requested and not self.should_load_data()

    def should_execute(self) -> bool:
        return not self.simulate_requested and not self.should_load_data()

    def _dc_bias_qubit_name(self, configured_names: set[str]) -> str:
        if self.qubit in configured_names:
            return str(self.qubit)
        if len(configured_names) == 1:
            return next(iter(configured_names))

        active_names = set(getattr(self.machine, "active_qubit_names", ()))
        active_configured = active_names & configured_names
        if len(active_configured) == 1:
            return next(iter(active_configured))

        selected_qubits = self.namespace.get("qubits", ())
        selected_names = {
            str(getattr(qubit, "name", qubit)) for qubit in selected_qubits
        }
        selected_configured = selected_names & configured_names
        if len(selected_configured) == 1:
            return next(iter(selected_configured))

        raise CalibrationError(
            "Automatic DC bias requires exactly one selected qubit with a "
            "configured dc_bias_v."
        )

    @contextmanager
    def automatic_dc_bias(self) -> Iterable[None]:
        """Apply the selected nonzero profile bias during real execution only."""
        dc_bias = getattr(self.machine, "dc_bias", None)
        qubit_biases_v = getattr(dc_bias, "qubit_biases_v", {})
        if not qubit_biases_v:
            yield
            return

        qubit_name = self._dc_bias_qubit_name(set(qubit_biases_v))
        voltage_v = dc_bias.voltage_for_qubit(qubit_name)
        if voltage_v == 0:
            yield
            return

        self.log(
            f"Applying DC bias for {qubit_name}: {voltage_v:g} V on "
            f"channel {dc_bias.output_channel}."
        )
        with dc_bias.applied_for_qubit(qubit_name):
            yield

    def run(self) -> CalibrationStatus:
        """Run the standard calibration lifecycle."""
        from utils.result_fetching import AcquisitionStopped

        loaded = False
        raw_data_saved = False
        figures_saved = False
        ai_review_saved = False
        profile_update_proposed = False
        self._start_run_timer()

        try:
            if self.should_load_data():
                if len(inspect.signature(self.load_data).parameters) == 0:
                    self.load_data()
                else:
                    self.load_data(self.load_data_id)
                loaded = True
            else:
                if self.should_execute():
                    assert_outputs_allowed()
                self.namespace["qua_program"] = self.create_qua_program()
                if self.should_execute() and self.options.report_runtime_estimate:
                    self.report_runtime_estimate()
                if self.should_simulate():
                    self.simulate_qua_program()
                elif self.should_execute():
                    # Re-check after QUA construction in case an operator
                    # engaged the latch while the program was being built.
                    assert_outputs_allowed()
                    execution_started_s = time.perf_counter()
                    try:
                        with self.automatic_dc_bias():
                            self.execute_qua_program()
                    finally:
                        self.namespace["execution_duration_s"] = (
                            time.perf_counter() - execution_started_s
                        )
                    if self.options.save_raw_data:
                        self.save_raw_results()
                        raw_data_saved = True

            if not self.simulate_requested:
                self.prepare_acquisition_results()
                if getattr(self.parameters, "use_readout_mitigation", False):
                    self.apply_readout_mitigation()
                    if raw_data_saved:
                        self.save_readout_mitigated_results()
                if self.options.analyse_data:
                    self.analyse_data()
                    if self.options.save_analysis_result:
                        self.save_analysis_result()
                if self.options.plot_data:
                    self.plot_data()
                if self.options.save_figures:
                    figures_saved = self.save_figures()
                if self.options.ai_review:
                    ai_review_saved = self.save_ai_review()
                if self.options.update_state and not self.namespace.get("acquisition_interrupted"):
                    self.update_state()
                if self.options.propose_profile_update and not self.namespace.get("acquisition_interrupted"):
                    profile_update_proposed = self._propose_profile_update_from_options()
        except AcquisitionStopped as error:
            self.namespace["acquisition_interrupted"] = True
            self.log(str(error))
        finally:
            self._finish_run_timer()
            self.cleanup()

        return CalibrationStatus(
            name=self.name,
            mode="load" if loaded else ("simulate" if self.simulate_requested else "execute"),
            simulated=self.simulate_requested,
            loaded=loaded,
            raw_data_saved=raw_data_saved,
            figures_saved=figures_saved,
            ai_review_saved=ai_review_saved,
            profile_update_proposed=profile_update_proposed,
            outcomes=dict(self.outcomes),
            interrupted=bool(self.namespace.get("acquisition_interrupted")),
        )

    @abstractmethod
    def create_qua_program(self) -> Any:
        """Create and return the QUA program.

        Subclasses should also populate ``namespace["sweep_axes"]`` for
        fetching xarray data.
        """

    def simulate_qua_program(self) -> None:
        """Simulate the QUA program and store samples/report in results."""
        from utils.simulation import simulate_and_plot

        qmm = self.connect_machine()
        config = self.machine.generate_config()
        samples, figure, wf_report = simulate_and_plot(
            qmm,
            config,
            self.namespace["qua_program"],
            self.parameters,
        )
        self.results["simulation"] = {
            "figure": figure,
            "wf_report": wf_report,
            "samples": samples,
        }
        if self.options.plot_data:
            plt.show()

    def fetch_result_dataset(self, job):
        """Fetch results with live progress, recovering completed sweeps on Ctrl+C."""
        from utils.result_fetching import AcquisitionError, fetch_result_dataset

        progress_started = time.time()

        def show_progress(count, started):
            nonlocal progress_started
            progress_started = started
            self.report_progress(count, start_time=started)

        try:
            layout = self.namespace.get("acquisition_layout")
            partial_axis = None
            if layout is not None and (layout.single_shot or layout.loops[0][0] != layout.shot_axis):
                partial_axis = layout.loops[0][0]
            dataset = fetch_result_dataset(
                job, self.namespace["sweep_axes"], on_progress=show_progress,
                partial_axis=partial_axis,
            )
            if dataset.attrs.get("acquisition_interrupted"):
                self.namespace["acquisition_interrupted"] = True
                total = self.progress_total()
                if total is not None:
                    dataset.attrs["requested_iterations"] = int(total)
                count = dataset.attrs.get("completed_iterations")
                detail = "the completed measurements"
                if count is not None:
                    axis = dataset.attrs["partial_axis"]
                    detail = f"{count} completed iterations along {axis}"
                self.log(f"Acquisition stopped; continuing analysis with {detail}.")
                return dataset
            # The zero-based QUA counter stops at total - 1. Mark completion only
            # after the job and its measurement buffers have finished successfully.
            total = self.progress_total()
            if total is not None:
                self.report_progress(total, start_time=progress_started)
            return dataset
        except AcquisitionError as error:
            raise CalibrationError(str(error)) from error
        finally:
            # Keep the original failure / interruption even if report retrieval fails.
            try:
                report = job.execution_report()
                self.namespace["execution_report"] = report
                self.log(report)
            except Exception as error:
                self.log(f"Could not retrieve the job execution report: {error}")

    def execute_qua_program(self) -> None:
        """Execute the QUA program and fetch xarray data into ``ds_raw``."""
        from utils.qm_session import qm_session

        if "sweep_axes" not in self.namespace:
            raise CalibrationError("create_qua_program() must set namespace['sweep_axes'].")

        qmm = self.connect_machine()
        config = self.machine.generate_config()
        with qm_session(qmm, config, timeout=self.timeout) as qm:
            self.namespace["job"] = job = qm.execute(self.namespace["qua_program"])
            dataset = self.fetch_result_dataset(job)
        self.results["ds_raw"] = self.annotate_readout_dataset(dataset)

    def report_progress(self, count, *, start_time):
        """Report the completed outer iterations; batch workflows may override this."""
        from calibrations.runtime_estimation import progress_counter
        total = self.progress_total()
        if total is not None:
            progress_counter(count, total, start_time=start_time)

    def progress_total(self) -> int | None:
        return getattr(self.parameters, "num_shots", None)

    def estimate_runtime(self) -> Any:
        """Estimate execution time from sweep size and comparable saved runs."""
        from calibrations.runtime_estimation import estimate_runtime

        estimate = estimate_runtime(
            experiment_name=self.name,
            axes=self.namespace.get("sweep_axes"),
            parameters=self.parameters,
            progress_total=self.progress_total(),
            output_root=self.saver.output_root,
        )
        self.namespace["runtime_estimate"] = estimate.to_dict()
        return estimate

    def report_runtime_estimate(self) -> None:
        """Log workload immediately before hardware execution."""
        from calibrations.runtime_estimation import format_duration

        estimate = self.estimate_runtime()
        self.log(
            "Planned workload: "
            f"{estimate.sweep_points:,} sweep points x "
            f"{estimate.repetitions:,} repetitions = "
            f"{estimate.workload_units:,} normalized workload units."
        )
        if estimate.estimated_seconds is None:
            self.log(
                "Estimated execution time: unavailable (no comparable saved run); "
                "live ETA starts after one outer iteration completes."
            )
            return
        self.log(
            f"Estimated execution time: about {format_duration(estimate.estimated_seconds)} "
            f"from {estimate.historical_runs} comparable saved run(s); "
            "live ETA will refine it."
        )

    def annotate_readout_dataset(self, dataset):
        from utils.experiment_readout import readout_operation, readout_states
        dataset.attrs.update(readout_states=readout_states(self.parameters),
                             readout_operation=readout_operation(self.parameters))
        if "state" in dataset:
            dataset["state"].attrs.update(long_name="excited-state population", population_state="e")
        from calibration_utils.state_acquisition import annotate_acquisition
        return annotate_acquisition(dataset, self.parameters)

    def _mitigate_population_dataset(self, dataset, qubits, strength):
        from utils.experiment_readout import readout_operation, readout_states
        from utils.readout_macro import readout_settings
        states = list(dataset.attrs.get("readout_states", readout_states(self.parameters)))
        operation = dataset.attrs.get("readout_operation", readout_operation(self.parameters))
        names = [f"population_{label}" for label in states]
        if not all(name in dataset for name in names):
            raise CalibrationError("Three-state mitigation requires separately measured population_g/e/f.")
        corrected = {name: dataset[name].astype(float).copy(deep=True) for name in names}
        for q in qubits:
            matrix = np.asarray(readout_settings(q, operation).get("confusion_matrix"), dtype=float)
            size = len(states)
            if matrix.shape != (size, size) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < size:
                raise CalibrationError(f"{q.name} {operation} needs a finite invertible {size}x{size} confusion matrix")
            has_qubit = "qubit" in dataset[names[0]].dims
            if not has_qubit and len(qubits) != 1:
                raise CalibrationError("Multiple qubits require a qubit dimension for mitigation")
            measured = [dataset[name].sel(qubit=q.name) if has_qubit else dataset[name] for name in names]
            inverse = np.linalg.inv(matrix)
            for index, name in enumerate(names):
                fully_corrected = sum(measured[j] * inverse[j, index] for j in range(size))
                result = measured[index] + strength * (fully_corrected - measured[index])
                if has_qubit:
                    corrected[name].loc[{"qubit": q.name}] = result
                else:
                    corrected[name] = result
        additions = {f"{name}_unmitigated": dataset[name].copy(deep=True) for name in names}
        additions.update(corrected)
        additions["state_unmitigated"] = dataset["state"].copy(deep=True)
        additions["state"] = corrected["population_e"].copy(deep=True)
        additions["state"].attrs.update(readout_mitigated=True,
            readout_mitigation_method="inverse_assignment_matrix", readout_mitigation_strength=strength)
        self.results["ds_raw"] = dataset.assign(additions)

    def apply_readout_mitigation(self) -> None:
        """Correct GE or GEF populations using the matching assignment matrix.

        IQ-blobs stores a matrix whose rows are prepared states and whose columns
        are measured states.  Therefore ``p_measured = p_true @ matrix`` and the
        fully mitigated population is obtained by applying the matrix inverse.
        Numeric ``use_readout_mitigation`` values blend between the measured and
        fully mitigated populations, which regularizes noisy assignment matrices.
        The result is intentionally not clipped because clipping would bias it.
        """
        self.prepare_acquisition_results()
        if not getattr(self.parameters, "use_state_discrimination", False):
            raise CalibrationError(
                "Readout mitigation requires use_state_discrimination=True."
            )

        mitigation_strength = float(
            getattr(self.parameters, "use_readout_mitigation", 1.0)
        )
        if not np.isfinite(mitigation_strength) or not 0 < mitigation_strength <= 1:
            raise CalibrationError(
                "use_readout_mitigation must be False or a strength in the interval (0, 1]."
            )

        dataset = self.results.get("ds_raw")
        if dataset is None or "state" not in dataset:
            raise CalibrationError(
                "Readout mitigation requires a discriminated 'state' variable in ds_raw."
            )
        if dataset["state"].attrs.get("readout_mitigated", False):
            return

        qubits = self.namespace.get("qubits")
        if qubits is None:
            qubits = self.get_qubits()
        qubits = list(qubits)
        if not qubits:
            raise CalibrationError("Readout mitigation requires at least one selected qubit.")

        if "population_g" in dataset or len(getattr(self.parameters, "readout_states", ["g", "e"])) == 3:
            self._mitigate_population_dataset(dataset, qubits, mitigation_strength)
            return

        state = dataset["state"]
        corrected = state.astype(float).copy(deep=True)
        has_qubit_axis = "qubit" in state.dims
        if not has_qubit_axis and len(qubits) != 1:
            raise CalibrationError(
                "The state data has no 'qubit' dimension, but multiple qubits were selected."
            )

        for qubit in qubits:
            qubit_name = str(getattr(qubit, "name", qubit))
            matrix_value = getattr(getattr(qubit, "resonator", None), "confusion_matrix", None)
            if matrix_value is None:
                raise CalibrationError(
                    f"No readout confusion matrix is calibrated for {qubit_name}. "
                    "Run IQ blobs and apply its profile update first."
                )
            matrix = np.asarray(matrix_value, dtype=float)
            if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
                raise CalibrationError(
                    f"Readout confusion matrix for {qubit_name} must be a finite 2x2 matrix; "
                    f"got shape {matrix.shape}."
                )
            if np.linalg.matrix_rank(matrix) < 2:
                raise CalibrationError(
                    f"Readout confusion matrix for {qubit_name} is singular and cannot mitigate data."
                )

            inverse = np.linalg.inv(matrix)
            measured_excited = state.sel(qubit=qubit_name) if has_qubit_axis else state
            fully_mitigated_excited = (
                (1.0 - measured_excited) * inverse[0, 1]
                + measured_excited * inverse[1, 1]
            )
            mitigated_excited = measured_excited + mitigation_strength * (
                fully_mitigated_excited - measured_excited
            )
            if has_qubit_axis:
                corrected.loc[{"qubit": qubit_name}] = mitigated_excited
            else:
                corrected = mitigated_excited

        corrected.attrs = dict(state.attrs)
        corrected.attrs.update(
            {
                "readout_mitigated": True,
                "readout_mitigation_method": "inverse_assignment_matrix",
                "readout_mitigation_strength": mitigation_strength,
            }
        )
        self.results["ds_raw"] = dataset.assign(
            state_unmitigated=state.copy(deep=True),
            state=corrected,
        )

    def save_raw_results(self, *, now: datetime | None = None) -> Path:
        """Save ``results['ds_raw']`` and a profile snapshot."""
        if "ds_raw" not in self.results:
            raise CalibrationError("No raw dataset found in results['ds_raw'].")
        run_directory = self.saver.save_xarray(
            self.name,
            self.results["ds_raw"],
            profile_name=self.active_profile_name(),
            parameters=self.parameters,
            extra_metadata=self.run_timing_metadata(),
            now=now,
        )
        self.namespace["calibration_run_directory"] = run_directory
        self.log(f"Raw calibration results saved to {run_directory}")
        return run_directory

    def save_readout_mitigated_results(self) -> Path:
        """Save the mitigated companion while preserving raw ``results.npz``."""
        run_directory = self.namespace.get("calibration_run_directory")
        if run_directory is None:
            raise CalibrationError(
                "Cannot save mitigated results before the unmitigated run is saved."
            )
        output_path = self.saver.save_readout_mitigated_xarray(
            run_directory,
            self.results["ds_raw"],
            strength=float(self.parameters.use_readout_mitigation),
        )
        self.namespace["readout_mitigated_results_path"] = output_path
        self.log(f"Readout-mitigated results saved to {output_path}")
        return output_path

    def save_arrays(
        self,
        sweep: Mapping[str, Any] | Any,
        results: Mapping[str, Any] | Any,
        *,
        now: datetime | None = None,
    ) -> Path:
        """Save explicit sweep/result arrays and a profile snapshot."""
        run_directory = self.saver.save(
            self.name,
            sweep,
            results,
            profile_name=self.active_profile_name(),
            parameters=self.parameters,
            extra_metadata=self.run_timing_metadata(),
            now=now,
        )
        self.namespace["calibration_run_directory"] = run_directory
        self.log(f"Calibration arrays saved to {run_directory}")
        return run_directory

    def save(self) -> Path | None:
        """Save available raw data and figures.

        Returns the raw-data run directory when ``results['ds_raw']`` exists.
        """
        run_directory = None
        if "ds_raw" in self.results:
            run_directory = self.save_raw_results()
        self.save_figures()
        return run_directory

    def save_qua_debug_script(self, output_directory: str | Path | None = None) -> Path:
        """Serialize the current QUA program and generated config for debugging."""
        from qm import generate_qua_script

        if "qua_program" not in self.namespace:
            raise CalibrationError("No QUA program found in namespace['qua_program'].")
        output_directory = (
            Path(output_directory)
            if output_directory is not None
            else Path(__file__).resolve().parents[1] / "debug"
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f"{self.name}.py"
        config = self.machine.generate_config()
        with output_path.open("w", encoding="utf-8") as source_file:
            print(generate_qua_script(self.namespace["qua_program"], config), file=source_file)
        self.log(f"Serialized QUA debug script saved to {output_path}")
        return output_path

    def load_data(self, run_directory: str | Path) -> None:
        """Load a run saved by :class:`CalibrationSaver` into ``results['ds_raw']``."""
        self.results["ds_raw"] = self.load_saved_run(run_directory)
        self.get_qubits()

    def load_from_id(self, run_directory: str | Path) -> None:
        """Compatibility shim for legacy node-style data loading."""
        self.results["ds_raw"] = self.load_saved_run(run_directory)

    def load_saved_run(self, run_directory: str | Path) -> Any:
        """Reconstruct an xarray dataset from ``sweep.npz`` and ``results.npz``."""
        import xarray as xr

        run_directory = Path(run_directory)
        sweep_path = run_directory / "sweep.npz"
        results_path = run_directory / "results.npz"
        if not sweep_path.is_file() or not results_path.is_file():
            raise FileNotFoundError(
                f"Expected sweep.npz and results.npz in calibration run: {run_directory}"
            )

        metadata_path = run_directory / "metadata.json"
        schema = {}
        if metadata_path.is_file():
            schema = json.loads(metadata_path.read_text(encoding="utf-8")).get("dataset_schema", {})
        if schema:
            with np.load(sweep_path, allow_pickle=False) as sweeps, np.load(
                results_path, allow_pickle=False
            ) as results:
                return xr.Dataset(
                    data_vars={name: (spec["dims"], np.array(results[name]), spec.get("attrs", {}))
                               for name, spec in schema["data_vars"].items()},
                    coords={name: (spec["dims"], np.array(sweeps[name]), spec.get("attrs", {}))
                            for name, spec in schema["coords"].items()},
                    attrs=schema.get("attrs", {}),
                )

        with np.load(sweep_path, allow_pickle=False) as sweep_file:
            coordinates = {
                name: np.array(sweep_file[name])
                for name in sweep_file.files
            }
        with np.load(results_path, allow_pickle=False) as results_file:
            data_vars = {
                name: self._array_to_data_var(np.array(results_file[name]), coordinates)
                for name in results_file.files
            }
        return xr.Dataset(data_vars=data_vars, coords=coordinates)

    def analyse(self) -> None:
        """Optional analysis hook.

        The British spelling matches the existing calibration scripts. New
        subclasses may override either this method or ``analyse_data``.
        """
        self.prepare_acquisition_results()
        analysis = self.create_analysis()
        if analysis is None:
            return
        if "ds_raw" not in self.results:
            raise CalibrationError("No raw dataset found in results['ds_raw'].")
        self.apply_analysis_result(analysis.run(self.results["ds_raw"]))

    def analyse_data(self) -> None:
        """Backward-compatible alias for ``analyse``."""
        self.analyse()

    def create_analysis(self) -> Any | None:
        """Return a calibration analysis object, or ``None`` when unmanaged."""
        return None

    def apply_analysis_result(self, analysis_result: Any) -> None:
        """Expose a structured analysis result through legacy result keys."""
        self.results["analysis"] = analysis_result
        if hasattr(analysis_result, "ds_processed") and analysis_result.ds_processed is not None:
            self.results["ds_raw"] = analysis_result.ds_processed
        if hasattr(analysis_result, "ds_fit") and analysis_result.ds_fit is not None:
            self.results["ds_fit"] = analysis_result.ds_fit
        if hasattr(analysis_result, "fit_results"):
            self.results["fit_results"] = dict(analysis_result.fit_results)
        if hasattr(analysis_result, "outcomes"):
            self.outcomes = dict(analysis_result.outcomes)

    def save_analysis_result(self) -> bool:
        """Save ``results['analysis']`` into the calibration run directory."""
        analysis_result = self.results.get("analysis")
        run_directory = self.namespace.get("calibration_run_directory")
        if analysis_result is None or run_directory is None:
            return False
        output_path = self.saver.save_analysis_result(run_directory, analysis_result)
        self.log(f"Calibration analysis result saved to {output_path}")
        return True

    def plot_data(self) -> None:
        """Optional plotting hook. Store figures in ``results['figures']``."""

    def save_figures(self) -> bool:
        """Save ``results['figures']`` when raw data has a run directory."""
        figures = self.results.get("figures")
        run_directory = self.namespace.get("calibration_run_directory")
        if not figures or run_directory is None:
            return False
        figures_directory = self.saver.save_figures(run_directory, figures)
        self.log(f"Calibration figures saved to {figures_directory}")
        return True

    def save_ai_review(self) -> bool:
        """Review saved figures with the configured NVIDIA Ising calibration endpoint."""
        run_directory = self.namespace.get("calibration_run_directory")
        if run_directory is None:
            self.log("AI review skipped because no calibration run directory was saved.")
            return False
        try:
            from calibration_ai import CalibrationAIReviewer

            review = CalibrationAIReviewer().review_run(run_directory)
        except Exception as error:
            self.log(f"AI review failed: {error}")
            return False

        self.namespace["ai_review"] = review.json_path
        self.log(f"AI calibration review saved to {review.json_path}")
        try:
            payload = json.loads(review.json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        status = payload.get("pass_fail", "unknown")
        summary = payload.get("summary")
        if summary:
            self.log(f"AI review: {status} - {summary}")
        else:
            self.log(f"AI review: {status}")
        return True

    def profile_update_results(self, *, successful_only=True):
        """Yield selected qubits and fits, respecting explicit failed outcomes."""
        fits = self.results.get("fit_results", {})
        for qubit in self.namespace.get("qubits", ()):
            fit = fits.get(qubit.name, {})
            outcome = self.outcomes.get(qubit.name)
            if successful_only and outcome != "successful":
                if outcome is not None or not fit.get("success", False):
                    continue
            yield qubit, fit

    @staticmethod
    def _profile_update_value(source, qubit, fit):
        """A string selects a fit key; a callable computes a value; other values are literal."""
        if callable(source):
            value = source(qubit, fit)
        elif isinstance(source, str):
            value = fit[source]
        else:
            value = source
        return value.item() if isinstance(value, np.generic) else value

    def _profile_field_updates(self, prefix, fields):
        updates = {}
        for qubit, fit in self.profile_update_results():
            for field, source in fields.items():
                value = self._profile_update_value(source, qubit, fit)
                updates[f"{prefix}.{qubit.name}.{field}"] = value
        return updates

    def qubit_profile_updates(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Map relative qubit fields to fit keys, callables ``(qubit, fit)``, or literals."""
        return self._profile_field_updates("qubits.json.qubits", fields)

    def metric_profile_updates(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Map relative metric fields to fit keys, callables ``(qubit, fit)``, or literals."""
        return self._profile_field_updates("metrics.json.qubits", fields)

    def pulse_profile_updates(
        self, operation: str, *, pulse_type: str | None = None, amplitude_scale=None, **fields,
    ) -> dict[str, float]:
        """Update dedicated drive pulses; optionally scale the profile amplitude by a fitted factor.

        Example: ``self.pulse_profile_updates("x180", amplitude="opt_amp")``.
        Derived operation aliases are skipped so their parent pulse is not changed.
        """
        from profiles import load_profile

        profile = load_profile(self.active_profile_name())
        qubits = profile["qubits"]["qubits"]
        pulses = profile["pulses"]["pulses"]
        updates = {}
        for qubit, fit in self.profile_update_results():
            pulse_name = qubits[qubit.name]["operations"].get(operation)
            pulse = pulses.get(qubit.name, {}).get(pulse_name)
            if pulse is None or (pulse_type is not None and pulse.get("type") != pulse_type):
                expected = f"a dedicated {pulse_type} pulse" if pulse_type else "a dedicated profile pulse"
                self.log(
                    f"Profile update skipped for {qubit.name}: operation {operation!r} "
                    f"does not map to {expected}."
                )
                continue
            try:
                values = {
                    field: float(self._profile_update_value(value, qubit, fit))
                    for field, value in fields.items()
                }
                if amplitude_scale is not None:
                    factor = float(self._profile_update_value(amplitude_scale, qubit, fit))
                    if not np.isfinite(factor) or factor <= 0:
                        raise ValueError("amplitude scale must be finite and positive")
                    values["amplitude"] = float(pulse["amplitude"]) * factor
                if not all(np.isfinite(value) for value in values.values()):
                    raise ValueError("pulse values must be finite")
                if abs(values.get("amplitude", 0)) > 0.7:
                    raise ValueError("pulse amplitude exceeds 0.7 V")
            except (KeyError, TypeError, ValueError) as error:
                self.log(f"Profile update skipped for {qubit.name}: {error}.")
                continue
            updates.update({f"pulses.json.pulses.{qubit.name}.{pulse_name}.{field}": value
                            for field, value in values.items()})
        return updates

    def readout_profile_updates(
        self, *, frequency=None, amplitude=None, fidelity=None, settings=None,
        invalidate_discrimination=False, successful_only=True,
    ):
        """Build updates for the selected GE/GEF readout and invalidate stale discrimination.

        Scalar sources use the same fit-key/callable convention as pulse updates.
        ``settings`` is a literal mapping or a callable returning settings for a
        qubit, or None when that qubit's readout fit is unsuitable.
        """
        section = "readout_gef" if len(self.parameters.readout_states) == 3 else "readout"
        frequency_field = (
            "readout_gef.frequency_hz" if section == "readout_gef" else "frequencies_hz.resonator"
        )
        updates = {}
        for qubit, fit in self.profile_update_results(successful_only=successful_only):
            values = {} if settings is None else self._profile_update_value(settings, qubit, fit)
            if values is None:
                continue
            values = dict(values)
            pending = {}
            if frequency is not None:
                value = float(self._profile_update_value(frequency, qubit, fit))
                if not np.isfinite(value):
                    self.log(f"Profile update skipped for {qubit.name}: non-finite readout frequency.")
                    continue
                pending[f"qubits.json.qubits.{qubit.name}.{frequency_field}"] = value
            if amplitude is not None:
                value = float(self._profile_update_value(amplitude, qubit, fit))
                if not np.isfinite(value) or abs(value) > 0.7:
                    self.log(
                        f"Profile update skipped for {qubit.name}: "
                        "non-finite readout amplitude or amplitude exceeds 0.7 V."
                    )
                    continue
                operation = self.parameters.readout_operation
                pulse = getattr(qubit.resonator, "readout_pulse_names", {}).get(operation, operation)
                pending[f"pulses.json.pulses.{qubit.name}.{pulse}.amplitude"] = value
            if frequency is not None or amplitude is not None or invalidate_discrimination:
                values.update(gef_centers=None, confusion_matrix=None)
            pending.update({f"qubits.json.qubits.{qubit.name}.{section}.{key}": value
                            for key, value in values.items()})
            reset = getattr(self.parameters, "reset_type", None)
            if fidelity is not None and section == "readout" and reset in {"active", "thermal"}:
                if not isinstance(fidelity, str) or fidelity in fit:
                    value = float(self._profile_update_value(fidelity, qubit, fit))
                    pending[f"metrics.json.qubits.{qubit.name}.readout.fidelity_percent.{reset}"] = value
            updates.update(pending)
        return updates

    def profile_updates(self) -> Mapping[str, Any]:
        """Return profile update paths to stage, or an empty mapping."""
        return {}

    def propose_profile_update(self, *, apply: bool = True) -> bool:
        """Stage profile updates and optionally ask for confirmation to apply."""
        updates = dict(self.profile_updates())
        if not updates:
            return False
        proposal = self.profile_updater.stage(
            self.name,
            updates,
            profile_name=self.active_profile_name(),
        )
        self.namespace["profile_update_proposal"] = proposal
        if apply:
            self.profile_updater.confirm_and_apply(proposal)
        return True

    def _propose_profile_update_from_options(self) -> bool:
        """Call subclass profile-update hooks while respecting base options."""
        signature = inspect.signature(self.propose_profile_update)
        if "apply" in signature.parameters:
            return bool(
                self.propose_profile_update(apply=self.options.apply_profile_update)
            )
        if not self.options.apply_profile_update:
            self.log(
                "Profile update skipped because this calibration overrides "
                "propose_profile_update() without an apply option."
            )
            return False
        return bool(self.propose_profile_update())

    @contextmanager
    def record_state_updates(self) -> Iterable[None]:
        """Compatibility shim for existing code that used QualibrationNode."""
        yield

    def update_state(self) -> None:
        """Optional in-memory machine update hook."""

    def cleanup(self) -> None:
        """Optional cleanup hook, such as reverting tracked temporary updates."""

    def active_profile_name(self) -> str:
        return self.profile_name or current_profile_name()

    def _start_run_timer(self) -> None:
        now = datetime.now().astimezone()
        self.namespace["run_started_at"] = now.isoformat()
        self.namespace["run_timer_started_s"] = time.perf_counter()

    def _finish_run_timer(self) -> None:
        started_s = self.namespace.get("run_timer_started_s")
        if started_s is None:
            return
        duration_s = time.perf_counter() - float(started_s)
        self.namespace["run_finished_at"] = datetime.now().astimezone().isoformat()
        self.namespace["run_duration_s"] = duration_s
        self._update_saved_run_timing_metadata()

    def run_timing_metadata(self) -> dict[str, Any]:
        return {
            "run_started_at": self.namespace.get("run_started_at"),
            "acquisition_interrupted": bool(self.namespace.get("acquisition_interrupted")),
            **(
                {"run_finished_at": self.namespace["run_finished_at"]}
                if "run_finished_at" in self.namespace
                else {}
            ),
            **(
                {"run_duration_s": self.namespace["run_duration_s"]}
                if "run_duration_s" in self.namespace
                else {}
            ),
            **(
                {"execution_duration_s": self.namespace["execution_duration_s"]}
                if "execution_duration_s" in self.namespace
                else {}
            ),
            **(
                {"runtime_estimate": self.namespace["runtime_estimate"]}
                if "runtime_estimate" in self.namespace
                else {}
            ),
        }

    def _update_saved_run_timing_metadata(self) -> None:
        run_directory = self.namespace.get("calibration_run_directory")
        if run_directory is None:
            return
        metadata_path = Path(run_directory) / "metadata.json"
        if not metadata_path.is_file():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        metadata.update(self.run_timing_metadata())
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")

    @staticmethod
    def _array_to_data_var(array: np.ndarray, coordinates: Mapping[str, np.ndarray]) -> Any:
        matching_dims = [
            name
            for name, coordinate in coordinates.items()
            if coordinate.ndim == 1 and coordinate.shape[0] in array.shape
        ]
        if len(matching_dims) == array.ndim:
            return (matching_dims, array)
        dims = tuple(f"dim_{index}" for index in range(array.ndim))
        return (dims, array)
