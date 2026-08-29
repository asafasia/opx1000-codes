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
        return qubits

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
                if self.options.update_state:
                    self.update_state()
                if self.options.propose_profile_update:
                    profile_update_proposed = self._propose_profile_update_from_options()
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

    def execute_qua_program(self) -> None:
        """Execute the QUA program and fetch xarray data into ``ds_raw``."""
        from qualang_tools.multi_user import qm_session
        from calibrations.runtime_estimation import progress_counter
        from qualibration_libs.data import XarrayDataFetcher

        if "sweep_axes" not in self.namespace:
            raise CalibrationError("create_qua_program() must set namespace['sweep_axes'].")

        qmm = self.connect_machine()
        config = self.machine.generate_config()
        total = self.progress_total()
        with qm_session(qmm, config, timeout=self.timeout) as qm:
            self.namespace["job"] = job = qm.execute(self.namespace["qua_program"])
            data_fetcher = XarrayDataFetcher(job, self.namespace["sweep_axes"])
            dataset = None
            for dataset in data_fetcher:
                if total is not None:
                    progress_counter(
                        data_fetcher.get("n", 0),
                        total,
                        start_time=data_fetcher.t_start,
                    )
            self.log(job.execution_report())

        if dataset is None:
            raise CalibrationError("Execution finished without fetched data.")
        self.results["ds_raw"] = dataset

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

    def apply_readout_mitigation(self) -> None:
        """Correct binary state populations using calibrated assignment matrices.

        IQ-blobs stores a matrix whose rows are prepared states and whose columns
        are measured states.  Therefore ``p_measured = p_true @ matrix`` and the
        fully mitigated population is obtained by applying the matrix inverse.
        Numeric ``use_readout_mitigation`` values blend between the measured and
        fully mitigated populations, which regularizes noisy assignment matrices.
        The result is intentionally not clipped because clipping would bias it.
        """
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
