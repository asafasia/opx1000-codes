"""Base class for class-oriented calibration experiments.

The v2 calibration shape keeps the useful parts of the existing nodes
(`parameters`, `machine`, `namespace`, `results`, `outcomes`, and `log`) while
moving the lifecycle into ordinary methods that can be overridden and tested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import inspect
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar

import numpy as np
import matplotlib.pyplot as plt

from calibration_io import CalibrationSaver
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
    profile_update_proposed: bool
    outcomes: Mapping[str, str] = field(default_factory=dict)


@dataclass
class CalibrationOptions:
    """Runtime switches for the shared calibration lifecycle."""

    save_raw_data: bool = True
    save_figures: bool = True
    analyse_data: bool = True
    plot_data: bool = True
    update_state: bool = True
    propose_profile_update: bool = True
    apply_profile_update: bool = True


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

    def run(self) -> CalibrationStatus:
        """Run the standard calibration lifecycle."""
        loaded = False
        raw_data_saved = False
        figures_saved = False
        profile_update_proposed = False

        try:
            if self.should_load_data():
                if len(inspect.signature(self.load_data).parameters) == 0:
                    self.load_data()
                else:
                    self.load_data(self.load_data_id)
                loaded = True
            else:
                self.namespace["qua_program"] = self.create_qua_program()
                if self.should_simulate():
                    self.simulate_qua_program()
                elif self.should_execute():
                    self.execute_qua_program()
                    if self.options.save_raw_data:
                        self.save_raw_results()
                        raw_data_saved = True

            if not self.simulate_requested:
                if self.options.analyse_data:
                    self.analyse_data()
                if self.options.plot_data:
                    self.plot_data()
                if self.options.save_figures:
                    figures_saved = self.save_figures()
                if self.options.update_state:
                    self.update_state()
                if self.options.propose_profile_update:
                    profile_update_proposed = self._propose_profile_update_from_options()
        finally:
            self.cleanup()

        return CalibrationStatus(
            name=self.name,
            mode="load" if loaded else ("simulate" if self.simulate_requested else "execute"),
            simulated=self.simulate_requested,
            loaded=loaded,
            raw_data_saved=raw_data_saved,
            figures_saved=figures_saved,
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
        from qualang_tools.results import progress_counter
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

    def save_raw_results(self, *, now: datetime | None = None) -> Path:
        """Save ``results['ds_raw']`` and a profile snapshot."""
        if "ds_raw" not in self.results:
            raise CalibrationError("No raw dataset found in results['ds_raw'].")
        run_directory = self.saver.save_xarray(
            self.name,
            self.results["ds_raw"],
            profile_name=self.active_profile_name(),
            parameters=self.parameters,
            now=now,
        )
        self.namespace["calibration_run_directory"] = run_directory
        self.log(f"Raw calibration results saved to {run_directory}")
        return run_directory

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

    def analyse_data(self) -> None:
        """Backward-compatible alias for ``analyse``."""
        self.analyse()

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
