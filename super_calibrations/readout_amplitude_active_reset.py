"""Sweep readout amplitude and compare IQ-blob fidelity with and without active reset.

This is intentionally an orchestration calibration: each readout amplitude runs
thermal IQ blobs first, uses that fitted discriminator for active reset, then
runs active-reset IQ blobs at the same amplitude.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import io
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)
from quam_config import Quam, create_machine
from calibration_io import CalibrationSaver, current_profile_name
from calibrations_v2.base import BaseCalibration, CalibrationOptions
from utils.plotting_settings import FIGURE_SIZE

iq_blobs_module = importlib.import_module("calibrations_v2.07_iq_blobs")
IqBlobs = iq_blobs_module.IqBlobs
IqBlobParameters = iq_blobs_module.Parameters


description = """
        READOUT AMPLITUDE ACTIVE-RESET COMPARISON
For each readout-amplitude prefactor, this super calibration first acquires
ordinary thermal-reset IQ blobs and fits the GE discriminator. It then installs
that fitted threshold in the in-memory machine and acquires IQ blobs again with
active reset enabled.

The resulting dashboard compares the assignment fidelity obtained without
active reset and with active reset as a function of readout amplitude. This is
useful when the best amplitude for clean discrimination is not the same as the
best amplitude for feedback-assisted reset.
"""


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 20000
    """Number of shots per IQ-blob acquisition."""
    start_amp: float = 0.5
    """Start readout-amplitude prefactor."""
    end_amp: float = 2.5
    """End readout-amplitude prefactor."""
    num_amps: int = 9
    """Number of readout-amplitude prefactors to sweep."""
    operation: Literal["readout", "readout_QND"] = "readout"
    """Readout operation to sweep."""
    qubit_operation: Literal["saturation", "x180_const"] = "x180_const"
    """Qubit operation used for the excited-state IQ blob."""
    qubit_amplitude_factor: float = 1.0
    """Amplitude factor applied to the selected qubit operation."""
    pi_repetitions: int = 1
    """Number of x180_const pulses used for excited-state preparation."""
    xy_to_readout_delay_in_ns: int = 100
    """Delay between the prepared-state XY pulse and readout."""
    save_inner_raw_data: bool = False
    """Save every inner IQ-blob run in addition to the aggregate super-calibration run."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass


@dataclass
class _OriginalReadoutState:
    amplitude: float
    threshold: float | None
    rus_exit_threshold: float | None
    integration_weights_angle: float


class _SharedConnectionMachine:
    """Forward machine access while reusing an existing QuantumMachinesManager."""

    def __init__(self, machine: Quam, qmm: Any) -> None:
        self._machine = machine
        self._qmm = qmm

    def connect(self):
        return self._qmm

    def __getattr__(self, name: str):
        return getattr(self._machine, name)


class ReadoutAmplitudeActiveReset(BaseCalibration[Parameters, Quam]):
    """Outer-loop calibration for readout amplitude vs active-reset fidelity."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="super_readout_amplitude_active_reset",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )
        self._original_readout_state: dict[str, _OriginalReadoutState] = {}
        self._shared_machine: _SharedConnectionMachine | None = None

    def create_qua_program(self) -> None:
        """This calibration builds QUA programs inside each inner IQ-blob run."""
        return None

    def run(self):
        if self.parameters.load_data_id is not None:
            self.load_data()
            self.analyse_data()
            if self.options.plot_data:
                self.plot_data()
            if self.options.save_figures:
                self.save_figures()
            return None

        self.namespace["qubits"] = self.get_qubits()
        self._remember_original_readout_state()
        try:
            self.execute_outer_loop()
            if self.options.save_raw_data:
                self.save_raw_results()
            if self.options.analyse_data:
                self.analyse_data()
            if self.options.plot_data:
                self.plot_data()
            if self.options.save_figures:
                self.save_figures()
            if self.options.update_state:
                self.update_state()
            if self.options.propose_profile_update:
                self.propose_profile_update(apply=self.options.apply_profile_update)
        finally:
            self.cleanup()
        return None

    def execute_outer_loop(self) -> None:
        amp_prefactors = np.linspace(
            self.parameters.start_amp,
            self.parameters.end_amp,
            self.parameters.num_amps,
        )
        total_steps = len(amp_prefactors) * 2
        completed_steps = 0
        thermal_datasets = []
        active_datasets = []
        fit_rows = []
        fit_results: dict[str, dict[str, dict[str, Any]]] = {}
        qmm = self.machine.connect()
        self._shared_machine = _SharedConnectionMachine(self.machine, qmm)

        for amp_prefactor in amp_prefactors:
            self._show_outer_progress(
                completed_steps,
                total_steps,
                f"amp prefactor {amp_prefactor:.6g}: thermal IQ blobs",
            )
            self._set_readout_amplitude_prefactor(float(amp_prefactor))
            self._restore_discriminators()

            thermal = self._run_inner_iq_blobs("thermal")
            completed_steps += 1
            self._show_outer_progress(
                completed_steps,
                total_steps,
                f"amp prefactor {amp_prefactor:.6g}: active-reset IQ blobs",
            )
            self._install_thermal_discriminator(thermal.results["fit_results"])
            active = self._run_inner_iq_blobs("active")
            completed_steps += 1
            self._show_outer_progress(
                completed_steps,
                total_steps,
                f"amp prefactor {amp_prefactor:.6g}: done",
            )

            thermal_datasets.append(
                self._tag_dataset(thermal.results["ds_raw"], amp_prefactor, "thermal")
            )
            active_datasets.append(
                self._tag_dataset(active.results["ds_raw"], amp_prefactor, "active")
            )

            for reset_mode, child in (("thermal", thermal), ("active", active)):
                for qubit_name, result in child.results["fit_results"].items():
                    iq_stats = self._iq_separation_stats(
                        child.results["ds_raw"], qubit_name
                    )
                    fit_rows.append(
                        {
                            "qubit": qubit_name,
                            "reset_mode": reset_mode,
                            "amp_prefactor": float(amp_prefactor),
                            "readout_amplitude": self._readout_amplitude_for_qubit(
                                qubit_name
                            ),
                            "readout_fidelity": float(result["readout_fidelity"]),
                            "separation_to_width": float(result["separation_to_width"]),
                            "iq_center_separation": iq_stats["center_separation"],
                            "iq_center_separation_std": iq_stats[
                                "center_separation_std"
                            ],
                            "success": bool(result["success"]),
                        }
                    )
                    fit_results.setdefault(qubit_name, {}).setdefault(reset_mode, {})[
                        f"{amp_prefactor:.12g}"
                    ] = result

        self.results["ds_raw"] = xr.combine_by_coords(
            thermal_datasets + active_datasets,
            combine_attrs="drop_conflicts",
        )
        self.results["fit_table"] = fit_rows
        self.results["fit_results_by_amplitude"] = fit_results
        print()

    def _run_inner_iq_blobs(self, reset_type: Literal["thermal", "active"]):
        parameters = self._inner_parameters(reset_type)
        options = CalibrationOptions(
            save_raw_data=self.parameters.save_inner_raw_data,
            save_analysis_result=False,
            save_figures=False,
            analyse_data=True,
            plot_data=False,
            update_state=False,
            propose_profile_update=False,
            apply_profile_update=False,
        )
        calibration = IqBlobs(
            parameters=parameters,
            options=options,
            logger=lambda _: None,
            machine=self._shared_machine or self.machine,
        )
        with self._quiet_inner_run():
            calibration.run()
        return calibration

    @contextlib.contextmanager
    def _quiet_inner_run(self):
        """Suppress SDK logs and printed inner progress while preserving exceptions."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                yield
        except Exception:
            captured = "\n".join(
                part.strip()
                for part in (stdout.getvalue(), stderr.getvalue())
                if part.strip()
            )
            if captured:
                self.log(
                    f"Suppressed inner-run output before failure:\n{captured[-4000:]}"
                )
            raise
        finally:
            logging.disable(previous_disable_level)

    def _show_outer_progress(
        self, completed_steps: int, total_steps: int, label: str
    ) -> None:
        fraction = completed_steps / max(total_steps, 1)
        width = 28
        filled = int(round(width * fraction))
        bar = "#" * filled + "." * (width - filled)
        print(
            f"Super IQ amplitude [{bar}] {100 * fraction:5.1f}% "
            f"({completed_steps}/{total_steps}) {label}",
            end="\r",
            flush=True,
        )

    def _inner_parameters(self, reset_type: Literal["thermal", "active"]):
        parameters = IqBlobParameters()
        for name in (
            "qubits",
            "num_shots",
            "operation",
            "qubit_operation",
            "qubit_amplitude_factor",
            "pi_repetitions",
            "xy_to_readout_delay_in_ns",
            "simulate",
            "timeout",
        ):
            if hasattr(self.parameters, name):
                setattr(parameters, name, copy.deepcopy(getattr(self.parameters, name)))
        parameters.states = ["g", "e"]
        parameters.reset_type = reset_type
        parameters.load_data_id = None
        return parameters

    def _tag_dataset(
        self,
        dataset: xr.Dataset,
        amp_prefactor: float,
        reset_mode: Literal["thermal", "active"],
    ) -> xr.Dataset:
        tagged = dataset.expand_dims(
            reset_mode=[reset_mode],
            amp_prefactor=[float(amp_prefactor)],
        )
        amplitudes = [
            self._readout_amplitude_for_qubit(str(qubit_name))
            for qubit_name in tagged.qubit.values
        ]
        return tagged.assign_coords(
            readout_amplitude=(
                ["qubit", "amp_prefactor"],
                np.asarray(amplitudes, dtype=float)[:, np.newaxis],
            )
        )

    def _remember_original_readout_state(self) -> None:
        self._original_readout_state = {}
        for qubit in self.namespace["qubits"]:
            operation = qubit.resonator.operations[self.parameters.operation]
            self._original_readout_state[qubit.name] = _OriginalReadoutState(
                amplitude=float(operation.amplitude),
                threshold=(
                    None if operation.threshold is None else float(operation.threshold)
                ),
                rus_exit_threshold=(
                    None
                    if operation.rus_exit_threshold is None
                    else float(operation.rus_exit_threshold)
                ),
                integration_weights_angle=float(operation.integration_weights_angle),
            )

    def _restore_discriminators(self) -> None:
        for qubit in self.namespace["qubits"]:
            original = self._original_readout_state[qubit.name]
            operation = qubit.resonator.operations[self.parameters.operation]
            operation.threshold = original.threshold
            operation.rus_exit_threshold = original.rus_exit_threshold
            operation.integration_weights_angle = original.integration_weights_angle

    def _set_readout_amplitude_prefactor(self, amp_prefactor: float) -> None:
        for qubit in self.namespace["qubits"]:
            original = self._original_readout_state[qubit.name]
            operation = qubit.resonator.operations[self.parameters.operation]
            operation.amplitude = original.amplitude * amp_prefactor

    def _install_thermal_discriminator(
        self, thermal_fit_results: dict[str, dict[str, Any]]
    ) -> None:
        for qubit in self.namespace["qubits"]:
            fit_result = thermal_fit_results[qubit.name]
            operation = qubit.resonator.operations[self.parameters.operation]
            operation.integration_weights_angle -= float(fit_result["iw_angle"])
            operation.threshold = (
                float(fit_result["ge_threshold"]) * operation.length / 2**12
            )
            operation.rus_exit_threshold = (
                float(fit_result["rus_threshold"]) * operation.length / 2**12
            )
            self.log(
                f"{qubit.name}: active-reset discriminator from thermal blobs at "
                f"{operation.amplitude:.6g} V, threshold={operation.threshold:.6g}"
            )

    def _readout_amplitude_for_qubit(self, qubit_name: str) -> float:
        qubit = self.machine.qubits[qubit_name]
        return float(qubit.resonator.operations[self.parameters.operation].amplitude)

    def _iq_separation_stats(
        self, dataset: xr.Dataset, qubit_name: str
    ) -> dict[str, float]:
        """Return IQ-center separation and projected shot-noise standard error."""
        selected = dataset.sel(qubit=qubit_name)
        ground_i = selected.Ig
        ground_q = selected.Qg
        excited_i = selected.Ie
        excited_q = selected.Qe
        delta_i = float(excited_i.mean() - ground_i.mean())
        delta_q = float(excited_q.mean() - ground_q.mean())
        separation = float(np.hypot(delta_i, delta_q))
        run_count = max(int(selected.sizes.get("n_runs", 1)), 1)
        if not np.isfinite(separation) or separation == 0:
            return {
                "center_separation": separation,
                "center_separation_std": np.nan,
            }

        unit_i = delta_i / separation
        unit_q = delta_q / separation
        variance = (
            unit_i**2 * (float(ground_i.var()) + float(excited_i.var()))
            + unit_q**2 * (float(ground_q.var()) + float(excited_q.var()))
        ) / run_count
        return {
            "center_separation": separation,
            "center_separation_std": float(np.sqrt(max(variance, 0.0))),
        }

    def save_raw_results(self, *, now=None):
        run_directory = CalibrationSaver().save_xarray(
            self.name,
            self.results["ds_raw"],
            profile_name=current_profile_name(),
            parameters=self.parameters,
            now=now,
        )
        self.namespace["calibration_run_directory"] = run_directory
        fit_path = Path(run_directory) / "fit_results_by_amplitude.json"
        with fit_path.open("w", encoding="utf-8") as file:
            json.dump(self.results["fit_results_by_amplitude"], file, indent=2)
            file.write("\n")
        self.log(f"Super-calibration results saved to {run_directory}")
        return run_directory

    def load_data(self) -> None:
        load_data_id = self.parameters.load_data_id
        self.results["ds_raw"] = self.load_saved_run(load_data_id)
        self.parameters.load_data_id = load_data_id
        self.namespace["qubits"] = self.get_qubits()

    def analyse_data(self) -> None:
        table = self.results.get("fit_table")
        if not self._original_readout_state:
            self._remember_original_readout_state()
        if table is None:
            self.results["ds_fit"] = self._fit_dataset_from_raw(self.results["ds_raw"])
        else:
            self.results["ds_fit"] = self._fit_dataset_from_table(table)
        self._set_outcomes_and_best_points()

    def _fit_dataset_from_table(self, rows: list[dict[str, Any]]) -> xr.Dataset:
        qubits = sorted({row["qubit"] for row in rows})
        reset_modes = ["thermal", "active"]
        amp_prefactors = np.asarray(
            sorted({float(row["amp_prefactor"]) for row in rows}),
            dtype=float,
        )
        coords = {
            "qubit": qubits,
            "reset_mode": reset_modes,
            "amp_prefactor": amp_prefactors,
        }
        shape = (len(qubits), len(reset_modes), len(amp_prefactors))
        data = {
            "readout_fidelity": np.full(shape, np.nan),
            "separation_to_width": np.full(shape, np.nan),
            "iq_center_separation": np.full(shape, np.nan),
            "iq_center_separation_std": np.full(shape, np.nan),
            "readout_amplitude": np.full(shape, np.nan),
            "success": np.zeros(shape, dtype=bool),
        }
        qubit_index = {name: index for index, name in enumerate(qubits)}
        mode_index = {name: index for index, name in enumerate(reset_modes)}
        amp_index = {value: index for index, value in enumerate(amp_prefactors)}
        for row in rows:
            index = (
                qubit_index[row["qubit"]],
                mode_index[row["reset_mode"]],
                amp_index[float(row["amp_prefactor"])],
            )
            for name in data:
                data[name][index] = row[name]
        return xr.Dataset(
            {
                name: (("qubit", "reset_mode", "amp_prefactor"), values)
                for name, values in data.items()
            },
            coords=coords,
        )

    def _fit_dataset_from_raw(self, ds: xr.Dataset) -> xr.Dataset:
        if "readout_fidelity" not in ds:
            raise ValueError(
                "Loaded aggregate data does not contain fit_table; rerun from raw inner IQ blobs to reanalyse."
            )
        return ds

    def _set_outcomes_and_best_points(self) -> None:
        fit = self.results["ds_fit"]
        best_points = {}
        outcomes = {}
        for qubit_name in fit.qubit.values:
            selected = fit.sel(qubit=qubit_name)
            best_points[str(qubit_name)] = {
                "current": {
                    "readout_amplitude": self._current_readout_amplitude(
                        str(qubit_name)
                    )
                }
            }
            outcomes[str(qubit_name)] = "successful"
            for reset_mode in fit.reset_mode.values:
                trace = selected.readout_fidelity.sel(reset_mode=reset_mode)
                if np.all(~np.isfinite(trace)):
                    outcomes[str(qubit_name)] = "failed"
                    continue
                best_index = int(trace.argmax(dim="amp_prefactor"))
                best_amp_prefactor = float(fit.amp_prefactor.values[best_index])
                best_points[str(qubit_name)][str(reset_mode)] = {
                    "amp_prefactor": best_amp_prefactor,
                    "readout_amplitude": float(
                        selected.readout_amplitude.sel(
                            reset_mode=reset_mode,
                            amp_prefactor=best_amp_prefactor,
                        )
                    ),
                    "readout_fidelity": float(
                        trace.sel(amp_prefactor=best_amp_prefactor)
                    ),
                }
        self.results["best_points"] = best_points
        self.outcomes = outcomes

    def _current_readout_amplitude(self, qubit_name: str) -> float:
        if qubit_name in self._original_readout_state:
            return self._original_readout_state[qubit_name].amplitude
        return self._readout_amplitude_for_qubit(qubit_name)

    def plot_data(self) -> None:
        fit = self.results["ds_fit"]
        figures = {}
        for qubit_name in fit.qubit.values:
            selected = fit.sel(qubit=qubit_name)
            figure, axes = plt.subplots(
                3,
                1,
                figsize=FIGURE_SIZE,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1.3, 1.1], "hspace": 0.08},
            )
            fidelity_axis, separation_axis, uncertainty_axis = axes
            current_amplitude = self.results["best_points"][str(qubit_name)]["current"][
                "readout_amplitude"
            ]
            for reset_mode, color, marker, label in (
                ("thermal", "tab:blue", "o", "No active reset"),
                ("active", "tab:green", "s", "Active reset"),
            ):
                trace = selected.sel(reset_mode=reset_mode)
                x = 1e3 * trace.readout_amplitude.values
                y = trace.readout_fidelity.values
                fidelity_axis.plot(
                    x,
                    y,
                    marker=marker,
                    linewidth=1.9,
                    markersize=5,
                    color=color,
                    label=label,
                )
                best = self.results["best_points"][str(qubit_name)][reset_mode]
                fidelity_axis.plot(
                    1e3 * best["readout_amplitude"],
                    best["readout_fidelity"],
                    marker="*",
                    markersize=13,
                    color=color,
                    markeredgecolor="0.15",
                )
                separation_axis.errorbar(
                    x,
                    1e3 * trace.iq_center_separation.values,
                    yerr=1e3 * trace.iq_center_separation_std.values,
                    marker=marker,
                    markersize=4,
                    linewidth=1.4,
                    capsize=3,
                    color=color,
                    alpha=0.85,
                    label=label,
                )
                uncertainty_axis.plot(
                    x,
                    1e3 * trace.iq_center_separation_std.values,
                    marker=marker,
                    markersize=4,
                    linewidth=1.4,
                    color=color,
                    alpha=0.85,
                    label=label,
                )
            fidelity_axis.axvline(
                1e3 * current_amplitude,
                color="0.2",
                linestyle=":",
                linewidth=1.5,
                label=f"Current amplitude ({1e3 * current_amplitude:.3g} mV)",
            )
            for lower_axis in (separation_axis, uncertainty_axis):
                lower_axis.axvline(
                    1e3 * current_amplitude,
                    color="0.2",
                    linestyle=":",
                    linewidth=1.2,
                )

            fidelity_axis.set_title(
                f"{qubit_name}: readout amplitude vs IQ-blob fidelity"
            )
            fidelity_axis.set_ylabel("Assignment fidelity [%]")
            fidelity_axis.set_ylim(0, 100)
            fidelity_axis.grid(alpha=0.25)
            fidelity_axis.spines[["top", "right"]].set_visible(False)
            fidelity_axis.legend()
            separation_axis.set_ylabel("IQ sep. [mV]")
            separation_axis.grid(alpha=0.25)
            separation_axis.spines[["top", "right"]].set_visible(False)
            uncertainty_axis.set_xlabel("Readout amplitude [mV]")
            uncertainty_axis.set_ylabel("IQ std. [mV]")
            uncertainty_axis.grid(alpha=0.25)
            uncertainty_axis.spines[["top", "right"]].set_visible(False)
            summary = self._format_best_point_summary(str(qubit_name))
            fidelity_axis.text(
                0.02,
                0.03,
                summary,
                transform=fidelity_axis.transAxes,
                va="bottom",
                ha="left",
                fontsize="small",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "fc": "white",
                    "ec": "0.8",
                    "alpha": 0.9,
                },
            )
            figure.tight_layout()
            figures[str(qubit_name)] = figure
        self.results["figures"] = figures
        plt.show()

    def _format_best_point_summary(self, qubit_name: str) -> str:
        best = self.results["best_points"][qubit_name]
        return "\n".join(
            [
                f"current amp: {1e3 * best['current']['readout_amplitude']:.3g} mV",
                f"best no reset: {1e3 * best['thermal']['readout_amplitude']:.3g} mV, "
                f"{best['thermal']['readout_fidelity']:.1f}%",
                f"best active reset: {1e3 * best['active']['readout_amplitude']:.3g} mV, "
                f"{best['active']['readout_fidelity']:.1f}%",
            ]
        )

    def update_state(self) -> None:
        """Leave the machine restored; this calibration is comparative by default."""

    def propose_profile_update(self, *, apply: bool = True) -> bool:
        updates = {}
        for qubit_name, best in self.results.get("best_points", {}).items():
            if "active" not in best:
                continue
            updates[f"pulses.json.pulses.{qubit_name}.readout.amplitude"] = float(
                best["active"]["readout_amplitude"]
            )
            updates[
                f"metrics.json.qubits.{qubit_name}.readout.fidelity_percent.active"
            ] = float(best["active"]["readout_fidelity"])

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

    def cleanup(self) -> None:
        if not self._original_readout_state:
            return
        for qubit in self.namespace.get("qubits", []):
            original = self._original_readout_state[qubit.name]
            operation = qubit.resonator.operations[self.parameters.operation]
            operation.amplitude = original.amplitude
            operation.threshold = original.threshold
            operation.rus_exit_threshold = original.rus_exit_threshold
            operation.integration_weights_angle = original.integration_weights_angle


if __name__ == "__main__":
    parameters = Parameters()
    # parameters.qubits = ["q1"]
    parameters.num_shots = 5000
    parameters.start_amp = 0
    parameters.end_amp = 2.0
    parameters.num_amps = 10

    options = CalibrationOptions(update_state=False, propose_profile_update=True)
    calibration = ReadoutAmplitudeActiveReset(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q2"),
    )
    calibration.run()
