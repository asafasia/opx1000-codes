"""Measure readout-fidelity stability by repeatedly running IQ blobs."""

from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from calibration_io import CalibrationSaver, current_profile_name
from calibration_utils.iq_blobs import Parameters as IqBlobsParameters
from calibrations_v2 import CalibrationOptions
from quam_config import Quam, create_machine

IqBlobs = importlib.import_module("calibrations_v2.07_iq_blobs").IqBlobs


@dataclass
class IqBlobsStabilitySweepParameters:
    """Parameters for repeated IQ-blobs readout-stability measurements."""

    qubit: str = "q12"
    profile_name: str | None = None
    duration_seconds: float = 30 * 60
    interval_seconds: float = 0.0
    max_points: int | None = None
    connect_before_run: bool = True
    close_existing_qms: bool = True
    save_results: bool = True
    plot_results: bool = True
    iq_blobs: IqBlobsParameters = field(default_factory=IqBlobsParameters)
    iq_blobs_options: CalibrationOptions = field(
        default_factory=lambda: CalibrationOptions(
            save_raw_data=False,
            save_analysis_result=False,
            save_figures=False,
            analyse_data=True,
            plot_data=False,
            update_state=False,
            propose_profile_update=False,
            apply_profile_update=False,
        )
    )


class IqBlobsStabilitySweep:
    """Run IQ blobs repeatedly and track readout fidelity versus time."""

    name = "iq_blobs_stability_sweep"

    def __init__(
        self,
        parameters: IqBlobsStabilitySweepParameters | None = None,
        *,
        saver: CalibrationSaver | None = None,
        machine_factory: Callable[..., Quam] = create_machine,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.parameters = parameters or IqBlobsStabilitySweepParameters()
        self.saver = saver or CalibrationSaver()
        self.machine_factory = machine_factory
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.results: dict[str, Any] = {}
        self.run_directory: Path | None = None
        self.machine: Quam | None = None

    def _build_machine(self) -> Quam:
        profile_name = self.parameters.profile_name
        kwargs = {"qubit": self.parameters.qubit}
        if profile_name is not None:
            kwargs["profile_name"] = profile_name
        machine = self.machine_factory(**kwargs)
        if self.parameters.connect_before_run:
            machine.connect()
            if self.parameters.close_existing_qms and hasattr(machine, "qmm"):
                machine.qmm.close_all_qms()
        self.machine = machine
        return machine

    def run(self) -> dict[str, Any]:
        """Run repeated IQ-blobs measurements until duration or max-points ends."""
        start_time = self.time_fn()
        rows: list[dict[str, Any]] = []
        calibrations = []
        interrupted = False

        try:
            point_index = 0
            while self._should_run_next_point(start_time, point_index):
                point_index += 1
                elapsed = self.time_fn() - start_time
                print(
                    f"\n=== IQ-blobs stability point {point_index}: "
                    f"elapsed={elapsed / 60:.2f} min ==="
                )
                machine = self._build_machine()
                calibration = IqBlobs(
                    parameters=self.parameters.iq_blobs,
                    machine=machine,
                    options=self.parameters.iq_blobs_options,
                )
                status = calibration.run()
                calibrations.append(calibration)
                rows.extend(
                    self._rows_from_calibration(
                        point_index,
                        self.time_fn() - start_time,
                        calibration,
                        status,
                    )
                )
                self._sleep_before_next_point(start_time, point_index)
        except KeyboardInterrupt:
            interrupted = True
            self._finalize_results(start_time, rows, calibrations, interrupted=True)
            if rows and self.run_directory is not None:
                print(
                    "Interrupted IQ-blobs stability sweep; "
                    f"partial results saved to {self.run_directory}"
                )
            raise

        self._finalize_results(start_time, rows, calibrations, interrupted=interrupted)
        return self.results

    def _should_run_next_point(self, start_time: float, completed_points: int) -> bool:
        if self.parameters.max_points is not None and completed_points >= self.parameters.max_points:
            return False
        elapsed = self.time_fn() - start_time
        return completed_points == 0 or elapsed < self.parameters.duration_seconds

    def _sleep_before_next_point(self, start_time: float, completed_points: int) -> None:
        if self.parameters.interval_seconds <= 0:
            return
        if self.parameters.max_points is not None and completed_points >= self.parameters.max_points:
            return
        remaining = self.parameters.duration_seconds - (self.time_fn() - start_time)
        if remaining <= 0:
            return
        self.sleep_fn(min(self.parameters.interval_seconds, remaining))

    def _rows_from_calibration(
        self,
        point_index: int,
        elapsed_seconds: float,
        calibration: Any,
        status: Any,
    ) -> list[dict[str, Any]]:
        rows = []
        fit_results = calibration.results.get("fit_results", {})
        for qubit_name, fit in fit_results.items():
            rows.append(
                {
                    "point": int(point_index),
                    "elapsed_seconds": float(elapsed_seconds),
                    "qubit": qubit_name,
                    "success": bool(fit.get("success", False)),
                    "readout_fidelity": float(fit.get("readout_fidelity", np.nan)),
                    "readout_fidelity_std": float(fit.get("readout_fidelity_std", np.nan)),
                    "average_fidelity": float(fit.get("average_fidelity", np.nan)),
                    "average_fidelity_std": float(fit.get("average_fidelity_std", np.nan)),
                    "separation_to_width": float(fit.get("separation_to_width", np.nan)),
                    "center_separation": float(fit.get("center_separation", np.nan)),
                    "iw_angle": float(fit.get("iw_angle", np.nan)),
                    "ge_threshold": float(fit.get("ge_threshold", np.nan)),
                    "status_mode": status.mode,
                }
            )
        if not rows:
            rows.append(
                {
                    "point": int(point_index),
                    "elapsed_seconds": float(elapsed_seconds),
                    "qubit": self.parameters.qubit,
                    "success": False,
                    "readout_fidelity": np.nan,
                    "readout_fidelity_std": np.nan,
                    "average_fidelity": np.nan,
                    "average_fidelity_std": np.nan,
                    "separation_to_width": np.nan,
                    "center_separation": np.nan,
                    "iw_angle": np.nan,
                    "ge_threshold": np.nan,
                    "status_mode": status.mode,
                }
            )
        return rows

    def _finalize_results(
        self,
        start_time: float,
        rows: list[dict[str, Any]],
        calibrations: list[Any],
        *,
        interrupted: bool,
    ) -> None:
        if rows:
            self.results = self._aggregate_results(rows)
        else:
            self.results = {
                "point": np.asarray([], dtype=int),
                "elapsed_seconds": np.asarray([], dtype=float),
                "qubit": np.asarray([], dtype=str),
                "rows": [],
            }
        self.results["duration_seconds"] = float(self.time_fn() - start_time)
        self.results["target_duration_seconds"] = float(self.parameters.duration_seconds)
        self.results["interrupted"] = bool(interrupted)
        self.results["completed_points"] = int(len(np.unique(self.results["point"])))
        self.results["calibrations"] = calibrations
        if rows and self.parameters.plot_results:
            self.results["figures"] = self._plot_results()
        if rows and self.parameters.save_results:
            self.run_directory = self.save_results()

    def _aggregate_results(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        points = np.asarray(sorted({row["point"] for row in rows}), dtype=int)
        qubits = np.asarray(sorted({row["qubit"] for row in rows}), dtype=str)
        point_index = {point: index for index, point in enumerate(points)}
        qubit_index = {qubit: index for index, qubit in enumerate(qubits)}

        shape = (len(qubits), len(points))
        metrics = {
            "readout_fidelity": np.full(shape, np.nan),
            "readout_fidelity_std": np.full(shape, np.nan),
            "average_fidelity": np.full(shape, np.nan),
            "average_fidelity_std": np.full(shape, np.nan),
            "separation_to_width": np.full(shape, np.nan),
            "center_separation": np.full(shape, np.nan),
            "iw_angle": np.full(shape, np.nan),
            "ge_threshold": np.full(shape, np.nan),
        }
        success = np.zeros(shape, dtype=bool)
        elapsed_seconds = np.full(len(points), np.nan)

        for row in rows:
            qi = qubit_index[row["qubit"]]
            pi = point_index[row["point"]]
            elapsed_seconds[pi] = row["elapsed_seconds"]
            success[qi, pi] = row["success"]
            for name in metrics:
                metrics[name][qi, pi] = row[name]

        if len(points) > 1:
            stability_std = np.nanstd(metrics["readout_fidelity"], axis=1, ddof=1)
        else:
            stability_std = np.full(len(qubits), np.nan)
        stability_span = np.nanmax(metrics["readout_fidelity"], axis=1) - np.nanmin(
            metrics["readout_fidelity"],
            axis=1,
        )
        mean_readout_fidelity = np.nanmean(metrics["readout_fidelity"], axis=1)

        return {
            "point": points,
            "elapsed_seconds": elapsed_seconds,
            "qubit": qubits,
            **metrics,
            "success": success,
            "mean_readout_fidelity": mean_readout_fidelity,
            "readout_fidelity_stability_std": stability_std,
            "readout_fidelity_stability_span": stability_span,
            "rows": rows,
        }

    def _plot_results(self) -> dict[str, Any]:
        figures = {}
        elapsed_minutes = self.results["elapsed_seconds"] / 60
        for index, qubit in enumerate(self.results["qubit"]):
            figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            axes[0].errorbar(
                elapsed_minutes,
                self.results["readout_fidelity"][index],
                yerr=self.results["readout_fidelity_std"][index],
                marker="o",
                capsize=3,
                label="readout fidelity",
            )
            axes[0].axhline(
                self.results["mean_readout_fidelity"][index],
                color="tab:orange",
                linestyle="--",
                label="mean",
            )
            axes[0].set_ylabel("Fidelity (%)")
            axes[0].set_title(f"{qubit} readout-fidelity stability")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(
                elapsed_minutes,
                self.results["separation_to_width"][index],
                marker="o",
                label="separation / width",
            )
            axes[1].set_xlabel("Elapsed time (min)")
            axes[1].set_ylabel("Separation / width")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            figure.tight_layout()
            figures[f"{qubit}_iq_blobs_stability"] = figure
        plt.show()
        return figures

    def save_results(self) -> Path:
        """Save aggregate stability arrays, metadata, and figures."""
        run_directory = self.saver.save(
            self.name,
            sweep={
                "point": self.results["point"],
                "elapsed_seconds": self.results["elapsed_seconds"],
                "qubit": self.results["qubit"],
            },
            results={
                "readout_fidelity": self.results["readout_fidelity"],
                "readout_fidelity_std": self.results["readout_fidelity_std"],
                "average_fidelity": self.results["average_fidelity"],
                "average_fidelity_std": self.results["average_fidelity_std"],
                "separation_to_width": self.results["separation_to_width"],
                "center_separation": self.results["center_separation"],
                "iw_angle": self.results["iw_angle"],
                "ge_threshold": self.results["ge_threshold"],
                "success": self.results["success"],
                "mean_readout_fidelity": self.results["mean_readout_fidelity"],
                "readout_fidelity_stability_std": self.results[
                    "readout_fidelity_stability_std"
                ],
                "readout_fidelity_stability_span": self.results[
                    "readout_fidelity_stability_span"
                ],
                "duration_seconds": np.asarray(self.results["duration_seconds"]),
                "target_duration_seconds": np.asarray(
                    self.results["target_duration_seconds"]
                ),
                "interrupted": np.asarray(self.results["interrupted"]),
                "completed_points": np.asarray(self.results["completed_points"]),
            },
            profile_name=self.parameters.profile_name or current_profile_name(),
            parameters=self.parameters,
        )
        summary = {
            "interrupted": bool(self.results["interrupted"]),
            "completed_points": int(self.results["completed_points"]),
            "duration_seconds": float(self.results["duration_seconds"]),
            "target_duration_seconds": float(self.results["target_duration_seconds"]),
            "qubit": self.results["qubit"].tolist(),
            "point": self.results["point"].tolist(),
            "elapsed_seconds": self.results["elapsed_seconds"].tolist(),
            "mean_readout_fidelity": self.results["mean_readout_fidelity"].tolist(),
            "readout_fidelity_stability_std": self.results[
                "readout_fidelity_stability_std"
            ].tolist(),
            "readout_fidelity_stability_span": self.results[
                "readout_fidelity_stability_span"
            ].tolist(),
            "rows": self.results["rows"],
        }
        with (run_directory / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, allow_nan=True)
            file.write("\n")
        if self.results.get("figures"):
            self.saver.save_figures(run_directory, self.results["figures"])
        print(f"IQ-blobs stability sweep saved to {run_directory}")
        return run_directory


def default_parameters() -> IqBlobsStabilitySweepParameters:
    """Return the default 30-minute readout-stability sweep setup."""
    parameters = IqBlobsStabilitySweepParameters()
    parameters.duration_seconds = 30 * 60
    parameters.interval_seconds = 0.0
    parameters.iq_blobs.reset_type = "thermal"
    parameters.iq_blobs.num_shots = 10000
    parameters.iq_blobs.states = ["g", "e"]
    parameters.iq_blobs.qubit_operation = "x180_const"
    parameters.iq_blobs.pi_repetitions = 1
    return parameters


if __name__ == "__main__":
    sweep_parameters = default_parameters()
    sweep_parameters.qubit = "q12"
    sweep = IqBlobsStabilitySweep(sweep_parameters)
    sweep.run()
