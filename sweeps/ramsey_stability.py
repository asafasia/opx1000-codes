"""Timed T2* stability acquisition. See docs/calibrations/ramsey_stability.md.

Run with ``python -m sweeps.ramsey_stability --qubit q3 --duration-hours 24``.
Use ``--dry-run`` to inspect settings without connecting to any hardware.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import html
import importlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sweeps.stability_analysis import point_diagnostics, summarize

ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write(path: Path, content: str | bytes) -> None:
    """Publish a complete file; flush before replacing the previous version."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(
                content.encode("utf-8") if isinstance(content, str) else content
            )
            stream.flush()
            os.fsync(stream.fileno())
        # Windows viewers/scanners may briefly hold the old report open.
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    atomic_write(path, json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def save_dataset(directory: Path, name: str, dataset) -> None:
    """Store arrays without pickle, plus dimensions/attributes for reconstruction."""
    arrays = {key: variable.values for key, variable in dataset.variables.items()}
    for key, array in arrays.items():
        if array.dtype.hasobject:
            if all(isinstance(item, str) for item in array.flat):
                arrays[key] = array.astype(str)
            else:
                raise ValueError(f"Cannot safely save object array {key}")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write(directory / f"{name}.npz", buffer.getvalue())
    write_json(
        directory / f"{name}.json",
        {
            "coords": list(dataset.coords),
            "attrs": dataset.attrs,
            "variables": {
                key: {"dims": list(v.dims), "attrs": v.attrs}
                for key, v in dataset.variables.items()
            },
        },
    )


@dataclass(frozen=True)
class StabilitySettings:
    qubit: str
    duration_seconds: float = 24 * 3600
    interval_seconds: float = 0.0
    max_points: int | None = None
    rolling_window: int = 20
    jump_sigma: float = 5.0
    max_relative_error: float = 0.5
    max_consecutive_failures: int = 5

    def __post_init__(self):
        for name in ("duration_seconds", "jump_sigma", "max_relative_error"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.interval_seconds) or self.interval_seconds < 0:
            raise ValueError("interval_seconds must be finite and nonnegative")
        for name, minimum in (
            ("rolling_window", 5),
            ("max_consecutive_failures", 1),
            ("max_points", 1),
        ):
            value = getattr(self, name)
            if name == "max_points" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not self.qubit.strip():
            raise ValueError("qubit must be supplied")


def finite_number(value, multiplier=1.0):
    try:
        result = float(value) * multiplier
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


class RamseyStabilityRun:
    """Orchestrate fresh BaseCalibration instances, retaining only scalar history.

    The factory takes a point directory. Production uses PersistentRamsey below;
    tests inject a fake calibration and monotonic clock, never QOP hardware.
    """

    def __init__(
        self,
        settings: StabilitySettings,
        run_directory: Path,
        calibration_factory: Callable,
        *,
        time_fn=time.monotonic,
        sleep_fn=time.sleep,
        utc_fn=utc_now,
        render_reports=True,
    ):
        self.settings = settings
        self.run_directory = Path(run_directory)
        self.calibration_factory = calibration_factory
        self.time_fn, self.sleep_fn, self.utc_fn = time_fn, sleep_fn, utc_fn
        self.render_reports = render_reports
        self.rows: list[dict] = []
        self.status = "created"
        self.error = None
        self.started = 0.0

    def _publish(self):
        summary = summarize(self.rows)
        summary.update(
            status=self.status,
            error=self.error,
            updated_utc=self.utc_fn(),
            elapsed_seconds=self.time_fn() - self.started,
            target_duration_seconds=self.settings.duration_seconds,
            qubit=self.settings.qubit,
        )
        write_json(self.run_directory / "summary.json", summary)
        if self.rows:
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
            atomic_write(self.run_directory / "points.csv", stream.getvalue())
        if self.render_reports:
            render_report(self.run_directory, self.rows, summary)
        return summary

    def _record(self, point, directory, began, ended, began_utc, calibration, error):
        fit = (
            calibration.results.get("fit_results", {}).get(self.settings.qubit, {})
            if calibration
            else {}
        )
        namespace = getattr(calibration, "namespace", {})
        measured_start = namespace.get("measurement_started_monotonic", began)
        measured_end = namespace.get("measurement_finished_monotonic", ended)
        t2 = finite_number(
            fit.get("decay"), 1e6
        )  # Ramsey FitParameters.decay is seconds.
        uncertainty = finite_number(fit.get("decay_error"), 1e6)
        offset = finite_number(fit.get("freq_offset"))
        reasons = []
        if error:
            reasons.append(error)
        if not fit.get("success", False):
            reasons.append("fit_failed_or_missing")
        if t2 is None or t2 <= 0:
            reasons.append("invalid_t2")
        if uncertainty is None or uncertainty < 0:
            reasons.append("invalid_uncertainty")
        elif (
            t2 is not None
            and t2 > 0
            and uncertainty / t2 > self.settings.max_relative_error
        ):
            reasons.append("relative_uncertainty_exceeds_limit")
        if offset is None:
            reasons.append("invalid_frequency_offset")
        row = dict(
            point=point,
            qubit=self.settings.qubit,
            started_utc=began_utc,
            finished_utc=self.utc_fn(),
            measurement_started_utc=namespace.get("measurement_started_utc"),
            measurement_finished_utc=namespace.get("measurement_finished_utc"),
            elapsed_seconds=(measured_start + measured_end) / 2 - self.started,
            measurement_duration_seconds=measured_end - measured_start,
            point_duration_seconds=ended - began,
            t2_us=t2,
            t2_error_us=uncertainty,
            frequency_offset_hz=offset,
            fit_success=bool(fit.get("success", False)),
            accepted=not reasons,
            rejection_reason="; ".join(reasons),
            point_directory=str(directory.relative_to(self.run_directory)),
        )
        row.update(
            point_diagnostics(
                self.rows, row, self.settings.rolling_window, self.settings.jump_sigma
            )
        )
        # Independent point files survive a torn final journal line or stale report.
        write_json(directory / "point.json", row)
        with (self.run_directory / "points.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)
        self._publish()
        print(
            f"Point {point}: T2*={t2} +/- {uncertainty} us; "
            f"{'accepted' if row['accepted'] else row['rejection_reason']}",
            flush=True,
        )

    def run(self):
        # Refuse accidental reuse/overwrites. Recovery data remains readable on disk.
        (self.run_directory / "points").mkdir(parents=True, exist_ok=False)
        self.started = self.time_fn()
        self.status = "running"
        write_json(
            self.run_directory / "stability_settings.json", asdict(self.settings)
        )
        self._publish()
        point, consecutive_failures = 0, 0
        next_start = self.started
        try:
            while self.time_fn() < self.started + self.settings.duration_seconds:
                if (
                    self.settings.max_points is not None
                    and point >= self.settings.max_points
                ):
                    break
                delay = (
                    min(next_start, self.started + self.settings.duration_seconds)
                    - self.time_fn()
                )
                if delay > 0:
                    self.sleep_fn(
                        min(delay, 1.0)
                    )  # responsive Ctrl+C during long intervals
                    continue
                point += 1
                directory = self.run_directory / "points" / f"{point:06d}"
                directory.mkdir()
                began, began_utc = self.time_fn(), self.utc_fn()
                write_json(
                    directory / "started.json",
                    {
                        "point": point,
                        "started_utc": began_utc,
                        "elapsed_seconds": began - self.started,
                    },
                )
                calibration = None
                error = None
                try:
                    calibration = self.calibration_factory(directory)
                    calibration.run()
                except (Exception, KeyboardInterrupt) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    self._record(
                        point,
                        directory,
                        began,
                        self.time_fn(),
                        began_utc,
                        calibration,
                        error,
                    )
                consecutive_failures = (
                    0 if self.rows[-1]["accepted"] else consecutive_failures + 1
                )
                if consecutive_failures >= self.settings.max_consecutive_failures:
                    raise RuntimeError(
                        f"Stopped after {consecutive_failures} consecutive rejected fits"
                    )
                # Fixed start grid; skip missed slots, never queue catch-up acquisitions.
                if self.settings.interval_seconds:
                    elapsed = self.time_fn() - self.started
                    previous_slot = round(
                        (next_start - self.started) / self.settings.interval_seconds
                    )
                    slot = max(
                        previous_slot + 1,
                        math.ceil(elapsed / self.settings.interval_seconds),
                    )
                    next_start = self.started + slot * self.settings.interval_seconds
                else:
                    next_start = self.time_fn()
            self.status = "completed"
        except KeyboardInterrupt:
            self.status = "interrupted"
            raise
        except Exception as exc:
            self.status, self.error = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._publish()
        return summarize(self.rows)


def render_report(directory: Path, rows: list[dict], summary: dict) -> None:
    """Use an Agg canvas directly: no blocking windows or retained figures."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    hours = np.asarray([r["elapsed_seconds"] / 3600 for r in rows])
    t2 = [r["t2_us"] if r["accepted"] else np.nan for r in rows]
    errors = [r["t2_error_us"] if r["accepted"] else np.nan for r in rows]
    axes[0, 0].errorbar(
        hours, t2, yerr=errors, fmt="o-", ms=3, capsize=2, label="T2* +/- fit error"
    )
    jumps = [r for r in rows if r["jump_candidate"]]
    axes[0, 0].scatter(
        [r["elapsed_seconds"] / 3600 for r in jumps],
        [r["t2_us"] for r in jumps],
        color="red",
        marker="x",
        s=65,
        label="jump candidate",
    )
    rejected = [r["elapsed_seconds"] / 3600 for r in rows if not r["accepted"]]
    axes[0, 0].plot(
        rejected,
        [0.03] * len(rejected),
        "rx",
        transform=axes[0, 0].get_xaxis_transform(),
        label="rejected",
    )
    axes[0, 0].set(ylabel="T2* (us)", xlabel="Elapsed time (h)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(hours, [r["rolling_variance_us2"] for r in rows], "o-", ms=3)
    axes[0, 1].set(ylabel="Rolling sample variance (us²)", xlabel="Elapsed time (h)")
    axes[1, 0].plot(
        hours,
        [r["frequency_offset_hz"] if r["accepted"] else np.nan for r in rows],
        "o-",
        ms=3,
    )
    axes[1, 0].set(ylabel="Ramsey frequency offset (Hz)", xlabel="Elapsed time (h)")
    allan = summary["allan"]
    if allan["tau_seconds"]:
        axes[1, 1].plot(allan["tau_seconds"], allan["deviation_us"], "o-")
        axes[1, 1].set_xscale("log")
        if any(v > 0 for v in allan["deviation_us"]):
            axes[1, 1].set_yscale("log")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            allan["reason"],
            ha="center",
            va="center",
            wrap=True,
            transform=axes[1, 1].transAxes,
            fontsize=9,
        )
    axes[1, 1].set(
        ylabel="Overlapping Allan deviation (us)", xlabel="Averaging time (s)"
    )
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    figure.suptitle(f"{summary['qubit']} T2* stability — {summary['status']}")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120)
    atomic_write(directory / "stability.png", buffer.getvalue())
    table = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if key != "allan"
    )
    atomic_write(
        directory / "index.html",
        f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="15"><title>T2* stability</title>
<style>body{{font:16px system-ui;margin:2em;max-width:1200px}} img{{width:100%}}
th,td{{text-align:left;padding:.25em 1em;border-bottom:1px solid #ddd}}</style></head><body>
<h1>{html.escape(summary['qubit'])} T2* stability</h1><p>Refreshes every 15 seconds.
Fit error bars are approximate uncertainties from the Ramsey fitter. Jump flags are candidates,
not confirmed physical events. Drift is descriptive; temporal variance includes fit noise.</p>
<img src="stability.png?v={len(rows)}" alt="T2, variance, frequency offset and Allan deviation">
<table>{table}</table><p>{html.escape(allan['reason'])}</p>
<p><a href="points.csv">Measurements CSV</a> · <a href="summary.json">Summary JSON</a></p>
</body></html>""",
    )


def persistent_ramsey_class():
    """Reuse Ramsey's BaseCalibration lifecycle, pulse sequence and safety latch."""
    Ramsey = importlib.import_module("calibrations.06a_ramsey").Ramsey

    class PersistentRamsey(Ramsey):
        def __init__(self, *args, point_directory, **kwargs):
            self.point_directory = point_directory
            super().__init__(*args, **kwargs)

        def execute_qua_program(self):
            self.namespace["measurement_started_monotonic"] = time.monotonic()
            self.namespace["measurement_started_utc"] = utc_now()
            try:
                return super().execute_qua_program()
            finally:
                self.namespace["measurement_finished_monotonic"] = time.monotonic()
                self.namespace["measurement_finished_utc"] = utc_now()

        def save_raw_results(self):
            save_dataset(self.point_directory, "raw", self.results["ds_raw"])

        def save_readout_mitigated_results(self):
            save_dataset(self.point_directory, "mitigated", self.results["ds_raw"])

        def save_analysis_result(self):
            save_dataset(self.point_directory, "fit", self.results["ds_fit"])
            write_json(
                self.point_directory / "fit_results.json", self.results["fit_results"]
            )

        def cleanup(self):
            # Base Ramsey's qm_session closes the owned QM on exceptions. Halt only
            # this acquisition's job as well; never close unrelated lab sessions.
            job = self.namespace.get("job")
            try:
                if job is not None:
                    job.halt()
            except Exception as exc:
                self.log(f"Could not halt finished/closed Ramsey job: {exc}")
            finally:
                super().cleanup()

    return PersistentRamsey


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", required=True)
    parser.add_argument("--profile", default="single_qubit")
    parser.add_argument("--duration-hours", type=float, default=24)
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=0,
        help="Start-to-start period; 0 means back-to-back. Missed slots are skipped.",
    )
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--num-shots", type=int, default=1000)
    parser.add_argument("--min-wait-ns", type=int, default=16)
    parser.add_argument("--max-wait-ns", type=int, default=3000)
    parser.add_argument("--wait-points", type=int, default=50)
    parser.add_argument("--detuning-mhz", type=float, default=1.0)
    parser.add_argument(
        "--reset-type", choices=["thermal", "active"], default="thermal"
    )
    parser.add_argument("--state-discrimination", action="store_true")
    parser.add_argument("--readout-mitigation", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--jump-sigma", type=float, default=5)
    parser.add_argument("--max-relative-error", type=float, default=0.5)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without creating a machine or connecting.",
    )
    args = parser.parse_args(argv)
    try:
        settings = StabilitySettings(
            args.qubit,
            args.duration_hours * 3600,
            args.interval_seconds,
            args.max_points,
            args.rolling_window,
            args.jump_sigma,
            args.max_relative_error,
            args.max_consecutive_failures,
        )
        if args.num_shots < 1 or args.timeout_seconds < 1 or args.wait_points < 6:
            raise ValueError("shots/timeout must be positive; wait-points must be >= 6")
        if args.min_wait_ns < 16 or args.max_wait_ns <= args.min_wait_ns:
            raise ValueError("Require 16 <= min-wait-ns < max-wait-ns")
        if args.wait_points > (args.max_wait_ns - args.min_wait_ns) // 4 + 1:
            raise ValueError("Too many wait points for distinct 4 ns clock steps")
        if not math.isfinite(args.detuning_mhz) or args.detuning_mhz <= 0:
            raise ValueError("detuning-mhz must be finite and positive")
        if not 0 <= args.readout_mitigation <= 1 or (
            args.readout_mitigation and not args.state_discrimination
        ):
            raise ValueError(
                "readout-mitigation must be in [0, 1] and requires state-discrimination"
            )
    except ValueError as exc:
        parser.error(str(exc))
    from calibration_utils.ramsey import Parameters

    parameters = Parameters(
        qubits=[args.qubit],
        num_shots=args.num_shots,
        min_wait_time_in_ns=args.min_wait_ns,
        max_wait_time_in_ns=args.max_wait_ns,
        wait_time_num_points=args.wait_points,
        frequency_detuning_in_mhz=args.detuning_mhz,
        log_or_linear_sweep="linear",
        reset_type=args.reset_type,
        use_state_discrimination=args.state_discrimination,
        use_readout_mitigation=args.readout_mitigation,
        timeout=args.timeout_seconds,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stability": asdict(settings),
                    "ramsey": parameters.model_dump(),
                    "profile": args.profile,
                    "hardware_execution": False,
                },
                indent=2,
            )
        )
        return 0

    return run_stability(settings, parameters, profile_name=args.profile, argv=argv)


def run_stability(
    settings: StabilitySettings, parameters, *, profile_name="single_qubit", argv=None
) -> int:
    """Run repeated Ramsey measurements with an editable experiment configuration."""
    from calibrations.base import CalibrationOptions
    from calibrations.output_safety import assert_outputs_allowed
    from calibration_io import CalibrationSaver
    from quam_config import create_machine

    assert_outputs_allowed()
    machine = create_machine(profile_name=profile_name, qubit=settings.qubit)
    options = CalibrationOptions(
        save_raw_data=True,
        save_analysis_result=True,
        save_figures=False,
        plot_data=False,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
    )
    run_directory = CalibrationSaver().save(
        "ramsey_stability",
        sweep={"point": np.array([], dtype=int)},
        results={"t2_us": np.array([], dtype=float)},
        profile_name=profile_name,
        parameters={"stability": asdict(settings), "ramsey": parameters.model_dump()},
        extra_metadata={
            "note": "Live results are in points.csv, points.jsonl and summary.json; initial NPZ arrays are empty."
        },
    )
    # Full resolved machine/config captures kernels omitted by the standard profile snapshot.
    write_json(run_directory / "machine_snapshot.json", machine.to_dict())
    write_json(run_directory / "config_snapshot.json", machine.generate_config())
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, "unavailable"
    write_json(
        run_directory / "provenance.json",
        {
            "git_revision": revision,
            "git_status": dirty,
            "python": sys.version,
            "argv": list(argv) if argv is not None else sys.argv[1:],
            "calibration_options": asdict(options),
            "started_utc": utc_now(),
        },
    )
    PersistentRamsey = persistent_ramsey_class()
    runner = RamseyStabilityRun(
        settings,
        run_directory,
        lambda directory: PersistentRamsey(
            parameters=parameters.model_copy(deep=True),
            machine=machine,
            options=options,
            point_directory=directory,
            profile_name=profile_name,
            qubit=settings.qubit,
        ),
    )
    print(
        f"Results: {run_directory}\nOpen {run_directory / 'index.html'} for the live report.",
        flush=True,
    )
    try:
        runner.run()
    except KeyboardInterrupt:
        print(f"Interrupted; completed points preserved in {run_directory}")
        return 130
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Preserve command-line usage, including --dry-run.
        sys.exit(main())

    from calibration_utils.ramsey import Parameters

    # Example: edit the qubit, monitoring duration, and Ramsey recipe before running.
    # Running this example acquires real hardware data and saves a live report.
    settings = StabilitySettings(
        qubit="q6",
        duration_seconds=24 * 3600,
        interval_seconds=60,
        max_points=None,  # Set a small limit for a short initial run.
        rolling_window=20,
    )

    parameters = Parameters()
    parameters.qubits = [settings.qubit]
    parameters.num_shots = 1000
    parameters.reset_type = "thermal"
    parameters.use_state_discrimination = False
    parameters.use_readout_mitigation = 0
    parameters.min_wait_time_in_ns = 16
    parameters.max_wait_time_in_ns = 3000
    parameters.wait_time_num_points = 50
    parameters.frequency_detuning_in_mhz = 1.0
    parameters.log_or_linear_sweep = "linear"
    parameters.timeout = 120

    run_stability(settings, parameters, profile_name="single_qubit")
