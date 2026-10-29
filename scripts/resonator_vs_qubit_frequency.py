"""Sweep resonator spectroscopy over qubit RF frequencies.

Ported from opx1000_Or/scripts/resonator_vs_qubit_frequency.py.
Uses this repository's profiles and BaseCalibration resonator experiment.

Offline check (no hardware connection or output files):
    python scripts/resonator_vs_qubit_frequency.py --qubit q3b --dry-run
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from calibrations import CalibrationOptions
from calibration_utils.resonator_spectroscopy import Parameters, process_raw_dataset
from profiles import DEFAULT_PROFILE_NAME, Profile, validate_profile
from profiles.loader import _select_qubit
from quam_config.create_machine_from_profile import create_machine_from_profile

ResonatorSpectroscopy = importlib.import_module(
    "calibrations.02a_resonator_spectroscopy"
).ResonatorSpectroscopy


@dataclass(frozen=True)
class SweepSettings:
    # These are physical qubit drive frequencies (RF), including both endpoints.
    start_hz: int = 3_500_000_000
    stop_hz: int = 5_500_000_000
    points: int = 100
    intermediate_frequency_hz: int = 100_000_000  # RF = LO + 100 MHz

    def frequencies(self):
        for name in ("start_hz", "stop_hz", "points", "intermediate_frequency_hz"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.points < 2 or self.stop_hz <= self.start_hz:
            raise ValueError("Use at least two points and stop_hz > start_hz")
        frequencies = np.rint(
            np.linspace(self.start_hz, self.stop_hz, self.points)
        ).astype(np.int64)
        if np.any(np.diff(frequencies) <= 0):
            raise ValueError(
                "Qubit frequencies must be distinct at integer-Hz precision"
            )
        return frequencies


def write_manifest(directory, manifest):
    temporary = directory / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "manifest.json")


def summarize(directory, frequencies, resonator, scans, *, show_plot=False):
    """Collect saved scan statistics; raw shots remain in each scan directory."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    keys = (
        "amplitude_volts_g",
        "amplitude_volts_saturated",
        "contrast_volts",
        "separation",
    )
    rows = {key: [] for key in keys}
    for scan in scans:
        with np.load(Path(scan) / "results.npz") as data:
            if not np.array_equal(data["frequency_hz"], resonator):
                raise ValueError("Readout frequency axis changed between scans")
            for key in keys:
                if data[key].shape != resonator.shape:
                    raise ValueError(f"Unexpected {key} shape in {scan}")
                rows[key].append(data[key].copy())
    data = {key: np.stack(values) for key, values in rows.items()}
    valid = np.isfinite(data["separation"])
    best = None
    maximum = np.full(len(frequencies), np.nan)
    for row in range(len(frequencies)):
        if valid[row].any():
            maximum[row] = data["separation"][row, valid[row]].max()
    if valid.any():
        qi, ri = np.unravel_index(
            np.argmax(np.where(valid, data["separation"], -np.inf)), valid.shape
        )
        best = {
            "qubit_frequency_hz": int(frequencies[qi]),
            "resonator_frequency_hz": int(resonator[ri]),
            "separation": float(data["separation"][qi, ri]),
        }
    np.savez_compressed(
        directory / "summary.npz",
        qubit_frequency_hz=frequencies,
        resonator_frequency_hz=resonator,
        maximum_separation=maximum,
        **data,
    )
    x, y = resonator / 1e9, frequencies / 1e9
    with plt.rc_context(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "font.size": 9,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for ax, key, title in zip(
            axes[0],
            ("contrast_volts", "separation"),
            ("Complex IQ contrast (V)", "IQ separation / pooled std"),
        ):
            mesh = ax.pcolormesh(
                x, y, np.ma.masked_invalid(data[key]), shading="auto", cmap="viridis"
            )
            fig.colorbar(mesh, ax=ax, label=title)
            ax.set(
                xlabel="Readout frequency (GHz)",
                ylabel="Qubit drive RF (GHz)",
                title=title,
            )
            if best is not None:
                ax.plot(x[ri], y[qi], "wx", ms=9, mew=2)
        axes[1, 0].plot(y, maximum, color="#7c3aed", lw=1.5)
        axes[1, 0].set(
            xlabel="Qubit drive RF (GHz)",
            ylabel="Maximum separation / pooled std",
            title="Best readout separation at each qubit frequency",
        )
        if best is not None:
            axes[1, 0].plot(y[qi], maximum[qi], "o", color="#dc2626")
            for key, label, color in (
                ("amplitude_volts_g", "Drive off", "#2563eb"),
                ("amplitude_volts_saturated", "After saturation", "#ea580c"),
            ):
                axes[1, 1].plot(x, data[key][qi], label=label, color=color)
            axes[1, 1].axvline(x[ri], color="#7c3aed", ls="--", label="Best separation")
            axes[1, 1].legend()
            axes[1, 1].set(
                xlabel="Readout frequency (GHz)",
                ylabel="Mean response amplitude (V)",
                title=f"Best sampled qubit RF: {y[qi]:.6f} GHz",
            )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No finite separation scores",
                ha="center",
                transform=axes[1, 1].transAxes,
            )
        fig.suptitle(
            "Resonator spectroscopy vs qubit drive frequency\nDrive off / after saturation"
        )
        fig.savefig(directory / "summary.png", dpi=160)

        # Five readable 4 x 5 pages for the default 100 points.
        for start in range(0, len(frequencies), 20):
            grid, panels = plt.subplots(
                5, 4, figsize=(16, 16), sharex=True, sharey=True
            )
            for index, ax in enumerate(panels.flat, start):
                if index >= len(frequencies):
                    ax.set_visible(False)
                    continue
                ax.plot(x, data["amplitude_volts_g"][index], color="#2563eb", lw=1.2)
                ax.plot(
                    x, data["amplitude_volts_saturated"][index], color="#ea580c", lw=1.2
                )
                ax.set_title(f"RF {y[index]:.6f} GHz | separation {maximum[index]:.2f}")
                ax.grid(alpha=0.2)
                ax.tick_params(axis="x", labelrotation=25)
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            grid.suptitle(
                f"Resonator traces: qubit frequencies {start + 1}–{min(start + 20, len(frequencies))}",
                y=0.995,
            )
            grid.legend(
                handles=[
                    Line2D([], [], color="#2563eb", label="Drive off"),
                    Line2D([], [], color="#ea580c", label="After saturation"),
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.978),
                ncol=2,
            )
            grid.supxlabel("Readout frequency (GHz)")
            grid.supylabel("Mean response amplitude (V)")
            grid.tight_layout(rect=(0.025, 0.025, 1, 0.95))
            grid.savefig(
                directory / f"resonator_grid_{start // 20 + 1:02d}.png", dpi=150
            )
            plt.close(grid)
        if show_plot:
            plt.show()
        plt.close(fig)
    return best


class SweepProfile(Profile):
    """Validated in-memory configuration for one sweep point."""

    def __init__(self, documents):
        super().__init__(documents["manifest"]["name"])
        self.documents = deepcopy(documents)

    def load(self, *, qubit=None):
        documents = deepcopy(self.documents)
        validate_profile(documents)
        return documents


def point_settings(documents, qubit, frequency_hz, sweep):
    """Retune RF and the connected drive LO without changing profile files."""
    updated = deepcopy(documents)
    updated["qubits"]["qubits"][qubit]["frequencies_hz"]["qubit_f01"] = int(frequency_hz)
    connection = updated["connectivity"]["connections"][qubit]
    lo = int(frequency_hz) - sweep.intermediate_frequency_hz
    connection.setdefault("lo_frequencies_hz", {})["xy_output"] = lo
    port = connection["xy_output"]
    updated["connectivity"]["controllers"][port["controller"]]["fems"][
        str(port["fem"])
    ]["outputs"][str(port["port"])]["lo_frequency_hz"] = lo
    validate_profile(updated)
    return updated


def build_calibration(documents, parameters):
    machine = create_machine_from_profile(SweepProfile(documents), save=False)
    return ResonatorSpectroscopy(
        parameters=parameters.model_copy(deep=True),
        machine=machine,
        options=CalibrationOptions(
            save_raw_data=False, save_analysis_result=False, save_figures=False,
            analyse_data=False, plot_data=False, update_state=False,
            propose_profile_update=False, apply_profile_update=False,
        ),
    )


def run_resonator(documents, parameters, directory):
    """Use the shared hardware lifecycle and retain raw shots before analysis."""
    calibration = build_calibration(documents, parameters)
    directory.mkdir(parents=True)
    (directory / "profile.json").write_text(
        json.dumps(documents, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "parameters.json").write_text(
        parameters.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    status = calibration.run()
    if status.interrupted:
        raise KeyboardInterrupt("Resonator acquisition stopped")
    raw = calibration.results["ds_raw"]
    # Store dimension names, coordinates and attributes alongside raw shot arrays.
    np.savez_compressed(directory / "raw_shots.npz", **{
        key: value.values for key, value in raw.variables.items()
    })
    (directory / "raw_metadata.json").write_text(
        json.dumps(raw.to_dict(data=False), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    ds = process_raw_dataset(raw, calibration).sel(qubit=parameters.qubits[0])
    contrast = np.hypot(
        (ds.Im - ds.Ig).mean("n_runs"), (ds.Qm - ds.Qg).mean("n_runs")
    )
    np.savez_compressed(
        directory / "results.npz", frequency_hz=ds.full_freq.values,
        amplitude_volts_g=ds.ground_IQ_abs.values,
        amplitude_volts_saturated=ds.mixed_IQ_abs.values,
        contrast_volts=contrast.values, separation=ds.IQ_separation.values,
    )
    return directory


def main(*, qubit, profile_name=DEFAULT_PROFILE_NAME, parameters=None, sweep=None,
         dry_run=False, output_dir=Path("data"), show_plot=True,
         readout_amplitude_factor=1.0, profile_documents=None):
    sweep = SweepSettings() if sweep is None else sweep
    frequencies = sweep.frequencies()
    parameters = Parameters() if parameters is None else parameters
    if parameters.simulate or parameters.load_data_id is not None:
        raise ValueError("Use --dry-run for this outer sweep; simulation/loading is not supported")
    parameters = parameters.model_copy(update={
        "qubits": [qubit], "states": ["g", "e"], "readout_states": ["g", "e"],
        "qubit_operation": "saturation", "reset_type": "thermal",
        "use_state_discrimination": False, "use_readout_mitigation": False,
    }, deep=True)
    if (parameters.num_shots < 2 or not np.isfinite(parameters.frequency_span_in_mhz)
            or not np.isfinite(parameters.frequency_step_in_mhz)
            or parameters.frequency_step_in_mhz <= 0
            or parameters.frequency_span_in_mhz <= parameters.frequency_step_in_mhz):
        raise ValueError("Use at least two shots and a positive readout step smaller than the span")
    if not np.isfinite(readout_amplitude_factor) or readout_amplitude_factor <= 0:
        raise ValueError("readout_amplitude_factor must be finite and positive")
    documents = (Profile(profile_name, qubit=qubit).load()
                 if profile_documents is None else deepcopy(profile_documents))
    validate_profile(documents)
    if documents["manifest"]["name"] != profile_name:
        raise ValueError("Snapshot profile name does not match profile_name")
    if documents["manifest"].get("build_mode") != "single_qubit":
        raise ValueError("Select a single_qubit build-mode profile for independent LO retuning")
    documents = _select_qubit(documents, qubit)
    pulse_name = (documents["manifest"].get("readout_pulse")
                  or documents["qubits"]["qubits"][qubit]["operations"]["readout"])
    pulse = documents["pulses"]["pulses"][qubit][pulse_name]
    original_readout_amplitude = pulse["amplitude"]
    pulse["amplitude"] *= readout_amplitude_factor
    validate_profile(documents)
    points = [point_settings(documents, qubit, frequency, sweep) for frequency in frequencies]
    print(f"{qubit}: {len(points)} RF points, {frequencies[0]/1e9:g} to "
          f"{frequencies[-1]/1e9:g} GHz; IF +{sweep.intermediate_frequency_hz/1e6:g} MHz")
    # Build EVERY config and QUA program before connecting or creating output.
    for point in points:
        calibration = build_calibration(point, parameters)
        calibration.machine.generate_config()
        calibration.create_qua_program()
        calibration.cleanup()
    resonator = (
        np.asarray(calibration.namespace["sweep_axes"]["detuning"])
        + documents["qubits"]["qubits"][qubit]["frequencies_hz"]["resonator"]
    )
    if dry_run:
        print("All RF/LO configurations and QUA programs built; no hardware connection or files saved.")
        return None

    directory = (Path(output_dir) / "resonator_vs_qubit_frequency"
                 / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{qubit}")
    directory.mkdir(parents=True)
    manifest = {
        "status": "running", "sweep": asdict(sweep), "profile": profile_name,
        "qubit": qubit, "qubit_frequency_hz": frequencies.tolist(),
        "readout_pulse": pulse_name,
        "readout_amplitude_factor": readout_amplitude_factor,
        "original_readout_amplitude": original_readout_amplitude,
        "readout_amplitude": pulse["amplitude"],
        "drive_lo_frequency_hz": (frequencies - sweep.intermediate_frequency_hz).tolist(),
        "state_preparation": "Thermal recovery / continuous saturation during readout; not pure |0> and |1>.",
        "separation_formula": "abs(mean(S_saturated)-mean(S_g))/sqrt((var(S_g)+var(S_saturated))/2)",
        "variance_ddof": 0, "summary_axes": ["qubit_frequency", "readout_frequency"],
        "selection": "Sampled maximum; not assignment fidelity or a fitted qubit resonance.",
        "completed_scans": [],
    }
    write_manifest(directory, manifest)
    scans = []
    try:
        for index, point in enumerate(points):
            print(f"[{index + 1}/{len(points)}] Qubit RF {frequencies[index]/1e9:.9f} GHz")
            scan = run_resonator(point, parameters, directory / f"point_{index + 1:03d}")
            scans.append(scan)
            manifest["completed_scans"].append({
                "index": index, "qubit_frequency_hz": int(frequencies[index]),
                "drive_lo_frequency_hz": int(frequencies[index]) - sweep.intermediate_frequency_hz,
                "path": str(scan.relative_to(directory)),
            })
            write_manifest(directory, manifest)
        manifest["maximum_separation"] = summarize(
            directory, frequencies, resonator, scans, show_plot=show_plot
        )
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_manifest(directory, manifest)
        print(f"Saved sweep: {directory.resolve()}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", required=True, help="Profile qubit name, e.g. q3b")
    parser.add_argument("--profile", default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--start-hz", type=int, default=3_500_000_000)
    parser.add_argument("--stop-hz", type=int, default=5_500_000_000)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--if-hz", type=int, default=100_000_000)
    parser.add_argument("--num-shots", type=int, default=200)
    parser.add_argument("--readout-amplitude-factor", type=float, default=1.0)
    parser.add_argument("--readout-span-mhz", type=float, default=30.0)
    parser.add_argument("--readout-step-mhz", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true", help="Build all configurations without hardware")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening a window")
    args = parser.parse_args()
    main(qubit=args.qubit, profile_name=args.profile, dry_run=args.dry_run,
         readout_amplitude_factor=args.readout_amplitude_factor,
         output_dir=args.output_dir, show_plot=not args.no_show,
         sweep=SweepSettings(args.start_hz, args.stop_hz, args.points, args.if_hz),
         parameters=Parameters(num_shots=args.num_shots,
                               frequency_span_in_mhz=args.readout_span_mhz,
                               frequency_step_in_mhz=args.readout_step_mhz))


