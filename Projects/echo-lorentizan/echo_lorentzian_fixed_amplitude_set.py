"""Loop fixed-amplitude echo-Lorentzian spectroscopy over selected amplitudes."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np

from calibrations_v2.base import CalibrationOptions
from echo_lorentzian_fixed_amplitude_v2 import EchoLorentzianFixedAmplitude
from lorentzian import _t2_seconds
from parameters import Parameters
from quam_config import create_machine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", default="q1")
    parser.add_argument("--cutoff", type=float, default=0.005)
    parser.add_argument("--pulse-length-us", type=float, default=20.0)
    parser.add_argument("--template-length-us", type=float, default=20.0)
    parser.add_argument("--peak-amplitude", type=float, default=0.2)
    parser.add_argument("--num-shots", type=int, default=1500)
    parser.add_argument("--num-points", type=int, default=1000)
    parser.add_argument("--frequency-span-mhz", type=float, default=1.0)
    parser.add_argument(
        "--rabi-amplitudes-mhz",
        type=float,
        nargs="+",
        default=[2.32, 4.64, 7.58, 11.45],
    )
    parser.add_argument(
        "--echo-modes",
        choices=("echo", "no-echo"),
        nargs="+",
        default=["no-echo", "echo"],
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parameters_for(
    args: argparse.Namespace, *, echo: bool, rabi_mhz: float
) -> Parameters:
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = echo
    parameters.cutoff = args.cutoff
    parameters.fixed_rabi_frequency_mhz = float(rabi_mhz)
    parameters.num_shots = args.num_shots
    parameters.lorentzian_length_in_ns = int(round(args.pulse_length_us * 1000 / 4) * 4)
    parameters.waveform_template_length_in_ns = int(
        round(args.template_length_us * 1000 / 4) * 4
    )
    parameters.lorentzian_peak_amplitude = args.peak_amplitude
    parameters.frequency_span_in_mhz = args.frequency_span_mhz
    parameters.frequency_step_in_mhz = args.frequency_span_mhz / max(
        args.num_points - 1,
        1,
    )
    return parameters


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = (
        REPOSITORY_ROOT / "data" / "echo_lorentzian_fixed_amplitude_set" / timestamp
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.csv"
    machine = create_machine(qubit=args.qubit)
    options = CalibrationOptions(
        save_raw_data=not args.no_save,
        save_analysis_result=False,
        save_figures=False,
        plot_data=False,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
    )

    rows = []
    for rabi_mhz in args.rabi_amplitudes_mhz:
        for echo_mode in args.echo_modes:
            rows.append((float(rabi_mhz), echo_mode == "echo"))

    traces: list[dict[str, object]] = []
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "index",
            "started_at",
            "finished_at",
            "status",
            "echo",
            "rabi_mhz",
            "run_directory",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Prepared {len(rows)} fixed-amplitude experiments.")
        for index, (rabi_mhz, echo) in enumerate(rows, start=1):
            started_at = datetime.now().astimezone().isoformat()
            mode = "echo" if echo else "no-echo"
            print(f"[{index}/{len(rows)}] {mode} {rabi_mhz:.2f} MHz", flush=True)
            run_directory = ""
            status = "dry-run"
            if not args.dry_run:
                parameters = parameters_for(args, echo=echo, rabi_mhz=rabi_mhz)
                calibration = EchoLorentzianFixedAmplitude(
                    parameters=parameters,
                    options=options,
                    machine=machine,
                    qubit=args.qubit,
                    auto_connect=index == 1,
                    name=f"echo_lorentzian_fixed_{mode}_{rabi_mhz:.2f}MHz",
                )
                calibration.run()
                run_directory = str(
                    calibration.namespace.get("calibration_run_directory", "")
                )
                traces.append(
                    trace_from_dataset(calibration.results["ds_raw"], rabi_mhz, echo)
                )
                status = "ok"
            finished_at = datetime.now().astimezone().isoformat()
            writer.writerow(
                {
                    "index": index,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": status,
                    "echo": echo,
                    "rabi_mhz": rabi_mhz,
                    "run_directory": run_directory,
                }
            )
            file.flush()

    if traces:
        plot_spectroscopy_traces(
            traces,
            manifest_dir,
            t2_seconds=_t2_seconds(machine.qubits[args.qubit]),
        )
    print(f"Manifest: {manifest_path}")
    return 0


def trace_from_dataset(dataset, rabi_mhz: float, echo: bool) -> dict[str, object]:
    detuning_mhz = np.asarray(dataset.detuning.values, dtype=float) / 1e6
    if "state" in dataset:
        values = np.asarray(dataset["state"].isel(qubit=0).values, dtype=float)
        ylabel = "Excitation"
    else:
        values = np.hypot(
            np.asarray(dataset["I"].isel(qubit=0).values, dtype=float),
            np.asarray(dataset["Q"].isel(qubit=0).values, dtype=float),
        )
        ylabel = "IQ amplitude [V]"
    if values.ndim == 2:
        values = np.nanmean(values, axis=1)
    return {
        "echo": bool(echo),
        "rabi_mhz": float(rabi_mhz),
        "detuning_mhz": detuning_mhz,
        "values": values,
        "ylabel": ylabel,
    }


def plot_spectroscopy_traces(
    traces: list[dict[str, object]],
    output_dir: Path,
    *,
    t2_seconds: float | None = None,
) -> dict[str, Path]:
    paths = {
        False: output_dir / "fixed_amplitude_lorentzian_no_echo.png",
        True: output_dir / "fixed_amplitude_echo_lorentzian.png",
    }
    titles = {
        False: "Lorentzian, no echo",
        True: "Echo Lorentzian",
    }
    saved_paths: dict[str, Path] = {}
    for echo in (False, True):
        selected = [trace for trace in traces if trace["echo"] is echo]
        if not selected:
            continue
        figure, ax = plt.subplots(figsize=(9, 5))
        for trace in sorted(selected, key=lambda item: item["rabi_mhz"]):
            ax.plot(
                trace["detuning_mhz"],
                trace["values"],
                linewidth=1.6,
                label=f"{trace['rabi_mhz']:.2f} MHz",
            )
        ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
        _add_t2_limit_lines(ax, t2_seconds)
        ax.set_title(titles[echo])
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel(str(selected[0]["ylabel"]))
        ax.grid(alpha=0.25)
        ax.legend(title="Rabi amplitude")
        figure.tight_layout()
        figure.savefig(paths[echo], dpi=180)
        plt.close(figure)
        saved_paths["echo" if echo else "no_echo"] = paths[echo]
        print(f"Saved {'echo' if echo else 'no-echo'} plot: {paths[echo]}")
    return saved_paths


def _add_t2_limit_lines(ax, t2_seconds: float | None) -> None:
    if t2_seconds is None:
        return
    t2_seconds = float(t2_seconds)
    if not np.isfinite(t2_seconds) or t2_seconds <= 0:
        return
    limit_mhz = 1 / (2 * np.pi * t2_seconds) / 1e6
    for index, detuning_mhz in enumerate((-limit_mhz, limit_mhz)):
        ax.axvline(
            detuning_mhz,
            color="0.55",
            linestyle=":",
            linewidth=1.2,
            label="T2 limit: ±1/(2πT2)" if index == 0 else None,
        )


if __name__ == "__main__":
    raise SystemExit(main())
