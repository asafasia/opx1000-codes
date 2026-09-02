"""Run echo shaped-pulse spectroscopy maps for ten pulse lengths.

The default campaign acquires one detuning-versus-amplitude map for each of
1, 2, 5, 10, 15, 20, 25, 30, 35, and 40 us.  Every map uses 200 detuning
points across a 2 MHz total span and 100 log-spaced peak amplitudes from
0.01 V through the lab-requested 1 V ceiling.  The pulse midpoint phase jump
is enabled (``echo=True``).

The command is plan-only unless ``--execute`` is supplied.  Use
``--execute --simulate`` to compile/simulate without driving hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "pulse_length_spectroscopy"
DEFAULT_LENGTHS_US = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
MAX_REQUESTED_AMPLITUDE_V = 1.0

for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", default="q6")
    parser.add_argument(
        "--lengths-us",
        type=float,
        nargs="+",
        default=DEFAULT_LENGTHS_US,
        help="Physical pulse lengths in microseconds.",
    )
    parser.add_argument("--num-shots", type=int, default=30)
    parser.add_argument("--frequency-span-mhz", type=float, default=2.0)
    parser.add_argument("--frequency-points", type=int, default=200)
    parser.add_argument("--amplitude-points", type=int, default=100)
    parser.add_argument("--min-amplitude-v", type=float, default=0.01)
    parser.add_argument("--max-amplitude-v", type=float, default=1.0)
    parser.add_argument("--pulse-shape", choices=["lorentzian", "root_lorentzian", "gaussian"], default="root_lorentzian")
    parser.add_argument("--cutoff", type=float, default=0.005)
    parser.add_argument(
        "--template-length-ns",
        type=int,
        default=2_000,
        help=(
            "Maximum stored waveform length. Shorter physical pulses use their "
            "full length; longer pulses stretch this template in QUA."
        ),
    )
    parser.add_argument(
        "--ac-stark-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--stark-kappa-mhz-inv", type=float, default=0.005)
    parser.add_argument("--stark-chirp-max-error-hz", type=float, default=10.0)
    parser.add_argument("--readout-mitigation", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--campaign-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing campaign directory, skipping successful lengths.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the planned campaign. Without this flag, only print the plan.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="With --execute, compile/simulate rather than run hardware.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.lengths_us or any(length <= 0 for length in args.lengths_us):
        raise ValueError("All pulse lengths must be positive.")
    lengths_ns = [int(round(length * 1_000)) for length in args.lengths_us]
    if any(length_ns % 4 for length_ns in lengths_ns):
        raise ValueError("Every pulse length must resolve to a multiple of 4 ns.")
    if args.num_shots <= 0:
        raise ValueError("num_shots must be positive.")
    if args.frequency_span_mhz <= 0:
        raise ValueError("frequency_span_mhz must be positive.")
    if args.frequency_points < 2:
        raise ValueError("frequency_points must be at least 2.")
    if args.amplitude_points < 2:
        raise ValueError("amplitude_points must be at least 2.")
    if not 0 < args.min_amplitude_v < args.max_amplitude_v:
        raise ValueError(
            "Log spacing requires 0 < min_amplitude_v < max_amplitude_v."
        )
    if args.max_amplitude_v > MAX_REQUESTED_AMPLITUDE_V:
        raise ValueError(
            f"max_amplitude_v cannot exceed {MAX_REQUESTED_AMPLITUDE_V:g} V."
        )
    if not 0 < args.cutoff <= 1:
        raise ValueError("cutoff must satisfy 0 < cutoff <= 1.")
    if args.template_length_ns < 4:
        raise ValueError("template_length_ns must be at least 4 ns.")
    if args.stark_kappa_mhz_inv < 0:
        raise ValueError("stark_kappa_mhz_inv cannot be negative.")
    if args.stark_chirp_max_error_hz <= 0:
        raise ValueError("stark_chirp_max_error_hz must be positive.")


def campaign_directory(args: argparse.Namespace) -> Path:
    campaign_id = args.campaign_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return (args.output_root / campaign_id).resolve()


def parameters_for(args: argparse.Namespace, pulse_length_us: float):
    from shaped_pulse_spectroscopy.parameters import Parameters

    pulse_length_ns = int(round(pulse_length_us * 1_000))
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.use_readout_mitigation = args.readout_mitigation
    parameters.simulate = bool(args.simulate)
    parameters.pulse_shape = args.pulse_shape
    parameters.echo = True
    parameters.ac_stark_correction = bool(args.ac_stark_correction)
    parameters.stark_kappa_mhz_inv = float(args.stark_kappa_mhz_inv)
    parameters.stark_chirp_max_error_hz = float(args.stark_chirp_max_error_hz)
    parameters.cutoff = float(args.cutoff)
    parameters.num_shots = int(args.num_shots)
    parameters.lorentzian_length_in_ns = pulse_length_ns
    parameters.waveform_template_length_in_ns = min(
        pulse_length_ns, int(args.template_length_ns)
    )

    # Using the requested maximum as the unscaled waveform amplitude makes the
    # prefactors dimensionless fractions of that ceiling.  The final point is
    # therefore exactly the requested maximum peak voltage.
    parameters.lorentzian_peak_amplitude = float(args.max_amplitude_v)
    parameters.min_amp_factor = float(
        args.min_amplitude_v / args.max_amplitude_v
    )
    parameters.max_amp_factor = 1.0
    parameters.amp_factor_points = int(args.amplitude_points)
    parameters.amp_factor_spacing = "log"
    parameters.amp_factor_step = 1.0 / (args.amplitude_points - 1)

    parameters.frequency_span_in_mhz = float(args.frequency_span_mhz)
    parameters.frequency_points = int(args.frequency_points)
    parameters.frequency_step_in_mhz = (
        args.frequency_span_mhz / (args.frequency_points - 1)
    )
    parameters.fit_fwhm = False
    return parameters


def settings_snapshot(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "simulation" if args.simulate else "hardware",
        "qubit": args.qubit,
        "echo": True,
        "pulse_shape": args.pulse_shape,
        "pulse_lengths_us": [float(value) for value in args.lengths_us],
        "num_shots": args.num_shots,
        "frequency_span_mhz": args.frequency_span_mhz,
        "frequency_points": args.frequency_points,
        "amplitude_spacing": "log",
        "amplitude_points": args.amplitude_points,
        "min_amplitude_v": args.min_amplitude_v,
        "max_amplitude_v": args.max_amplitude_v,
        "cutoff": args.cutoff,
        "template_length_ns": args.template_length_ns,
        "ac_stark_correction": args.ac_stark_correction,
        "stark_kappa_mhz_inv": args.stark_kappa_mhz_inv,
        "stark_chirp_max_error_hz": args.stark_chirp_max_error_hz,
        "readout_mitigation": args.readout_mitigation,
        "output_directory": str(output_dir),
        "runs": [],
    }


def device_snapshot(machine: Any, qubit_name: str) -> dict[str, Any]:
    """Capture the calibrated quantities required by matched simulations."""
    qubit = machine.qubits[qubit_name]
    x180 = qubit.xy.operations["x180"]
    return {
        "rf_frequency_hz": float(qubit.xy.RF_frequency),
        "anharmonicity_hz": float(qubit.anharmonicity),
        "t1_s": float(qubit.T1),
        "t2_star_s": float(qubit.T2ramsey),
        "x180_amplitude_v": float(x180.amplitude),
        "x180_length_ns": float(x180.length),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def length_label(pulse_length_us: float) -> str:
    return f"{pulse_length_us:g}".replace(".", "p") + "us"


def save_plot(calibration: Any, path: Path) -> None:
    from shaped_pulse_spectroscopy.lorentzian import plot_raw_data

    figure = plot_raw_data(
        calibration.results["ds_raw"],
        calibration.namespace["qubits"],
        use_state_discrimination=True,
    )
    if isinstance(figure, (list, tuple)):
        figure = figure[0]
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_plan(args: argparse.Namespace, output_dir: Path) -> None:
    total_grid_points = (
        len(args.lengths_us)
        * args.frequency_points
        * args.amplitude_points
        * args.num_shots
    )
    mode = "SIMULATION" if args.simulate else "HARDWARE"
    print(f"[{'EXECUTE' if args.execute else 'PLAN ONLY'}] {mode} campaign")
    print(f"  qubit / echo: {args.qubit} / True")
    print(f"  pulse lengths (us): {', '.join(f'{x:g}' for x in args.lengths_us)}")
    print(
        f"  detuning: {args.frequency_points} points across "
        f"{args.frequency_span_mhz:g} MHz total span"
    )
    print(
        f"  amplitude: {args.amplitude_points} log points, "
        f"{args.min_amplitude_v:g}..{args.max_amplitude_v:g} V"
    )
    print(f"  shots per grid point: {args.num_shots}")
    print(f"  total shot-grid iterations: {total_grid_points:,}")
    print(f"  pulse shape / cutoff: {args.pulse_shape} / {args.cutoff:g}")
    print(f"  AC-Stark correction: {args.ac_stark_correction}")
    print(f"  output: {output_dir}")
    if not args.execute:
        print("  add --execute to start this campaign")


def run_campaign(args: argparse.Namespace, output_dir: Path) -> None:
    from calibrations.base import CalibrationOptions
    from experiments.detuning_amplitude_sweep import EchoLorentzian
    from quam_config import create_machine

    options = CalibrationOptions(
        save_raw_data=not args.simulate,
        save_analysis_result=False,
        save_figures=False,
        analyse_data=not args.simulate,
        plot_data=False,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
        ai_review=False,
    )
    machine = create_machine(qubit=args.qubit)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if args.resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = settings_snapshot(args, output_dir)
        immutable_keys = (
            "mode",
            "qubit",
            "echo",
            "pulse_shape",
            "pulse_lengths_us",
            "num_shots",
            "frequency_span_mhz",
            "frequency_points",
            "amplitude_spacing",
            "amplitude_points",
            "min_amplitude_v",
            "max_amplitude_v",
            "cutoff",
            "template_length_ns",
            "ac_stark_correction",
            "readout_mitigation",
        )
        mismatches = [
            key for key in immutable_keys if manifest.get(key) != expected.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Cannot resume because campaign settings differ: "
                + ", ".join(mismatches)
            )
    else:
        manifest = settings_snapshot(args, output_dir)
        manifest["device"] = device_snapshot(machine, args.qubit)
    write_manifest(manifest_path, manifest)
    total = len(args.lengths_us)
    successful_lengths = {
        float(record["pulse_length_us"])
        for record in manifest.get("runs", [])
        if record.get("status") == "ok"
    }
    connected = False

    for index, pulse_length_us in enumerate(args.lengths_us, start=1):
        if float(pulse_length_us) in successful_lengths:
            print(
                f"[{index}/{total}] Skipping completed {pulse_length_us:g} us.",
                flush=True,
            )
            continue
        label = length_label(pulse_length_us)
        record = {
            "index": index,
            "pulse_length_us": pulse_length_us,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "finished_at": None,
            "run_directory": None,
            "figure": None,
            "error": None,
        }
        manifest["runs"].append(record)
        write_manifest(manifest_path, manifest)
        print(f"[{index}/{total}] Starting {pulse_length_us:g} us...", flush=True)

        try:
            calibration = EchoLorentzian(
                parameters=parameters_for(args, pulse_length_us),
                options=options,
                machine=machine,
                qubit=args.qubit,
                auto_connect=(not connected and not args.simulate),
                name=f"pulse_length_{label}_echo",
            )
            connected = True
            calibration.run()
            run_directory = calibration.namespace.get("calibration_run_directory")
            record["run_directory"] = (
                str(run_directory) if run_directory is not None else None
            )
            if not args.simulate:
                figure_path = output_dir / f"{index:02d}_{label}_echo.png"
                save_plot(calibration, figure_path)
                record["figure"] = str(figure_path)
            record["status"] = "ok"
            print(f"[{index}/{total}] Finished {pulse_length_us:g} us.", flush=True)
        except BaseException as error:
            record["status"] = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
            )
            record["error"] = f"{type(error).__name__}: {error}"
            print(f"[{index}/{total}] {record['error']}", flush=True)
            raise
        finally:
            record["finished_at"] = datetime.now().astimezone().isoformat()
            write_manifest(manifest_path, manifest)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    output_dir = campaign_directory(args)
    print_plan(args, output_dir)
    if args.execute:
        run_campaign(args, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
