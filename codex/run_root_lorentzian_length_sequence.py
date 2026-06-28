"""Run a 9-step root-Lorentzian length sequence and build a 3x3 figure.

The default settings are meant to be about 30 minutes per experiment on the
same dense grid that was used for the long echo-Lorentzian runs, but with
100 averages instead of 500.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "Projects" / "echo-lorentizan"
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_LENGTHS_US = (2, 5, 10, 30, 60, 90, 130, 160, 250)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run root-Lorentzian scans for multiple pulse lengths."
    )
    parser.add_argument("--qubit", default="q9")
    parser.add_argument(
        "--lengths-us",
        nargs="+",
        type=float,
        default=list(DEFAULT_LENGTHS_US),
        help="Pulse lengths to run, in microseconds.",
    )
    parser.add_argument("--num-shots", type=int, default=100)
    parser.add_argument("--echo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cutoff", type=float, default=0.0005)
    parser.add_argument("--peak-amplitude", type=float, default=0.5)
    parser.add_argument("--min-amp-factor", type=float, default=0.0)
    parser.add_argument("--max-amp-factor", type=float, default=1.0)
    parser.add_argument("--amp-factor-step", type=float, default=0.005)
    parser.add_argument("--frequency-span-mhz", type=float, default=5.0)
    parser.add_argument("--frequency-step-mhz", type=float, default=0.01)
    parser.add_argument(
        "--max-template-length-us",
        type=float,
        default=60.0,
        help="Use the pulse length below this value, otherwise stretch this template length.",
    )
    parser.add_argument("--reset-type", default="active")
    parser.add_argument(
        "--state-discrimination",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "codex" / "root_lorentzian_sequence",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only build a montage from --run-dirs.",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        type=Path,
        default=[],
        help="Existing run directories to use with --skip-run.",
    )
    return parser


def make_parameters(args: argparse.Namespace, length_us: float):
    from parameters import Parameters

    length_ns = int(round(length_us * 1000))
    template_ns = int(round(min(length_us, args.max_template_length_us) * 1000))

    parameters = Parameters()
    parameters.use_state_discrimination = args.state_discrimination
    parameters.reset_type = args.reset_type
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = args.echo
    parameters.cutoff = args.cutoff
    parameters.num_shots = args.num_shots
    parameters.lorentzian_length_in_ns = length_ns
    parameters.waveform_template_length_in_ns = template_ns
    parameters.lorentzian_peak_amplitude = args.peak_amplitude
    parameters.min_amp_factor = args.min_amp_factor
    parameters.max_amp_factor = args.max_amp_factor
    parameters.amp_factor_step = args.amp_factor_step
    parameters.frequency_span_in_mhz = args.frequency_span_mhz
    parameters.frequency_step_in_mhz = args.frequency_step_mhz
    return parameters


def run_one(args: argparse.Namespace, length_us: float) -> Path:
    from calibrations_v2.base import CalibrationOptions
    from echo_lorentzian_v2 import EchoLorentzian
    from quam_config import create_machine

    parameters = make_parameters(args, length_us)
    print(
        "=== START root_lorentzian length "
        f"{length_us:g} us, echo={parameters.echo}, shots={parameters.num_shots} ===",
        flush=True,
    )
    calibration = EchoLorentzian(
        parameters=parameters,
        options=CalibrationOptions(),
        machine=create_machine(qubit=args.qubit),
        auto_connect=True,
    )
    calibration.run()
    run_dir = calibration.namespace.get("calibration_run_directory")
    if run_dir is None:
        raise RuntimeError(f"Run for {length_us:g} us did not save a run directory.")
    print(f"=== DONE {length_us:g} us: {run_dir} ===", flush=True)
    return Path(run_dir)


def figure_path(run_dir: Path, qubit: str) -> Path:
    return run_dir / "figures" / f"echo_lorentzian_{qubit}.png"


def build_montage(
    run_dirs: list[Path],
    lengths_us: list[float],
    qubit: str,
    echo: bool,
    output_path: Path,
) -> None:
    if len(run_dirs) != len(lengths_us):
        raise ValueError("run_dirs and lengths_us must have the same length.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    for ax, run_dir, length_us in zip(axes.ravel(), run_dirs, lengths_us):
        image_path = figure_path(run_dir, qubit)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing figure for {length_us:g} us: {image_path}")
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(f"{length_us:g} us", fontsize=14)
        ax.axis("off")

    fig.suptitle(f"Root-Lorentzian length sequence, echo={echo}", fontsize=18)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved 3x3 sequence figure to {output_path}", flush=True)


def write_manifest(
    output_dir: Path,
    lengths_us: list[float],
    run_dirs: list[Path],
    montage_path: Path,
    args: argparse.Namespace,
) -> None:
    manifest = {
        "created": datetime.now().isoformat(),
        "qubit": args.qubit,
        "pulse_shape": "root_lorentzian",
        "echo": args.echo,
        "lengths_us": lengths_us,
        "run_dirs": [str(path) for path in run_dirs],
        "montage": str(montage_path),
        "num_shots": args.num_shots,
        "cutoff": args.cutoff,
        "peak_amplitude": args.peak_amplitude,
        "amp_factor": {
            "min": args.min_amp_factor,
            "max": args.max_amp_factor,
            "step": args.amp_factor_step,
        },
        "detuning": {
            "span_mhz": args.frequency_span_mhz,
            "step_mhz": args.frequency_step_mhz,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sequence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved sequence manifest to {manifest_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    lengths_us = [float(length) for length in args.lengths_us]
    if len(lengths_us) != 9:
        raise ValueError("This sequence expects exactly 9 pulse lengths for a 3x3 figure.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    montage_path = output_dir / "root_lorentzian_length_sequence_3x3.png"

    if args.skip_run:
        run_dirs = [Path(path) for path in args.run_dirs]
        if len(run_dirs) != len(lengths_us):
            raise ValueError("--skip-run requires one --run-dirs entry per length.")
    else:
        print(
            f"Running 9 root-Lorentzian experiments with echo={args.echo}. "
            "Expected total time is roughly 4.5 hours.",
            flush=True,
        )
        run_dirs = [run_one(args, length_us) for length_us in lengths_us]

    build_montage(run_dirs, lengths_us, args.qubit, args.echo, montage_path)
    write_manifest(output_dir, lengths_us, run_dirs, montage_path, args)


if __name__ == "__main__":
    main()
