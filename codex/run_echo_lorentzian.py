"""Codex-friendly runner for the echo-Lorentzian calibration.

This script keeps a stable command-line entry point for running short smoke
tests or full echo-Lorentzian sweeps without editing the experiment source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "Projects" / "echo-lorentizan"
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrations_v2.base import CalibrationOptions
from echo_lorentzian_v2 import EchoLorentzian
from parameters import Parameters
from quam_config import create_machine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the echo-Lorentzian v2 calibration with CLI overrides."
    )
    parser.add_argument(
        "--preset",
        choices=("short", "full"),
        default="short",
        help="short is a quick hardware smoke test; full uses the dense scan.",
    )
    parser.add_argument("--qubit", default="q9")
    parser.add_argument(
        "--pulse-shape",
        choices=("lorentzian", "root_lorentzian", "gaussian"),
        default="root_lorentzian",
    )
    parser.add_argument("--echo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--state-discrimination",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--reset-type", default="active")
    parser.add_argument("--cutoff", type=float, default=0.0005)
    parser.add_argument("--num-shots", type=int)
    parser.add_argument("--pulse-length-ns", type=int)
    parser.add_argument("--template-length-ns", type=int)
    parser.add_argument("--peak-amplitude", type=float)
    parser.add_argument("--min-amp-factor", type=float)
    parser.add_argument("--max-amp-factor", type=float)
    parser.add_argument("--amp-factor-step", type=float)
    parser.add_argument("--frequency-span-mhz", type=float)
    parser.add_argument("--frequency-step-mhz", type=float)
    parser.add_argument(
        "--simulate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass through the Parameters.simulate flag.",
    )
    return parser


def apply_preset(parameters: Parameters, preset: str) -> None:
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = True
    parameters.cutoff = 0.0005

    if preset == "full":
        parameters.num_shots = 500
        parameters.lorentzian_length_in_ns = 160000
        parameters.waveform_template_length_in_ns = 60000
        parameters.lorentzian_peak_amplitude = 0.5
        parameters.min_amp_factor = 0.0
        parameters.max_amp_factor = 1.0
        parameters.amp_factor_step = 0.005
        parameters.frequency_span_in_mhz = 5
        parameters.frequency_step_in_mhz = 0.01
        return

    parameters.num_shots = 2
    parameters.lorentzian_length_in_ns = 4000
    parameters.waveform_template_length_in_ns = 2000
    parameters.lorentzian_peak_amplitude = 0.1
    parameters.min_amp_factor = 0.0
    parameters.max_amp_factor = 0.1
    parameters.amp_factor_step = 0.05
    parameters.frequency_span_in_mhz = 0.2
    parameters.frequency_step_in_mhz = 0.2


def apply_overrides(parameters: Parameters, args: argparse.Namespace) -> None:
    parameters.use_state_discrimination = args.state_discrimination
    parameters.reset_type = args.reset_type
    parameters.pulse_shape = args.pulse_shape
    parameters.echo = args.echo
    parameters.cutoff = args.cutoff
    parameters.simulate = args.simulate

    overrides = {
        "num_shots": args.num_shots,
        "lorentzian_length_in_ns": args.pulse_length_ns,
        "waveform_template_length_in_ns": args.template_length_ns,
        "lorentzian_peak_amplitude": args.peak_amplitude,
        "min_amp_factor": args.min_amp_factor,
        "max_amp_factor": args.max_amp_factor,
        "amp_factor_step": args.amp_factor_step,
        "frequency_span_in_mhz": args.frequency_span_mhz,
        "frequency_step_in_mhz": args.frequency_step_mhz,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(parameters, name, value)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    parameters = Parameters()
    apply_preset(parameters, args.preset)
    apply_overrides(parameters, args)

    print(
        "Running echo-Lorentzian:",
        f"qubit={args.qubit}",
        f"preset={args.preset}",
        f"pulse_shape={parameters.pulse_shape}",
        f"echo={parameters.echo}",
        f"shots={parameters.num_shots}",
        f"amp={parameters.min_amp_factor}:{parameters.amp_factor_step}:{parameters.max_amp_factor}",
        f"detuning_span={parameters.frequency_span_in_mhz} MHz",
        f"detuning_step={parameters.frequency_step_in_mhz} MHz",
        flush=True,
    )

    calibration = EchoLorentzian(
        parameters=parameters,
        options=CalibrationOptions(),
        machine=create_machine(qubit=args.qubit),
        auto_connect=True,
    )
    calibration.run()


if __name__ == "__main__":
    main()
