"""Benchmark readout fidelity using the repository QuAM machine config.

The selected profile decides whether optimized weights are used. For example,
profiles/single_qubit/qubits.json can set:

    qubits.q1.readout.use_kernel = true

Then create_machine(qubit="q1") loads profiles/single_qubit/kernels/
q1_readout_kernel.npz and machine.generate_config() emits those weights.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from calibration_utils.iq_blobs import Parameters
from calibrations.core import CalibrationOptions


IqBlobs = importlib.import_module("calibrations.07_iq_blobs").IqBlobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="single_qubit")
    parser.add_argument("--qubit", default="q1")
    parser.add_argument("--operation", default="readout")
    parser.add_argument("--num-shots", type=int, default=20000)
    parser.add_argument("--reset-type", default="thermal")
    parser.add_argument("--qubit-operation", default="x180_const")
    parser.add_argument("--qubit-amplitude-factor", type=float, default=1.0)
    parser.add_argument("--pi-repetitions", type=int, default=1)
    parser.add_argument("--states", default="g,e", help="Comma-separated states: g,e or g,e,f.")
    parser.add_argument(
        "--readout-length-ns",
        type=int,
        help="Temporarily override the readout pulse length for this benchmark.",
    )
    parser.add_argument(
        "--flat-integration-weights",
        action="store_true",
        help="Benchmark with a flat [1.0, readout_length] kernel and zero angle.",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--propose-profile-update",
        action="store_true",
        help="Stage fitted threshold/angle updates after the benchmark.",
    )
    parser.add_argument(
        "--apply-profile-update",
        action="store_true",
        help="Prompt to apply fitted threshold/angle updates.",
    )
    parser.add_argument("--auto-connect", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the machine config and QUA program without executing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    parameters = Parameters()
    parameters.num_shots = args.num_shots
    parameters.operation = args.operation
    parameters.states = [state.strip() for state in args.states.split(",") if state.strip()]
    parameters.reset_type = args.reset_type
    parameters.qubit_operation = args.qubit_operation
    parameters.qubit_amplitude_factor = args.qubit_amplitude_factor
    parameters.pi_repetitions = args.pi_repetitions

    options = CalibrationOptions(
        save_raw_data=not args.no_save,
        save_analysis_result=not args.no_save,
        save_figures=bool(args.plot and not args.no_save),
        analyse_data=True,
        plot_data=args.plot,
        update_state=True,
        propose_profile_update=args.propose_profile_update or args.apply_profile_update,
        apply_profile_update=args.apply_profile_update,
        ai_review=False,
    )

    calibration = IqBlobs(
        parameters=parameters,
        profile_name=args.profile,
        qubit=args.qubit,
        options=options,
        auto_connect=args.auto_connect,
    )
    readout = calibration.machine.qubits[args.qubit].resonator.operations[
        args.operation
    ]
    if args.readout_length_ns is not None:
        if args.readout_length_ns <= 0:
            raise ValueError("--readout-length-ns must be positive.")
        readout.length = int(args.readout_length_ns)

    if args.flat_integration_weights:
        readout.integration_weights = [[1.0, int(readout.length)]]
        readout.integration_weights_angle = 0.0
    elif args.readout_length_ns is not None:
        remaining = int(readout.length)
        truncated_weights = []
        for weight, segment_length in readout.integration_weights:
            if remaining <= 0:
                break
            segment = min(int(segment_length), remaining)
            truncated_weights.append([float(weight), segment])
            remaining -= segment
        if remaining:
            raise ValueError(
                "Existing integration weights do not cover requested readout length "
                f"{readout.length} ns."
            )
        readout.integration_weights = truncated_weights

    if args.dry_run:
        calibration.create_qua_program()
        calibration.machine.generate_config()
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "profile": args.profile,
                    "qubit": args.qubit,
                    "operation": args.operation,
                    "states": parameters.states,
                    "readout_length": readout.length,
                    "integration_weight_segments": len(readout.integration_weights),
                    "integration_weights_angle": readout.integration_weights_angle,
                    "flat_integration_weights": args.flat_integration_weights,
                },
                indent=2,
            )
        )
        return 0

    status = calibration.run()
    fit_results = calibration.results.get("fit_results", {})
    qubit_fit = fit_results.get(args.qubit, {})
    summary = {
        "name": status.name,
        "mode": status.mode,
        "outcomes": dict(status.outcomes),
        "readout_fidelity": qubit_fit.get("readout_fidelity"),
        "readout_fidelity_std": qubit_fit.get("readout_fidelity_std"),
        "separation_to_width": qubit_fit.get("separation_to_width"),
        "iw_angle": qubit_fit.get("iw_angle"),
        "ge_threshold": qubit_fit.get("ge_threshold"),
        "run_directory": str(calibration.namespace.get("calibration_run_directory", "")),
        "profile_update_proposed": status.profile_update_proposed,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
