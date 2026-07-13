"""Train optimized readout weights using the repository QuAM machine.

This is the current-machine version of the original use-case training script.
It builds the same generated config used by the calibration suite, runs sliced
readout-weight optimization, and saves the kernel under:

    profiles/<profile>/kernels/<qubit>_readout_kernel.npz
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

from calibration_utils.readout_weights_optimization import Parameters
from calibrations.core import CalibrationOptions


ReadoutWeightsOptimization = importlib.import_module(
    "calibrations.10d_readout_weights_optimization"
).ReadoutWeightsOptimization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="single_qubit")
    parser.add_argument("--qubit", default="q1")
    parser.add_argument("--operation", default="readout")
    parser.add_argument("--num-shots", type=int, default=100)
    parser.add_argument(
        "--division-length-clock-cycles",
        type=int,
        default=10,
        help="Sliced-demod chunk length. One QUA clock cycle is 4 ns.",
    )
    parser.add_argument(
        "--use-current-integration-weights",
        action="store_true",
        help="Measure sliced traces with the profile's current weights instead of a flat kernel.",
    )
    parser.add_argument("--xy-to-readout-delay-ns", type=int, default=100)
    parser.add_argument("--reset-type", default="thermal")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--propose-profile-update",
        action="store_true",
        help="Stage readout.use_kernel=true after saving the kernel.",
    )
    parser.add_argument(
        "--apply-profile-update",
        action="store_true",
        help="Prompt to apply the staged profile update.",
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
    parameters.division_length_clock_cycles = args.division_length_clock_cycles
    parameters.use_current_integration_weights = args.use_current_integration_weights
    parameters.xy_to_readout_delay_in_ns = args.xy_to_readout_delay_ns
    parameters.reset_type = args.reset_type

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

    calibration = ReadoutWeightsOptimization(
        parameters=parameters,
        profile_name=args.profile,
        qubit=args.qubit,
        options=options,
        auto_connect=args.auto_connect,
    )
    if args.dry_run:
        calibration.create_qua_program()
        readout = calibration.machine.qubits[args.qubit].resonator.operations[
            args.operation
        ]
        calibration.machine.generate_config()
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "profile": args.profile,
                    "qubit": args.qubit,
                    "operation": args.operation,
                    "readout_length": readout.length,
                    "integration_weight_segments": len(readout.integration_weights),
                    "slice_length_ns": calibration.slice_length_ns,
                    "number_of_divisions": calibration.namespace.get(
                        "number_of_divisions"
                    ),
                },
                indent=2,
            )
        )
        return 0

    status = calibration.run()
    summary = {
        "name": status.name,
        "mode": status.mode,
        "outcomes": dict(status.outcomes),
        "raw_data_saved": status.raw_data_saved,
        "kernel_artifact_directory": str(
            calibration.namespace.get("kernel_artifact_directory", "")
        ),
        "run_directory": str(calibration.namespace.get("calibration_run_directory", "")),
        "profile_update_proposed": status.profile_update_proposed,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
