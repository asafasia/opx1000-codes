"""Run two q6 pulse-shaped spectroscopy maps at beta=0 and beta=-0.22.

The hardware run is intentionally opt-in: without ``--execute`` this script
only builds both waveforms and their QUA programs offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrations.base import CalibrationOptions
from experiments.detuning_amplitude_sweep import EchoLorentzian
from experiments.drag_beta_kappa_calibration import (
    CalibrationPlan,
    CalibrationPoint,
    SpectroscopyGrid,
    WaveformSpec,
    parameters_for_point,
    plan_sha256,
    run_approved_plan,
)
from quam_config import create_machine


TARGET_QUBIT = "q6"
BETAS_IN_RUN_ORDER = (0.0, -0.22)

PLAN = CalibrationPlan(
    stage="joint",
    target_qubit=TARGET_QUBIT,
    waveform=WaveformSpec(
        pulse_shape="root_lorentzian",
        duration_ns=50_000,
        template_length_ns=50_000,
        cutoff=0.00075,
        peak_amplitude_v=1.0,
        echo_transition_time_ns=16.0,
    ),
    grid=SpectroscopyGrid(
        # The grid spans -0.2 MHz through +0.2 MHz, inclusively.
        detuning_span_mhz=0.4,
        detuning_points=200,
        # With a 1 V waveform peak these factors are also volts.
        min_amp_factor=0.0,
        max_amp_factor=1.0,
        amplitude_points=200,
        amplitude_spacing="linear",
        num_shots=2_000,
    ),
    points=(
        CalibrationPoint(
            label="beta_0",
            category="no_correction",
            drag_beta=0.0,
            stark_kappa_mhz_inv=0.0,
            ac_stark_correction=False,
        ),
        CalibrationPoint(
            label="drag_beta_-0.22",
            category="drag_same_kappa",
            drag_beta=-0.22,
            stark_kappa_mhz_inv=0.0,
            ac_stark_correction=False,
        ),
    ),
)


def validate_locally() -> dict[str, object]:
    """Build both exact waveforms and emit both QUA programs offline."""
    from qm import QuantumMachinesManager, generate_qua_script

    QuantumMachinesManager.set_capabilities_offline()
    validations: list[dict[str, object]] = []
    for point in PLAN.points:
        parameters = parameters_for_point(PLAN, point)
        calibration = EchoLorentzian(
            parameters=parameters,
            options=CalibrationOptions(
                save_raw_data=False,
                save_analysis_result=False,
                save_figures=False,
                analyse_data=False,
                plot_data=False,
                update_state=False,
                propose_profile_update=False,
                apply_profile_update=False,
                ai_review=False,
                report_runtime_estimate=False,
            ),
            machine=create_machine(qubit=PLAN.target_qubit),
            qubit=PLAN.target_qubit,
            auto_connect=False,
        )
        qua_program = calibration.create_qua_program()
        qua_script = generate_qua_script(
            qua_program,
            calibration.machine.generate_config(),
        )
        metrics = calibration.namespace["lorentzian_waveform_metrics"][
            PLAN.target_qubit
        ]
        axes = calibration.namespace["sweep_axes"]
        validations.append(
            {
                "label": point.label,
                "beta": point.drag_beta,
                "kappa_mhz_inv": point.stark_kappa_mhz_inv,
                "qua_script_generated": bool(qua_script),
                "detuning_hz": {
                    "first": int(axes["detuning"].values[0]),
                    "last": int(axes["detuning"].values[-1]),
                    "points": int(axes["detuning"].size),
                },
                "amplitude_v": {
                    "first": float(axes["amp_prefactor"].values[0]),
                    "last": float(axes["amp_prefactor"].values[-1]),
                    "points": int(axes["amp_prefactor"].size),
                },
                **metrics,
            }
        )

    return {
        "plan": asdict(PLAN),
        "plan_sha256": plan_sha256(PLAN),
        "beta_run_order": list(BETAS_IN_RUN_ORDER),
        "validations": validations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Connect to q6 and acquire both hardware sweeps.",
    )
    args = parser.parse_args()

    validation = validate_locally()
    print(json.dumps(validation, indent=2), flush=True)
    if not args.execute:
        print(
            "Offline validation complete; pass --execute to run hardware.",
            flush=True,
        )
        return

    approved_hash = plan_sha256(PLAN)
    result = run_approved_plan(PLAN, approved_plan_sha256=approved_hash)
    print(
        json.dumps(
            {
                "output_dir": str(result["output_dir"]),
                "plan_sha256": result["plan_sha256"],
                "records": result["records"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
