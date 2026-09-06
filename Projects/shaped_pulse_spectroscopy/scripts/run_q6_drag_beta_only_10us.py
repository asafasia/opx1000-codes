"""Run the explicitly approved q6 10 us DRAG-only spectroscopy sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.drag_beta_kappa_calibration import (
    SpectroscopyGrid,
    WaveformSpec,
    beta_only_plan,
    parameters_for_point,
    plan_sha256,
    run_approved_plan,
)
from experiments.detuning_amplitude_sweep import EchoLorentzian
from calibrations.base import CalibrationOptions
from quam_config import create_machine


PLAN = beta_only_plan(
    target_qubit="q6",
    betas=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
    waveform=WaveformSpec(
        pulse_shape="root_lorentzian",
        duration_ns=10_000,
        template_length_ns=10_000,
        cutoff=0.005,
        peak_amplitude_v=0.7,
        echo_transition_time_ns=16.0,
    ),
    grid=SpectroscopyGrid(
        detuning_span_mhz=2.0,
        detuning_points=200,
        min_amp_factor=0.01,
        max_amp_factor=1.0,
        amplitude_points=200,
        amplitude_spacing="log",
        num_shots=200,
    ),
)


def validate_locally() -> dict[str, object]:
    """Build the worst-case waveform and emit a complete offline QUA script."""
    from qm import QuantumMachinesManager, generate_qua_script

    QuantumMachinesManager.set_capabilities_offline()
    worst_point = max(PLAN.points, key=lambda point: abs(point.drag_beta))
    parameters = parameters_for_point(PLAN, worst_point)
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
    qua_script = generate_qua_script(qua_program, calibration.machine.generate_config())
    metrics = calibration.namespace["lorentzian_waveform_metrics"][PLAN.target_qubit]
    return {
        "qua_script_generated": bool(qua_script),
        "worst_beta": worst_point.drag_beta,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    approved_hash = plan_sha256(PLAN)
    print(f"Approved plan SHA256: {approved_hash}", flush=True)
    if args.validate_only:
        print(json.dumps(validate_locally(), indent=2), flush=True)
        return
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
