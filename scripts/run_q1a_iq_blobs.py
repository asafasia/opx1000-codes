"""Run 07_iq_blobs for Qubit 1A on the new chip.

Calibrates:
- Optimal rotation angle (theta) aligning |g> - |e> separation onto Rotated-I.
- Optimal discrimination threshold V_th.
- Confusion matrix and single-shot readout fidelity.
"""

import importlib
import json
from pathlib import Path
import shutil
import sys
import time

REPO_ROOT = Path("c:/Users/owner/Developer/opx1000-codes")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibrations.core import CalibrationOptions
from calibration_utils.iq_blobs import Parameters
from quam_config import create_machine


def main():
    print("=== Launching Experiment: 07_iq_blobs for Qubit 1A ===", flush=True)
    t0 = time.time()

    machine = create_machine(profile_name="Or_profile", qubit="q1a")
    q = machine.qubits["q1a"]

    print(f"Target Qubit: {q.name}", flush=True)
    print(f"Qubit Frequency: {q.f_01/1e9:.6f} GHz (Drive LO: {q.xy.LO_frequency/1e9:.3f} GHz, IF: {q.xy.intermediate_frequency/1e6:.1f} MHz)", flush=True)
    print(f"Readout Frequency: {q.resonator.f_01/1e9:.6f} GHz (Readout LO: {q.resonator.LO_frequency/1e9:.3f} GHz, IF: {q.resonator.intermediate_frequency/1e6:.1f} MHz)", flush=True)
    print(f"Pi Pulse: {q.xy.operations['x180'].length} ns, amp = {q.xy.operations['x180'].amplitude}", flush=True)
    print(f"Readout Pulse: {q.resonator.operations['readout'].length} ns, amp = {q.resonator.operations['readout'].amplitude}", flush=True)

    num_shots = 10000
    parameters = Parameters(
        num_shots=num_shots,
        reset_type="thermal",
        readout_states=["g", "e"],
        pi_repetitions=1,
        center_method="mean",
    )

    cls = importlib.import_module("calibrations.07_iq_blobs").IqBlobs
    options = CalibrationOptions(
        save_raw_data=True,
        save_analysis_result=True,
        save_figures=True,
        analyse_data=True,
        plot_data=True,
        update_state=True,
        propose_profile_update=True,
        apply_profile_update=True,
    )

    calibration = cls(
        parameters=parameters,
        machine=machine,
        profile_name="Or_profile",
        qubit="q1a",
        options=options,
    )

    print(f"\nAcquiring {num_shots} shots per state (|g> and |e>) with thermal reset...", flush=True)
    status = calibration.run()

    elapsed = time.time() - t0
    print(f"\nCalibration finished in {elapsed:.2f} s. Status: {status}", flush=True)

    run_dir = Path(calibration.namespace.get("calibration_run_directory", ""))
    print(f"Saved directory: {run_dir}", flush=True)

    # Print fit results
    fit_res = calibration.results.get("fit_results", {}).get("q1a", {})
    if fit_res:
        print("\n=== Fitted Calibration Results ===", flush=True)
        print(f"Fitted Rotation Angle (IW Angle): {np.rad2deg(fit_res.get('iw_angle', 0.0)):.2f}° ({fit_res.get('iw_angle', 0.0):.4f} rad)", flush=True)
        print(f"GE Threshold: {fit_res.get('ge_threshold', 0.0):.6f}", flush=True)
        print(f"Fidelity: {fit_res.get('fidelity', 0.0)*100:.2f}%", flush=True)
        conf_mat = fit_res.get("confusion_matrix") or fit_res.get("fidelity_matrix")
        if conf_mat is not None:
            print("Confusion Matrix:", flush=True)
            print(np.array(conf_mat), flush=True)

    # Copy figure to artifacts and brain directory
    figures = calibration.results.get("figures", {})
    brain_dir = Path(r"C:\Users\owner\.gemini\antigravity\brain\b64907bd-bf8f-47ef-8f88-93815c70c67e")
    new_chip_exp_dir = Path(r"C:\Users\owner\Developer\opx1000_Or\data\new_chip_experiments\12_q1_iq_blobs")
    new_chip_exp_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        dst_brain = brain_dir / f"12_q1_iq_blobs_{name}.png"
        dst_new_chip = new_chip_exp_dir / f"{name}.png"
        fig.savefig(dst_brain, dpi=180)
        fig.savefig(dst_new_chip, dpi=180)
        print(f"Saved figure artifact: {dst_brain}", flush=True)


if __name__ == "__main__":
    main()
