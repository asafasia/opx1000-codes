"""Readout Fidelity Optimization for Qubit 1A.

Sweeps readout amplitude across candidate operating points:
  [0.040, 0.050, 0.060, 0.070, 0.080] V
Using the calibrated hardware integration weights rotation angle (iw_angle = 2.7263 rad / -156.2°).

Acquires 10,000 shots per state (|g> and |e>) for each amplitude.
Extracts:
  - Center separation (Delta V) [µV]
  - Separation-to-width ratio (d/sigma)
  - Confusion matrix
  - Assignment fidelity F = (P(g|g) + P(e|e)) / 2
Saves publication-quality figures and updates Or_profile on branch user/Or.
"""

from dataclasses import replace
from datetime import datetime
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

from qm.qua import (
    declare,
    declare_output_stream,
    fixed,
    for_,
    program,
    save,
    stream_processing,
    align,
)
from qualang_tools.units import unit
from quam_config import create_machine
from calibration_io import CalibrationSaver


def build_iq_blobs_prog(machine, num_shots: int, thermalization_cycles: int):
    q = machine.qubits["q1a"]
    with program() as iq_prog:
        n = declare(int)
        I_g = declare(fixed)
        Q_g = declare(fixed)
        I_e = declare(fixed)
        Q_e = declare(fixed)

        I_g_st = declare_output_stream()
        Q_g_st = declare_output_stream()
        I_e_st = declare_output_stream()
        Q_e_st = declare_output_stream()

        # Ground state shots
        with for_(n, 0, n < num_shots, n + 1):
            q.align()
            # Thermal reset wait
            q.resonator.wait(thermalization_cycles)
            q.xy.wait(thermalization_cycles)
            q.align()
            q.resonator.measure("readout", qua_vars=(I_g, Q_g))
            save(I_g, I_g_st)
            save(Q_g, Q_g_st)

        # Excited state shots
        with for_(n, 0, n < num_shots, n + 1):
            q.align()
            q.resonator.wait(thermalization_cycles)
            q.xy.wait(thermalization_cycles)
            q.align()
            q.xy.play("x180")
            q.align()
            q.resonator.measure("readout", qua_vars=(I_e, Q_e))
            save(I_e, I_e_st)
            save(Q_e, Q_e_st)

        with stream_processing():
            I_g_st.save_all("I_g")
            Q_g_st.save_all("Q_g")
            I_e_st.save_all("I_e")
            Q_e_st.save_all("Q_e")

    return iq_prog


def analyze_blobs(Ig_v, Qg_v, Ie_v, Qe_v):
    """Compute centers, rotation, threshold, confusion matrix, and fidelity."""
    c_g = np.array([np.mean(Ig_v), np.mean(Qg_v)])
    c_e = np.array([np.mean(Ie_v), np.mean(Qe_v)])
    delta = c_e - c_g
    sep = np.linalg.norm(delta)

    # Angle to rotate separation vector onto X (Rotated-I)
    theta = np.arctan2(delta[1], delta[0])

    def rot(I, Q):
        return I * np.cos(-theta) - Q * np.sin(-theta)

    Ig_rot = rot(Ig_v - c_g[0], Qg_v - c_g[1])
    Ie_rot = rot(Ie_v - c_g[0], Qe_v - c_g[1])

    mean_g_rot = np.mean(Ig_rot)
    mean_e_rot = np.mean(Ie_rot)
    std_g = np.std(Ig_rot)
    std_e = np.std(Ie_rot)
    avg_std = (std_g + std_e) / 2.0
    sep_width = sep / avg_std if avg_std > 0 else 0.0

    # Optimal threshold separating the 1D rotated distributions
    threshold = (mean_g_rot + mean_e_rot) / 2.0

    # Decision: state 1 (e) if > threshold (if mean_e > mean_g), else < threshold
    if mean_e_rot >= mean_g_rot:
        p_g_g = np.mean(Ig_rot < threshold)
        p_e_g = np.mean(Ig_rot >= threshold)
        p_g_e = np.mean(Ie_rot < threshold)
        p_e_e = np.mean(Ie_rot >= threshold)
    else:
        p_g_g = np.mean(Ig_rot >= threshold)
        p_e_g = np.mean(Ig_rot < threshold)
        p_g_e = np.mean(Ie_rot >= threshold)
        p_e_e = np.mean(Ie_rot < threshold)

    fidelity = (p_g_g + p_e_e) / 2.0

    return {
        "c_g": c_g,
        "c_e": c_e,
        "separation_v": float(sep),
        "separation_width": float(sep_width),
        "theta_rad": float(theta),
        "theta_deg": float(np.rad2deg(theta)),
        "std_g": float(std_g),
        "std_e": float(std_e),
        "threshold_rot": float(threshold),
        "p_g_g": float(p_g_g),
        "p_e_g": float(p_e_g),
        "p_g_e": float(p_g_e),
        "p_e_e": float(p_e_e),
        "fidelity": float(fidelity),
        "confusion_matrix": [[float(p_g_g), float(p_e_g)], [float(p_g_e), float(p_e_e)]],
        "Ig_rot": Ig_rot,
        "Ie_rot": Ie_rot,
    }


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("c:/Users/owner/Developer/opx1000_Or/data/new_chip_experiments/13_q1_iq_blobs_optimization") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting Qubit 1A Readout Amplitude Optimization ===", flush=True)
    t0 = time.time()

    machine = create_machine(profile_name="Or_profile", qubit="q1a")
    q = machine.qubits["q1a"]

    qmm = machine.connect()
    u = unit()

    readout_len_ns = q.resonator.operations["readout"].length
    thermal_cycles = int(150_000 // 4)  # 150 us in 4ns clock cycles
    num_shots = 10000

    amps_to_test = [0.070, 0.080, 0.090, 0.100, 0.115, 0.130]
    results = {}
    best_amp = None
    best_fid = -1.0

    for amp in amps_to_test:
        print(f"\n--- Testing Readout Amplitude = {amp:.3f} V ---", flush=True)
        q.resonator.operations["readout"].amplitude = float(amp)

        config = machine.generate_config()
        prog = build_iq_blobs_prog(machine, num_shots, thermal_cycles)

        qm = qmm.open_qm(config)
        job = qm.execute(prog)

        handle = job.result_handles
        handle.wait_for_all_values()

        Ig_raw = handle.get("I_g").fetch_all()
        Qg_raw = handle.get("Q_g").fetch_all()
        Ie_raw = handle.get("I_e").fetch_all()
        Qe_raw = handle.get("Q_e").fetch_all()

        def extract_vals(data):
            if hasattr(data, "dtype") and data.dtype.names and "value" in data.dtype.names:
                return data["value"]
            return data

        Ig_v = u.demod2volts(extract_vals(Ig_raw), readout_len_ns)
        Qg_v = u.demod2volts(extract_vals(Qg_raw), readout_len_ns)
        Ie_v = u.demod2volts(extract_vals(Ie_raw), readout_len_ns)
        Qe_v = u.demod2volts(extract_vals(Qe_raw), readout_len_ns)

        analysis = analyze_blobs(Ig_v, Qg_v, Ie_v, Qe_v)
        fid_pct = analysis["fidelity"] * 100
        sep_uv = analysis["separation_v"] * 1e6

        print(f"    Separation: {sep_uv:.1f} µV | Sep/Width: {analysis['separation_width']:.3f}", flush=True)
        print(f"    Confusion: |g>->|g>: {analysis['p_g_g']*100:.1f}%, |e>->|e>: {analysis['p_e_e']*100:.1f}%", flush=True)
        print(f"    Readout Fidelity: {fid_pct:.2f}% (Residual Angle: {analysis['theta_deg']:.1f}°)", flush=True)

        results[amp] = {
            "analysis": analysis,
            "Ig_v": Ig_v,
            "Qg_v": Qg_v,
            "Ie_v": Ie_v,
            "Qe_v": Qe_v,
        }

        if analysis["fidelity"] > best_fid:
            best_fid = analysis["fidelity"]
            best_amp = amp

    print(f"\n==========================================", flush=True)
    print(f"OPTIMAL READOUT AMPLITUDE: {best_amp:.3f} V", flush=True)
    print(f"MAX READOUT FIDELITY:      {best_fid*100:.2f}%", flush=True)
    print(f"==========================================", flush=True)

    # 4-Panel Summary Publication Plot
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Panel 1: Best Cloud Scatter + Contours
    ax_clouds = fig.add_subplot(gs[0, 0])
    best_data = results[best_amp]
    subsample = 2500  # Subsample points for clear scatter
    ax_clouds.scatter(best_data["Ig_v"][:subsample]*1e3, best_data["Qg_v"][:subsample]*1e3,
                      s=4, alpha=0.25, color="#1f77b4", label=r"Ground $|g\rangle$")
    ax_clouds.scatter(best_data["Ie_v"][:subsample]*1e3, best_data["Qe_v"][:subsample]*1e3,
                      s=4, alpha=0.25, color="#ff7f0e", label=r"Excited $|e\rangle$")

    # Plot centers
    c_g = best_data["analysis"]["c_g"] * 1e3
    c_e = best_data["analysis"]["c_e"] * 1e3
    ax_clouds.plot([c_g[0], c_e[0]], [c_g[1], c_e[1]], "k--", lw=1.5, label="Separation Vector")
    ax_clouds.scatter([c_g[0]], [c_g[1]], c="blue", s=80, edgecolors="k", zorder=5)
    ax_clouds.scatter([c_e[0]], [c_e[1]], c="red", s=80, edgecolors="k", zorder=5)

    ax_clouds.set_title(f"Optimal Single-Shot IQ Clouds (Amp = {best_amp:.3f} V, F = {best_fid*100:.1f}%)", fontweight="bold")
    ax_clouds.set_xlabel("I [mV]")
    ax_clouds.set_ylabel("Q [mV]")
    ax_clouds.grid(True, alpha=0.3)
    ax_clouds.legend(loc="upper right", fontsize=9)

    # Panel 2: 1D Rotated Histograms & Threshold
    ax_hist = fig.add_subplot(gs[0, 1])
    Ig_rot_mv = best_data["analysis"]["Ig_rot"] * 1e3
    Ie_rot_mv = best_data["analysis"]["Ie_rot"] * 1e3
    th_mv = best_data["analysis"]["threshold_rot"] * 1e3

    bins = np.linspace(min(np.min(Ig_rot_mv), np.min(Ie_rot_mv)), max(np.max(Ig_rot_mv), np.max(Ie_rot_mv)), 80)
    ax_hist.hist(Ig_rot_mv, bins=bins, alpha=0.6, color="#1f77b4", density=True, label=r"Ground $|g\rangle$")
    ax_hist.hist(Ie_rot_mv, bins=bins, alpha=0.6, color="#ff7f0e", density=True, label=r"Excited $|e\rangle$")
    ax_hist.axvline(th_mv, color="crimson", ls="--", lw=2, label=f"Threshold = {th_mv:.2f} mV")
    ax_hist.set_title(f"Projected Rotated-I Distributions & Threshold", fontweight="bold")
    ax_hist.set_xlabel("Rotated-I [mV]")
    ax_hist.set_ylabel("Probability Density")
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend(loc="upper right", fontsize=9)

    # Panel 3: Separation-to-Width & Center Separation vs Amplitude
    ax_sep = fig.add_subplot(gs[1, 0])
    amps_arr = np.array(amps_to_test)
    seps_uv = [results[a]["analysis"]["separation_v"] * 1e6 for a in amps_to_test]
    sep_widths = [results[a]["analysis"]["separation_width"] for a in amps_to_test]

    ax_sep.plot(amps_arr, seps_uv, "o-", color="#1f77b4", lw=1.8, label="Center Separation [µV]")
    ax_sep.set_xlabel("Readout Amplitude [V]")
    ax_sep.set_ylabel("Center Separation [µV]", color="#1f77b4")
    ax_sep.tick_params(axis="y", labelcolor="#1f77b4")
    ax_sep.grid(True, alpha=0.3)

    ax_sep_r = ax_sep.twinx()
    ax_sep_r.plot(amps_arr, sep_widths, "s--", color="#2ca02c", lw=1.8, label="Sep / Width Ratio")
    ax_sep_r.set_ylabel("Separation / Width Ratio", color="#2ca02c")
    ax_sep_r.tick_params(axis="y", labelcolor="#2ca02c")
    ax_sep.set_title("Cloud Separation vs. Readout Power", fontweight="bold")

    # Panel 4: Readout Fidelity vs Amplitude & Confusion Matrix
    ax_fid = fig.add_subplot(gs[1, 1])
    fids = [results[a]["analysis"]["fidelity"] * 100 for a in amps_to_test]
    p_gg = [results[a]["analysis"]["p_g_g"] * 100 for a in amps_to_test]
    p_ee = [results[a]["analysis"]["p_e_e"] * 100 for a in amps_to_test]

    ax_fid.plot(amps_arr, fids, "o-", color="purple", lw=2.2, label=f"Average Fidelity (Max {best_fid*100:.1f}%)")
    ax_fid.plot(amps_arr, p_gg, "^--", color="#1f77b4", lw=1.5, label=r"$P(g|g)$ Ground Accuracy")
    ax_fid.plot(amps_arr, p_ee, "v--", color="#ff7f0e", lw=1.5, label=r"$P(e|e)$ Excited Accuracy")
    ax_fid.axvline(best_amp, color="gold", ls=":", lw=2, label=f"Optimal Amp = {best_amp:.3f} V")

    ax_fid.set_xlabel("Readout Amplitude [V]")
    ax_fid.set_ylabel("Fidelity [%]")
    ax_fid.set_title("Single-Shot Readout Fidelity Optimization", fontweight="bold")
    ax_fid.grid(True, alpha=0.3)
    ax_fid.legend(loc="lower right", fontsize=9)

    fig_path = out_dir / "readout_amplitude_optimization_summary.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    # Save to brain artifact directory
    brain_fig = Path(r"C:\Users\owner\.gemini\antigravity\brain\b64907bd-bf8f-47ef-8f88-93815c70c67e\13_q1_iq_blobs_optimization.png")
    shutil.copy2(fig_path, brain_fig)

    # Save metadata
    meta = {
        "timestamp": timestamp,
        "best_amplitude": float(best_amp),
        "best_fidelity": float(best_fid),
        "tested_amplitudes": amps_to_test,
        "results_per_amp": {
            str(a): {
                "separation_uv": float(results[a]["analysis"]["separation_v"] * 1e6),
                "separation_width": float(results[a]["analysis"]["separation_width"]),
                "fidelity_pct": float(results[a]["analysis"]["fidelity"] * 100),
                "confusion_matrix": results[a]["analysis"]["confusion_matrix"],
            }
            for a in amps_to_test
        },
        "total_elapsed_s": time.time() - t0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Update Or_profile with the best amplitude if it improved fidelity
    if best_amp != 0.05:
        p_path = REPO_ROOT / "profiles" / "Or_profile" / "pulses.json"
        with open(p_path) as f:
            p_data = json.load(f)
        p_data["pulses"]["q1a"]["readout"]["amplitude"] = float(best_amp)
        with open(p_path, "w") as f:
            json.dump(p_data, f, indent=2)

        q_path = REPO_ROOT / "profiles" / "Or_profile" / "qubits.json"
        with open(q_path) as f:
            q_data = json.load(f)
        q_data["qubits"]["q1a"]["readout"]["confusion_matrix"] = results[best_amp]["analysis"]["confusion_matrix"]
        with open(q_path, "w") as f:
            json.dump(q_data, f, indent=2)

        m_path = REPO_ROOT / "profiles" / "Or_profile" / "metrics.json"
        with open(m_path) as f:
            m_data = json.load(f)
        m_data["qubits"]["q1a"]["readout"]["fidelity_percent"]["thermal"] = float(best_fid * 100)
        with open(m_path, "w") as f:
            json.dump(m_data, f, indent=2)
        print(f"\nApplied optimal readout amplitude {best_amp:.3f} V to Or_profile (branch user/Or).", flush=True)

    print(f"\nAll complete in {time.time()-t0:.1f} s! Figure saved to: {fig_path}", flush=True)


if __name__ == "__main__":
    main()
