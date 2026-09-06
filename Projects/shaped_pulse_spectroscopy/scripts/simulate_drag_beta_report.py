"""Simulate the exact DRAG-beta report grid with the vectorized qutrit model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_ROOT = PROJECT_ROOT / "simulation"
for path in (PROJECT_ROOT, SIMULATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qutrit_slices import simulate_qutrit_slices
from experiments.drag_beta_kappa_calibration import evaluate_pair


Q6 = {
    "t1_us": 48.82,
    "t2_star_us": 22.79,
    "anharmonicity_mhz": -237.95,
}
SIMULATION_BETA_SIGN = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--steps-per-us", type=int, default=800)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def beta_label(beta: float) -> str:
    return f"{beta:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def simulate_one(payload: dict[str, Any]) -> tuple[float, str]:
    beta = float(payload["beta"])
    simulation_beta = SIMULATION_BETA_SIGN * beta
    output_path = Path(payload["output_path"])
    if output_path.is_file() and not payload["force"]:
        with np.load(output_path) as cached:
            cached_beta = (
                float(cached["simulation_drag_beta"])
                if "simulation_drag_beta" in cached.files
                else np.nan
            )
        if np.isclose(cached_beta, simulation_beta, rtol=0.0, atol=1e-15):
            return beta, str(output_path)
    result = simulate_qutrit_slices(
        duration_us=float(payload["duration_us"]),
        detuning_mhz=np.asarray(payload["detuning_mhz"], dtype=float),
        rabi_mhz=np.asarray(payload["rabi_mhz"], dtype=float),
        t1_us=float(payload["t1_us"]),
        t2_star_us=float(payload["t2_star_us"]),
        anharmonicity_mhz=float(payload["anharmonicity_mhz"]),
        num_steps_per_half=int(payload["num_steps_per_half"]),
        cutoff=float(payload["cutoff"]),
        echo=True,
        pulse_shape="root_lorentzian",
        drag_beta=simulation_beta,
        echo_transition_time_ns=(
            float(payload["transition_time_ns"]) if simulation_beta != 0.0 else 0.0
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        beta=beta,
        simulation_drag_beta=simulation_beta,
        detuning_mhz=np.asarray(payload["detuning_mhz"], dtype=float),
        rabi_mhz=np.asarray(payload["rabi_mhz"], dtype=float),
        ground=result.ground,
        excited=result.excited,
        second_excited=result.second_excited,
        total_excited=result.excited + result.second_excited,
        num_steps_per_half=int(payload["num_steps_per_half"]),
    )
    return beta, str(output_path)


def add_center_fit(path: Path) -> dict[str, Any]:
    with np.load(path) as saved:
        arrays = {name: np.asarray(saved[name]) for name in saved.files}
    detuning_hz = np.asarray(arrays["detuning_mhz"], dtype=float) * 1e6
    rabi_mhz = np.asarray(arrays["rabi_mhz"], dtype=float)
    probability = np.asarray(arrays["total_excited"], dtype=float)
    dataset = xr.Dataset(
        {
            "state": (
                ("qubit", "detuning", "amp_prefactor"),
                probability.T[np.newaxis, ...],
            )
        },
        coords={
            "qubit": ["simulation"],
            "detuning": detuning_hz,
            "amp_prefactor": np.arange(rabi_mhz.size, dtype=float),
        },
    )
    metric = evaluate_pair(
        dataset,
        qubit="simulation",
        rabi_frequency_mhz=rabi_mhz,
    )
    arrays["sim_center_hz"] = np.asarray(metric["center_hz_vs_amplitude"])
    arrays["sim_center_accepted"] = np.asarray(metric["center_fit_accepted"])
    arrays["sim_center_rms_hz"] = np.asarray(metric["center_rms_hz"])
    arrays["sim_weighted_center_hz"] = np.asarray(metric["weighted_center_hz"])
    arrays["sim_spectroscopy_contrast"] = np.asarray(
        metric["spectroscopy_contrast"]
    )
    np.savez_compressed(path, **arrays)
    return metric


def main() -> None:
    args = parse_args()
    if args.steps_per_us < 1 or args.workers < 1:
        raise ValueError("steps-per-us and workers must be positive.")
    run_dir = args.run_dir.resolve()
    plan = json.loads((run_dir / "approved_plan.json").read_text(encoding="utf-8"))
    records = json.loads((run_dir / "records.json").read_text(encoding="utf-8"))
    waveform = plan["waveform"]
    grid = plan["grid"]
    duration_us = float(waveform["duration_ns"]) / 1e3
    span_mhz = float(grid["detuning_span_mhz"])
    detuning_mhz = np.rint(
        np.linspace(-span_mhz * 5e5, span_mhz * 5e5, int(grid["detuning_points"]))
    ) / 1e6
    num_steps_per_half = max(400, round(0.5 * duration_us * args.steps_per_us))
    output_dir = run_dir / "simulation"
    payloads = []
    for record in records:
        beta = float(record["drag_beta"])
        payloads.append(
            {
                "beta": beta,
                "output_path": str(output_dir / f"beta_{beta_label(beta)}.npz"),
                "duration_us": duration_us,
                "detuning_mhz": detuning_mhz.tolist(),
                "rabi_mhz": list(record["rabi_frequency_mhz"]),
                "t1_us": Q6["t1_us"],
                "t2_star_us": Q6["t2_star_us"],
                "anharmonicity_mhz": Q6["anharmonicity_mhz"],
                "num_steps_per_half": num_steps_per_half,
                "cutoff": float(waveform["cutoff"]),
                "transition_time_ns": float(waveform["echo_transition_time_ns"]),
                "force": bool(args.force),
            }
        )

    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(simulate_one, payload) for payload in payloads]
        for future in as_completed(futures):
            beta, path = future.result()
            completed.append({"beta": beta, "path": path})
            print(f"completed beta={beta:g}: {path}", flush=True)

    for item in completed:
        metric = add_center_fit(Path(item["path"]))
        item["sim_center_rms_hz"] = metric["center_rms_hz"]
        item["sim_spectroscopy_contrast"] = metric["spectroscopy_contrast"]

    metadata = {
        "model": "dissipative three-level transmon, vectorized RK4",
        "complex_envelope": "I+iQ",
        "beta_mapping": "simulation_drag_beta = -hardware_drag_beta",
        "simulation_beta_sign": SIMULATION_BETA_SIGN,
        "drag_formula": "Q=-beta*dI/dt/(2*pi*abs(alpha_MHz))",
        "duration_us": duration_us,
        "cutoff": float(waveform["cutoff"]),
        "echo_transition_time_ns": float(waveform["echo_transition_time_ns"]),
        "detuning_points": int(grid["detuning_points"]),
        "amplitude_points": int(grid["amplitude_points"]),
        "steps_per_us": int(args.steps_per_us),
        "num_steps_per_half": int(num_steps_per_half),
        **Q6,
        "results": sorted(completed, key=lambda item: item["beta"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output_dir / "metadata.json")


if __name__ == "__main__":
    main()
