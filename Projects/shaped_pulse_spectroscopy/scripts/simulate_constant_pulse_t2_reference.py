"""Simulate a weak, T2*-limited constant-pulse linewidth reference.

The fixed drive is chosen from the two-level steady-state power-broadening
formula so that its linewidth is 10% above the zero-drive T2* limit.  The
time-domain evolution itself uses the repository's dissipative qutrit model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_DIR = PROJECT_ROOT / "simulation"
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))

from qutrit_slices import simulate_qutrit_slices


DEFAULT_LENGTHS_US = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--power-broadening-ratio",
        type=float,
        default=1.10,
        help="Target CW linewidth divided by the zero-drive T2* limit.",
    )
    parser.add_argument("--detuning-span-mhz", type=float, default=2.0)
    parser.add_argument("--detuning-points", type=int, default=4001)
    parser.add_argument(
        "--steps-per-us",
        type=int,
        default=800,
        help="RK4 steps per microsecond (800 resolves the qutrit anharmonicity).",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Apply a sign flip at the midpoint (off by default for a standard constant pulse).",
    )
    return parser.parse_args()


def interpolated_crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    dy = y - level
    crossings = np.flatnonzero(dy[:-1] * dy[1:] <= 0)
    if not crossings.size:
        return np.nan
    # For a multi-lobed echo spectrum, use the half-height crossing adjacent
    # to the selected peak instead of an outer sidelobe crossing.
    index = int(crossings[-1])
    x0, x1 = float(x[index]), float(x[index + 1])
    y0, y1 = float(y[index]), float(y[index + 1])
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def direct_fwhm(detuning_hz: np.ndarray, signal: np.ndarray) -> tuple[float, float, float]:
    """Return full outer half-height width, contrast, and center frequency."""
    edge = max(5, len(signal) // 20)
    baseline = float(np.median(np.r_[signal[:edge], signal[-edge:]]))
    peak_index = int(np.argmax(signal))
    peak = float(signal[peak_index])
    contrast = peak - baseline
    if contrast <= 0:
        return np.nan, contrast, np.nan
    level = baseline + 0.5 * contrast
    left = interpolated_crossing(detuning_hz[: peak_index + 1], signal[: peak_index + 1], level)
    right_x = detuning_hz[peak_index:]
    right_y = signal[peak_index:]
    right_crossings = np.flatnonzero((right_y[:-1] - level) * (right_y[1:] - level) <= 0)
    if not right_crossings.size:
        return np.nan, contrast, float(detuning_hz[peak_index])
    right_index = int(right_crossings[0])
    x0, x1 = float(right_x[right_index]), float(right_x[right_index + 1])
    y0, y1 = float(right_y[right_index]), float(right_y[right_index + 1])
    right = x0 + (level - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else 0.5 * (x0 + x1)
    return right - left, contrast, float(detuning_hz[peak_index])


def main() -> int:
    args = parse_args()
    if args.power_broadening_ratio <= 1:
        raise ValueError("--power-broadening-ratio must be greater than 1.")
    if args.detuning_points < 101:
        raise ValueError("--detuning-points must be at least 101.")

    campaign_dir = args.campaign_dir.resolve()
    manifest = json.loads((campaign_dir / "manifest.json").read_text(encoding="utf-8"))
    device = manifest["device"]
    t1_us = float(device["t1_s"]) * 1e6
    t2_us = float(device["t2_star_s"]) * 1e6
    t2_limit_hz = 1.0 / (np.pi * float(device["t2_star_s"]))

    # CW linewidth / T2 limit = sqrt(1 + Omega^2 T1 T2), Omega in rad/us.
    omega_rad_per_us = np.sqrt(args.power_broadening_ratio**2 - 1.0) / np.sqrt(t1_us * t2_us)
    rabi_mhz = omega_rad_per_us / (2.0 * np.pi)
    x180_rabi_hz = 1.0 / (2.0 * float(device["x180_length_ns"]) * 1e-9)
    amplitude_v = rabi_mhz * 1e6 / x180_rabi_hz * float(device["x180_amplitude_v"])

    half_span_mhz = 0.5 * args.detuning_span_mhz
    detuning_mhz = np.linspace(-half_span_mhz, half_span_mhz, args.detuning_points)
    detuning_hz = detuning_mhz * 1e6
    lengths = [float(run["pulse_length_us"]) for run in manifest.get("runs", [])]
    if not lengths:
        lengths = list(DEFAULT_LENGTHS_US)

    records: list[dict[str, float | bool]] = []
    traces: dict[float, np.ndarray] = {}
    for length_us in lengths:
        steps = max(400, int(round(0.5 * length_us * args.steps_per_us)))
        print(f"Simulating {length_us:g} us constant pulse ({steps} steps/half)...", flush=True)
        result = simulate_qutrit_slices(
            duration_us=length_us,
            detuning_mhz=detuning_mhz,
            rabi_mhz=np.asarray([rabi_mhz]),
            t1_us=t1_us,
            t2_star_us=t2_us,
            anharmonicity_mhz=-abs(float(device["anharmonicity_hz"])) / 1e6,
            num_steps_per_half=steps,
            cutoff=float(manifest.get("cutoff", 0.005)),
            echo=args.echo,
            pulse_shape="constant",
        )
        signal = np.asarray(result.excited[0] + result.second_excited[0], dtype=float)
        width_hz, contrast, center_hz = direct_fwhm(detuning_hz, signal)
        traces[length_us] = signal
        records.append(
            {
                "pulse_length_us": length_us,
                "fwhm_hz": width_hz,
                "fwhm_t2_units": width_hz / t2_limit_hz,
                "contrast": contrast,
                "peak_detuning_hz": center_hz,
                "constant_amplitude_v": amplitude_v,
                "rabi_frequency_hz": rabi_mhz * 1e6,
                "echo": args.echo,
            }
        )

    output_dir = campaign_dir / "fwhm_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "constant_pulse_t2_limited_reference.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "definition": "CW power broadening is 10% above the zero-drive T2* linewidth",
        "power_broadening_ratio": args.power_broadening_ratio,
        "t1_us": t1_us,
        "t2_star_us": t2_us,
        "t2_limit_hz": t2_limit_hz,
        "constant_amplitude_v": amplitude_v,
        "rabi_frequency_hz": rabi_mhz * 1e6,
        "echo": args.echo,
        "detuning_span_mhz": args.detuning_span_mhz,
        "detuning_points": args.detuning_points,
    }
    json_path = output_dir / "constant_pulse_t2_limited_reference.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    x = np.asarray([float(row["pulse_length_us"]) for row in records])
    y = np.asarray([float(row["fwhm_t2_units"]) for row in records])
    axis.plot(x, y, "o-", color="#b23a48", label="Constant-pulse qutrit simulation")
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2, label=r"$T_2^*$ limit")
    axis.set_xlabel("Pulse length (µs)")
    axis.set_ylabel(r"FWHM / $[1/(\pi T_2^*)]$")
    axis.set_title("T2*-limited constant-pulse linewidth reference")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    plot_path = output_dir / "constant_pulse_t2_limited_fwhm_vs_length.png"
    figure.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"Amplitude: {amplitude_v:.6g} V; Rabi frequency: {rabi_mhz * 1e6:.6g} Hz")
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
