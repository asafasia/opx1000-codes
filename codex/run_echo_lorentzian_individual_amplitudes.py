"""Run fixed-amplitude echo-Lorentzian spectroscopy at selected Rabi amplitudes."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "codex" / "run_echo_lorentzian.py"
LOG_ROOT = REPO_ROOT / "codex" / "individual_amplitude_logs"

for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quam_config import create_machine
from utils.rabi_amplitude import rabi_frequency_hz_to_amplitude


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", default="q1")
    parser.add_argument("--cutoff", type=float, default=0.005)
    parser.add_argument("--pulse-length-us", type=float, default=20.0)
    parser.add_argument("--template-length-us", type=float, default=20.0)
    parser.add_argument("--peak-amplitude", type=float, default=0.2)
    parser.add_argument("--num-shots", type=int, default=100)
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--frequency-span-mhz", type=float, default=1.0)
    parser.add_argument(
        "--rabi-amplitudes-mhz",
        type=float,
        nargs="+",
        default=[2.32, 4.64, 7.58, 11.45],
    )
    parser.add_argument(
        "--echo-modes",
        choices=("echo", "no-echo"),
        nargs="+",
        default=["no-echo", "echo"],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def rabi_mhz_to_amp_factor(
    rabi_mhz: float,
    *,
    qubit_name: str,
    peak_amplitude: float,
) -> tuple[float, float]:
    machine = create_machine(qubit=qubit_name)
    qubit = machine.qubits[qubit_name]
    pi_pulse = qubit.xy.operations["x180"]
    full_amp = float(
        rabi_frequency_hz_to_amplitude(
            rabi_mhz * 1e6,
            float(pi_pulse.amplitude),
            float(pi_pulse.length),
        )
    )
    return full_amp, full_amp / peak_amplitude


def build_command(
    args: argparse.Namespace,
    *,
    echo_mode: str,
    rabi_mhz: float,
    amp_factor: float,
) -> list[str]:
    pulse_length_ns = int(round(args.pulse_length_us * 1000 / 4) * 4)
    template_length_ns = int(round(args.template_length_us * 1000 / 4) * 4)
    amp_step = 1e-6
    frequency_step_mhz = args.frequency_span_mhz / max(args.num_points - 1, 1)
    return [
        sys.executable,
        str(RUNNER),
        "--preset",
        "short",
        "--qubit",
        args.qubit,
        "--pulse-shape",
        "root_lorentzian",
        "--echo" if echo_mode == "echo" else "--no-echo",
        "--state-discrimination",
        "--reset-type",
        "active",
        "--cutoff",
        f"{args.cutoff:.12g}",
        "--num-shots",
        str(args.num_shots),
        "--pulse-length-ns",
        str(pulse_length_ns),
        "--template-length-ns",
        str(template_length_ns),
        "--peak-amplitude",
        f"{args.peak_amplitude:.12g}",
        "--min-amp-factor",
        f"{amp_factor:.12g}",
        "--max-amp-factor",
        f"{amp_factor + amp_step:.12g}",
        "--amp-factor-step",
        f"{amp_step:.12g}",
        "--frequency-span-mhz",
        f"{args.frequency_span_mhz:.12g}",
        "--frequency-step-mhz",
        f"{frequency_step_mhz:.12g}",
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / f"{args.qubit}_fixed_amplitudes_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / "manifest.csv"

    rows = []
    for rabi_mhz in args.rabi_amplitudes_mhz:
        full_amp, amp_factor = rabi_mhz_to_amp_factor(
            rabi_mhz,
            qubit_name=args.qubit,
            peak_amplitude=args.peak_amplitude,
        )
        for echo_mode in args.echo_modes:
            rows.append(
                {
                    "echo_mode": echo_mode,
                    "rabi_mhz": rabi_mhz,
                    "full_amp_v": full_amp,
                    "amp_factor": amp_factor,
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        fieldnames = [
            "index",
            "started_at",
            "finished_at",
            "status",
            "returncode",
            "echo_mode",
            "rabi_mhz",
            "full_amp_v",
            "amp_factor",
            "log_path",
        ]
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Prepared {len(rows)} fixed-amplitude runs.", flush=True)
        for index, row in enumerate(rows, start=1):
            log_path = log_dir / f"run_{index:03d}_{row['echo_mode']}_{row['rabi_mhz']:.2f}MHz.log"
            command = build_command(
                args,
                echo_mode=row["echo_mode"],
                rabi_mhz=float(row["rabi_mhz"]),
                amp_factor=float(row["amp_factor"]),
            )
            started_at = datetime.now().astimezone().isoformat()
            print(
                f"[{index}/{len(rows)}] {row['echo_mode']} "
                f"{row['rabi_mhz']:.2f} MHz "
                f"amp_factor={row['amp_factor']:.6g}",
                flush=True,
            )
            if args.dry_run:
                returncode = 0
                status = "dry-run"
                finished_at = datetime.now().astimezone().isoformat()
            else:
                with log_path.open("w", encoding="utf-8") as log_file:
                    log_file.write(" ".join(command) + "\n\n")
                    log_file.flush()
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                returncode = completed.returncode
                status = "ok" if returncode == 0 else "failed"
                finished_at = datetime.now().astimezone().isoformat()

            writer.writerow(
                {
                    "index": index,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": status,
                    "returncode": returncode,
                    "log_path": str(log_path),
                    **row,
                }
            )
            manifest_file.flush()

    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
