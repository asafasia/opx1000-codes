"""Run overnight q1 echo-Lorentzian sweeps until a fixed stop time.

The loop launches one calibration process per parameter point so a failed point
is logged and the next one can continue.  Defaults match the June 28 overnight
request: q1, active reset, both echo modes, 40 averages, 150 amplitude points,
four frequency spans, cutoffs from 0.99 to 1e-5, and pulse lengths from 2 us to
150 us.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "codex" / "run_echo_lorentzian.py"
LOG_ROOT = REPO_ROOT / "codex" / "overnight_logs"


def geomspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    log_start = math.log(start)
    log_stop = math.log(stop)
    return [
        math.exp(log_start + (log_stop - log_start) * i / (count - 1))
        for i in range(count)
    ]


def rounded_ns_values(start_us: float, stop_us: float, count: int) -> list[int]:
    values = []
    for value_us in geomspace(start_us, stop_us, count):
        value_ns = int(round(value_us * 1000 / 4) * 4)
        if value_ns not in values:
            values.append(value_ns)
    if int(stop_us * 1000) not in values:
        values.append(int(round(stop_us * 1000 / 4) * 4))
    return values


def next_stop_time(hour: int) -> datetime:
    now = datetime.now().astimezone()
    stop = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if stop <= now:
        stop += timedelta(days=1)
    return stop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", default="q1")
    parser.add_argument("--stop-hour", type=int, default=10)
    parser.add_argument("--cutoff-count", type=int, default=8)
    parser.add_argument("--length-count", type=int, default=8)
    parser.add_argument("--num-shots", type=int, default=40)
    parser.add_argument("--amp-points", type=int, default=150)
    parser.add_argument("--min-amp-factor", type=float, default=0.0)
    parser.add_argument("--max-amp-factor", type=float, default=1.0)
    parser.add_argument("--peak-amplitude", type=float, default=0.5)
    parser.add_argument("--max-template-length-ns", type=int, default=60000)
    parser.add_argument(
        "--echo-modes",
        choices=("true", "false"),
        nargs="+",
        default=["true", "false"],
    )
    parser.add_argument(
        "--frequency-spans-mhz",
        type=float,
        nargs="+",
        default=[500.0, 50.0, 5.0, 1.0],
        help="Broad, medium, narrow sweep spans in MHz.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stop_time = next_stop_time(args.stop_hour)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / f"echo_lorentzian_q1_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / "manifest.csv"

    cutoffs = geomspace(0.99, 1e-5, args.cutoff_count)
    lengths_ns = rounded_ns_values(2.0, 150.0, args.length_count)
    amp_step = (args.max_amp_factor - args.min_amp_factor) / args.amp_points

    rows = []
    for cutoff in cutoffs:
        for length_ns in lengths_ns:
            template_ns = min(length_ns, args.max_template_length_ns)
            template_ns = max(4, int(round(template_ns / 4) * 4))
            for span_mhz in args.frequency_spans_mhz:
                for echo_mode in args.echo_modes:
                    freq_step_mhz = span_mhz / (args.amp_points - 1)
                    rows.append(
                        {
                            "echo": echo_mode == "true",
                            "cutoff": cutoff,
                            "pulse_length_ns": length_ns,
                            "template_length_ns": template_ns,
                            "frequency_span_mhz": span_mhz,
                            "frequency_step_mhz": freq_step_mhz,
                        }
                    )

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "index",
                "started_at",
                "finished_at",
                "status",
                "returncode",
                "echo",
                "cutoff",
                "pulse_length_ns",
                "template_length_ns",
                "frequency_span_mhz",
                "frequency_step_mhz",
                "log_path",
            ],
        )
        writer.writeheader()

        print(
            f"Prepared {len(rows)} runs. Stop time: {stop_time.isoformat()}",
            flush=True,
        )
        for index, row in enumerate(rows, start=1):
            now = datetime.now().astimezone()
            if now >= stop_time:
                print("Reached stop time before starting next run.", flush=True)
                break

            log_path = log_dir / f"run_{index:04d}.log"
            command = [
                sys.executable,
                str(RUNNER),
                "--preset",
                "short",
                "--qubit",
                args.qubit,
                "--pulse-shape",
                "root_lorentzian",
                "--echo" if row["echo"] else "--no-echo",
                "--state-discrimination",
                "--reset-type",
                "active",
                "--cutoff",
                f"{row['cutoff']:.12g}",
                "--num-shots",
                str(args.num_shots),
                "--pulse-length-ns",
                str(row["pulse_length_ns"]),
                "--template-length-ns",
                str(row["template_length_ns"]),
                "--peak-amplitude",
                str(args.peak_amplitude),
                "--min-amp-factor",
                str(args.min_amp_factor),
                "--max-amp-factor",
                str(args.max_amp_factor),
                "--amp-factor-step",
                f"{amp_step:.12g}",
                "--frequency-span-mhz",
                str(row["frequency_span_mhz"]),
                "--frequency-step-mhz",
                f"{row['frequency_step_mhz']:.12g}",
            ]

            started_at = datetime.now().astimezone().isoformat()
            print(
                f"[{index}/{len(rows)}] echo={row['echo']} "
                f"cutoff={row['cutoff']:.4g} "
                f"length={row['pulse_length_ns'] / 1000:.3g} us "
                f"span={row['frequency_span_mhz']:.4g} MHz",
                flush=True,
            )
            if args.dry_run:
                status = "dry-run"
                returncode = 0
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
