"""Run the ten-length echo spectroscopy campaign over multiple cutoffs.

Defaults reproduce the completed high-statistics campaign at ten linearly
spaced root-Lorentzian cutoffs from 0.001 through 0.010.  Each cutoff owns a
normal ``run_pulse_length_sweeps.py`` campaign directory, while a parent
manifest makes the batch resumable at cutoff boundaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "cutoff_pulse_length_spectroscopy"
DEFAULT_CUTOFFS = tuple(value / 1000.0 for value in range(1, 11))
DEFAULT_LENGTHS_US = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pulse_length_sweeps as length_sweeps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", default="q6")
    parser.add_argument("--cutoffs", type=float, nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument("--lengths-us", type=float, nargs="+", default=DEFAULT_LENGTHS_US)
    parser.add_argument("--num-shots", type=int, default=200)
    parser.add_argument("--frequency-span-mhz", type=float, default=2.0)
    parser.add_argument("--frequency-points", type=int, default=200)
    parser.add_argument("--amplitude-points", type=int, default=200)
    parser.add_argument("--min-amplitude-v", type=float, default=0.01)
    parser.add_argument("--max-amplitude-v", type=float, default=1.0)
    parser.add_argument("--template-length-ns", type=int, default=2_000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-id")
    parser.add_argument(
        "--reuse-cutoff-005",
        type=Path,
        default=None,
        help="Optional completed cutoff=0.005 campaign to reference instead of reacquiring.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    return parser.parse_args()


def cutoff_label(cutoff: float) -> str:
    return f"cutoff_{cutoff:.3f}".replace(".", "p")


def manifest_complete(path: Path, expected_lengths: list[float], cutoff: float) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    successful = sorted(
        float(run["pulse_length_us"])
        for run in manifest.get("runs", [])
        if run.get("status") == "ok"
    )
    return (
        abs(float(manifest.get("cutoff", -1.0)) - cutoff) < 1e-12
        and successful == sorted(expected_lengths)
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def child_arguments(args: argparse.Namespace, cutoff: float, output_dir: Path) -> list[str]:
    values = [
        "--qubit", args.qubit,
        "--lengths-us", *[f"{value:g}" for value in args.lengths_us],
        "--num-shots", str(args.num_shots),
        "--frequency-span-mhz", f"{args.frequency_span_mhz:g}",
        "--frequency-points", str(args.frequency_points),
        "--amplitude-points", str(args.amplitude_points),
        "--min-amplitude-v", f"{args.min_amplitude_v:g}",
        "--max-amplitude-v", f"{args.max_amplitude_v:g}",
        "--pulse-shape", "root_lorentzian",
        "--cutoff", f"{cutoff:.12g}",
        "--template-length-ns", str(args.template_length_ns),
        "--no-ac-stark-correction",
        "--output-root", str(output_dir.parent),
        "--campaign-id", output_dir.name,
        "--resume",
    ]
    if args.execute:
        values.append("--execute")
    if args.simulate:
        values.append("--simulate")
    return values


def main() -> int:
    args = parse_args()
    cutoffs = [float(value) for value in args.cutoffs]
    if not cutoffs or any(not 0 < value <= 1 for value in cutoffs):
        raise ValueError("Every cutoff must satisfy 0 < cutoff <= 1.")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Cutoff values must be unique.")

    batch_id = args.batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = (args.output_root / batch_id).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    parent_path = batch_dir / "batch_manifest.json"
    records = [
        {
            "cutoff": cutoff,
            "status": "pending",
            "campaign_directory": str(batch_dir / cutoff_label(cutoff)),
            "reused": False,
            "started_at": None,
            "finished_at": None,
            "error": None,
        }
        for cutoff in cutoffs
    ]
    parent: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "simulation" if args.simulate else "hardware",
        "qubit": args.qubit,
        "cutoffs": cutoffs,
        "pulse_lengths_us": [float(value) for value in args.lengths_us],
        "echo": True,
        "ac_stark_correction": False,
        "num_shots": args.num_shots,
        "frequency_span_mhz": args.frequency_span_mhz,
        "frequency_points": args.frequency_points,
        "amplitude_points": args.amplitude_points,
        "min_amplitude_v": args.min_amplitude_v,
        "max_amplitude_v": args.max_amplitude_v,
        "total_maps": len(cutoffs) * len(args.lengths_us),
        "total_shot_grid_iterations": len(cutoffs) * len(args.lengths_us)
        * args.frequency_points * args.amplitude_points * args.num_shots,
        "campaigns": records,
    }
    if parent_path.is_file():
        previous = json.loads(parent_path.read_text(encoding="utf-8"))
        parent["created_at"] = previous.get("created_at", parent["created_at"])
        previous_records = {float(row["cutoff"]): row for row in previous.get("campaigns", [])}
        for record in records:
            if record["cutoff"] in previous_records:
                record.update(previous_records[record["cutoff"]])
    write_json(parent_path, parent)

    print(f"[{'EXECUTE' if args.execute else 'PLAN ONLY'}] cutoff x pulse-length batch")
    print("  cutoffs: " + ", ".join(f"{value:.3f}" for value in cutoffs))
    print("  lengths (us): " + ", ".join(f"{value:g}" for value in args.lengths_us))
    print(f"  maps: {parent['total_maps']}")
    print(f"  grid: {args.frequency_points} detunings x {args.amplitude_points} amplitudes x {args.num_shots} shots")
    print(f"  shot-grid iterations: {parent['total_shot_grid_iterations']:,}")
    print("  echo: True; AC Stark correction: False")
    print(f"  output: {batch_dir}")
    if not args.execute:
        print("  add --execute to start")
        return 0

    reuse_dir = args.reuse_cutoff_005.resolve() if args.reuse_cutoff_005 else None
    for index, record in enumerate(records, start=1):
        cutoff = float(record["cutoff"])
        output_dir = Path(record["campaign_directory"])
        if manifest_complete(output_dir, list(args.lengths_us), cutoff):
            record["status"] = "ok"
            print(f"[{index}/{len(records)}] cutoff {cutoff:.3f}: already complete", flush=True)
            continue
        if (
            abs(cutoff - 0.005) < 1e-12
            and reuse_dir is not None
            and manifest_complete(reuse_dir, list(args.lengths_us), cutoff)
        ):
            record.update(
                status="ok",
                campaign_directory=str(reuse_dir),
                reused=True,
                finished_at=datetime.now().astimezone().isoformat(),
            )
            write_json(parent_path, parent)
            print(f"[{index}/{len(records)}] cutoff 0.005: reused {reuse_dir}", flush=True)
            continue

        record.update(
            status="running",
            reused=False,
            started_at=datetime.now().astimezone().isoformat(),
            error=None,
        )
        write_json(parent_path, parent)
        print(f"[{index}/{len(records)}] cutoff {cutoff:.3f}: starting", flush=True)
        try:
            result = length_sweeps.main(child_arguments(args, cutoff, output_dir))
            if result != 0 or not manifest_complete(output_dir, list(args.lengths_us), cutoff):
                raise RuntimeError("Child campaign did not finish all requested lengths.")
            record["status"] = "ok"
        except BaseException as error:
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["finished_at"] = datetime.now().astimezone().isoformat()
            write_json(parent_path, parent)
    print(f"Completed cutoff batch: {batch_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
