"""Run qubit spectroscopy while monitoring OPX controller temperatures."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from temperature_monitor.temperature_monitor import TemperatureMonitor

DEFAULT_CALIBRATION = REPO_ROOT / "calibrations" / "03a_qubit_spectroscopy.py"

# Set experiment notes here to create README.md in every run's output folder.
# Command-line --notes or --notes-file values override this text.
EXPERIMENT_NOTES = 'run on fans with 22.5 % with qubit spectroscopy script "03a_qubit_spectroscopy.py"'


def save_experiment_readme(monitor, calibration_path, notes):
    readme_path = monitor.output_dir / "README.md"
    content = (
        "# Temperature-Monitored Calibration\n\n"
        f"- Started: {monitor.plot_date}\n"
        f"- Calibration: `{calibration_path}`\n"
        f"- Controller: `{monitor.controller_name}`\n"
        "- Monitored sensors: "
        + ", ".join(f"`{key}`" for key in monitor.temperature_keys)
        + "\n\n"
        "## Notes\n\n"
        f"{notes.strip()}\n"
    )
    readme_path.write_text(content, encoding="utf-8")
    print(f"Experiment README saved to: {readme_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a calibration while recording controller temperature increases."
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Calibration Python script to run.",
    )
    parser.add_argument("--controller", default="con1")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--warning-increase",
        type=float,
        default=1.0,
        help="Warn when a sensor rises this many degrees above its initial value.",
    )
    notes = parser.add_mutually_exclusive_group()
    notes.add_argument(
        "--notes",
        help="Text to save in README.md beside the temperature results.",
    )
    notes.add_argument(
        "--notes-file",
        type=Path,
        help="Text file whose contents will be saved in the experiment README.md.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    calibration_path = args.calibration.resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration script not found: {calibration_path}")
    if args.notes_file is not None and not args.notes_file.is_file():
        raise FileNotFoundError(f"Notes file not found: {args.notes_file}")

    monitor = TemperatureMonitor(
        controller_name=args.controller,
        poll_interval=args.poll_interval,
        max_points=None,
        warning_increase=args.warning_increase,
        fem_ids=(6, 2),
        include_chassis_and_crps0=True,
    )
    notes = args.notes if args.notes is not None else EXPERIMENT_NOTES
    if args.notes_file is not None:
        notes = args.notes_file.read_text(encoding="utf-8")
    if notes:
        save_experiment_readme(monitor, calibration_path, notes)

    print(f"Running calibration: {calibration_path}")
    calibration = subprocess.Popen(
        [sys.executable, str(calibration_path)], cwd=REPO_ROOT
    )
    try:
        monitor.run_while(lambda: calibration.poll() is None)
    except KeyboardInterrupt:
        calibration.terminate()
        calibration.wait()
        raise
    finally:
        if calibration.poll() is None:
            calibration.wait()
        print(f"Temperature results saved to: {monitor.output_dir}")

    if calibration.returncode:
        raise subprocess.CalledProcessError(calibration.returncode, calibration.args)


if __name__ == "__main__":
    main()
