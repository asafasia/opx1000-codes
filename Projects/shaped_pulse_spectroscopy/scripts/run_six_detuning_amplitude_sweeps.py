"""Run q6 active-reset Rabi/IQ checks and six shaped-pulse sweeps.

The campaign starts with power-Rabi and IQ-blob calibrations, then runs three
50 MHz shaped-pulse scans followed by the same three pulse settings over 1 MHz.
Each shaped-pulse scan uses 501 detuning points, 250 amplitude factors, 2,000
shots, and a maximum amplitude factor of one.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrations.base import CalibrationOptions
from calibration_utils.iq_blobs.parameters import Parameters as IqBlobsParameters
from calibration_utils.power_rabi.parameters import Parameters as PowerRabiParameters
from experiments.detuning_amplitude_sweep import EchoLorentzian
from shaped_pulse_spectroscopy.lorentzian import plot_raw_data
from shaped_pulse_spectroscopy.parameters import Parameters
from quam_config import create_machine


PowerRabi = importlib.import_module("calibrations.04b_power_rabi").PowerRabi
IqBlobs = importlib.import_module("calibrations.07_iq_blobs").IqBlobs


SCAN_SETTINGS = (
    ("cutoff_0p999_no_echo", 0.999, False),
    ("cutoff_0p005_no_echo", 0.005, False),
    ("cutoff_0p005_echo", 0.005, True),
)
SPANS_MHZ = (50.0, 1.0)
NUM_FREQUENCIES = 501
NUM_AMPLITUDES = 250
NUM_SHOTS = 2_000
MAX_AMP_FACTOR = 1.0
MAX_SAFE_WAVEFORM_AMPLITUDE_V = 0.7
PULSE_LENGTH_NS = 20_000
POWER_RABI_SHOTS = 200
IQ_BLOB_SHOTS = 10_000
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "six_detuning_amplitude_sweeps"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", default="q6")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--scan-indices",
        nargs="+",
        type=int,
        choices=range(1, 7),
        help="Run only the selected 1-based scan indices.",
    )
    parser.add_argument(
        "--skip-supporting",
        action="store_true",
        help="Skip the power-Rabi and IQ-blob support calibrations.",
    )
    parser.add_argument(
        "--resume-output-dir",
        type=Path,
        help="Append completed scan rows and figures to an existing campaign.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run on hardware. Without this flag, print the six-run plan only.",
    )
    return parser


def parameters_for(*, span_mhz: float, cutoff: float, echo: bool) -> Parameters:
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.use_readout_mitigation = 0.4

    parameters.simulate = False
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = echo
    parameters.ac_stark_correction = False
    parameters.stark_kappa_mhz_inv = 0.0
    parameters.stark_chirp_max_error_hz = 0.0
    parameters.cutoff = cutoff
    parameters.num_shots = NUM_SHOTS
    parameters.lorentzian_length_in_ns = PULSE_LENGTH_NS
    parameters.waveform_template_length_in_ns = PULSE_LENGTH_NS
    parameters.lorentzian_peak_amplitude = MAX_SAFE_WAVEFORM_AMPLITUDE_V
    parameters.min_amp_factor = 0.0
    parameters.max_amp_factor = MAX_AMP_FACTOR
    parameters.amp_factor_points = NUM_AMPLITUDES
    parameters.amp_factor_step = MAX_AMP_FACTOR / (NUM_AMPLITUDES - 1)
    parameters.amp_factor_spacing = "linear"
    parameters.frequency_span_in_mhz = span_mhz
    parameters.frequency_points = NUM_FREQUENCIES
    parameters.frequency_step_in_mhz = span_mhz / (NUM_FREQUENCIES - 1)
    parameters.fit_fwhm = False
    return parameters


def scan_plan() -> list[dict[str, object]]:
    scans: list[dict[str, object]] = []
    for span_mhz in SPANS_MHZ:
        width = "broad_50mhz" if span_mhz == 50.0 else "narrow_1mhz"
        for setting_name, cutoff, echo in SCAN_SETTINGS:
            scans.append(
                {
                    "name": f"{width}_{setting_name}",
                    "span_mhz": span_mhz,
                    "cutoff": cutoff,
                    "echo": echo,
                }
            )
    return scans


def save_plot(calibration: EchoLorentzian, path: Path) -> None:
    figures = plot_raw_data(
        calibration.results["ds_raw"],
        calibration.namespace["qubits"],
        use_state_discrimination=calibration.parameters.use_state_discrimination,
    )
    figure = figures[0] if isinstance(figures, (list, tuple)) else figures
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def supporting_options() -> CalibrationOptions:
    return CalibrationOptions(
        save_raw_data=True,
        save_analysis_result=True,
        save_figures=True,
        analyse_data=True,
        plot_data=True,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
        ai_review=False,
    )


def run_supporting_calibrations(*, qubit: str, machine, output_dir: Path) -> None:
    """Run and index the active-reset power-Rabi and IQ-blob checks."""
    manifest_path = output_dir / "supporting_calibrations.csv"
    fieldnames = [
        "index",
        "name",
        "started_at",
        "finished_at",
        "status",
        "reset_type",
        "num_shots",
        "run_directory",
        "error",
    ]
    calibrations = []

    rabi_parameters = PowerRabiParameters()
    rabi_parameters.reset_type = "active"
    rabi_parameters.use_state_discrimination = True
    rabi_parameters.use_readout_mitigation = False
    rabi_parameters.num_shots = POWER_RABI_SHOTS
    rabi_parameters.transition = "ge"
    rabi_parameters.pi_repetitions = 6
    rabi_parameters.operation = "x180"
    calibrations.append(
        (
            "power_rabi_active",
            rabi_parameters.num_shots,
            PowerRabi(
                parameters=rabi_parameters,
                options=supporting_options(),
                machine=machine,
                qubit=qubit,
                auto_connect=True,
                name="power_rabi_active",
            ),
        )
    )

    iq_parameters = IqBlobsParameters()
    iq_parameters.reset_type = "active"
    iq_parameters.states = ["g", "e"]
    iq_parameters.qubit_operation = "x180_const"
    iq_parameters.num_shots = IQ_BLOB_SHOTS
    calibrations.append(
        (
            "iq_blobs_active",
            iq_parameters.num_shots,
            IqBlobs(
                parameters=iq_parameters,
                options=supporting_options(),
                machine=machine,
                qubit=qubit,
                auto_connect=False,
            ),
        )
    )

    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, (name, num_shots, calibration) in enumerate(calibrations, start=1):
            row = {
                "index": index,
                "name": name,
                "started_at": datetime.now().astimezone().isoformat(),
                "finished_at": "",
                "status": "running",
                "reset_type": "active",
                "num_shots": num_shots,
                "run_directory": "",
                "error": "",
            }
            print(f"[support {index}/2] Starting {name}...", flush=True)
            try:
                calibration.run()
                row["status"] = "ok"
                row["run_directory"] = str(
                    calibration.namespace.get("calibration_run_directory", "")
                )
                print(
                    f"[support {index}/2] Saved {row['run_directory']}",
                    flush=True,
                )
            except BaseException as error:
                row["status"] = (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
                )
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"[support {index}/2] {row['error']}", flush=True)
                raise
            finally:
                row["finished_at"] = datetime.now().astimezone().isoformat()
                writer.writerow(row)
                stream.flush()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    all_scans = scan_plan()
    selected_indices = set(args.scan_indices or range(1, len(all_scans) + 1))
    scans = [
        (index, scan)
        for index, scan in enumerate(all_scans, start=1)
        if index in selected_indices
    ]
    print(f"Prepared {len(scans)} shaped-pulse sweeps for {args.qubit}.")
    for index, scan in scans:
        print(
            f"  {index}. {scan['name']}: span={scan['span_mhz']:g} MHz, "
            f"cutoff={scan['cutoff']:g}, echo={scan['echo']}"
        )
    print(
        f"Each sweep: {NUM_FREQUENCIES} frequencies x {NUM_AMPLITUDES} "
        f"amplitudes x {NUM_SHOTS} shots; max amp factor={MAX_AMP_FACTOR:g}; "
        f"pulse/template={PULSE_LENGTH_NS / 1000:g} us; "
        f"waveform peak={MAX_SAFE_WAVEFORM_AMPLITUDE_V:g} V."
    )
    if args.skip_supporting:
        print("Supporting calibrations: skipped.")
    else:
        print(
            f"Supporting calibrations: active-reset power Rabi ({POWER_RABI_SHOTS} shots) "
            f"and active-reset IQ blobs ({IQ_BLOB_SHOTS} shots)."
        )
    if not args.execute:
        print("Plan only. Add --execute to run on hardware.")
        return 0

    if args.resume_output_dir is not None:
        output_dir = args.resume_output_dir.resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Campaign directory does not exist: {output_dir}")
    else:
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = args.output_root / timestamp
        output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.csv"
    machine = create_machine(qubit=args.qubit)
    if not args.skip_supporting:
        run_supporting_calibrations(
            qubit=args.qubit,
            machine=machine,
            output_dir=output_dir,
        )
    options = CalibrationOptions(
        save_raw_data=True,
        save_analysis_result=False,
        save_figures=False,
        analyse_data=True,
        plot_data=False,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
        ai_review=False,
    )

    fieldnames = [
        "index",
        "name",
        "started_at",
        "finished_at",
        "status",
        "span_mhz",
        "cutoff",
        "echo",
        "frequency_points",
        "amplitude_points",
        "num_shots",
        "max_amp_factor",
        "waveform_peak_v",
        "run_directory",
        "figure",
        "error",
    ]
    append_manifest = args.resume_output_dir is not None and manifest_path.is_file()
    manifest_mode = "a" if append_manifest else "w"
    with manifest_path.open(manifest_mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not append_manifest:
            writer.writeheader()
        for position, (index, scan) in enumerate(scans, start=1):
            started_at = datetime.now().astimezone().isoformat()
            print(
                f"[{position}/{len(scans)}; scan {index}/6] Starting {scan['name']}...",
                flush=True,
            )
            row = {
                "index": index,
                "name": scan["name"],
                "started_at": started_at,
                "finished_at": "",
                "status": "running",
                "span_mhz": scan["span_mhz"],
                "cutoff": scan["cutoff"],
                "echo": scan["echo"],
                "frequency_points": NUM_FREQUENCIES,
                "amplitude_points": NUM_AMPLITUDES,
                "num_shots": NUM_SHOTS,
                "max_amp_factor": MAX_AMP_FACTOR,
                "waveform_peak_v": MAX_SAFE_WAVEFORM_AMPLITUDE_V,
                "run_directory": "",
                "figure": "",
                "error": "",
            }
            try:
                parameters = parameters_for(
                    span_mhz=float(scan["span_mhz"]),
                    cutoff=float(scan["cutoff"]),
                    echo=bool(scan["echo"]),
                )
                calibration = EchoLorentzian(
                    parameters=parameters,
                    options=options,
                    machine=machine,
                    qubit=args.qubit,
                    auto_connect=args.skip_supporting and position == 1,
                    name=str(scan["name"]),
                )
                calibration.run()
                figure_path = output_dir / f"{index:02d}_{scan['name']}.png"
                save_plot(calibration, figure_path)
                row["status"] = "ok"
                row["run_directory"] = str(
                    calibration.namespace.get("calibration_run_directory", "")
                )
                row["figure"] = str(figure_path)
                print(
                    f"[{position}/{len(scans)}; scan {index}/6] Saved {figure_path}",
                    flush=True,
                )
            except BaseException as error:
                row["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
                row["error"] = f"{type(error).__name__}: {error}"
                print(
                    f"[{position}/{len(scans)}; scan {index}/6] {row['error']}",
                    flush=True,
                )
                raise
            finally:
                row["finished_at"] = datetime.now().astimezone().isoformat()
                writer.writerow(row)
                stream.flush()

    print(f"Completed {len(scans)} selected sweep(s). Output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
