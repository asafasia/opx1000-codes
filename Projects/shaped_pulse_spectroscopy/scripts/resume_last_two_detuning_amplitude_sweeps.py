"""Resume q6 scans 5 and 6 of the 20 us shaped-pulse campaign."""

from __future__ import annotations

import csv
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
from experiments.detuning_amplitude_sweep import EchoLorentzian
from shaped_pulse_spectroscopy.lorentzian import plot_raw_data
from shaped_pulse_spectroscopy.parameters import Parameters
from quam_config import create_machine


QUBIT = "q6"
CAMPAIGN_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "six_detuning_amplitude_sweeps"
    / "2026-08-25_18-41-34"
)
SCAN_SETTINGS = (
    (5, "narrow_1mhz_cutoff_0p005_no_echo", False),
    (6, "narrow_1mhz_cutoff_0p005_echo", True),
)
NUM_FREQUENCIES = 501
NUM_AMPLITUDES = 250
NUM_SHOTS = 2_000
PULSE_LENGTH_NS = 20_000
WAVEFORM_PEAK_V = 0.7


def parameters_for(*, echo: bool) -> Parameters:
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.use_readout_mitigation = 1
    parameters.simulate = False
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = echo
    parameters.ac_stark_correction = False
    parameters.stark_kappa_mhz_inv = 0.0
    parameters.stark_chirp_max_error_hz = 0.0
    parameters.cutoff = 0.005
    parameters.num_shots = NUM_SHOTS
    parameters.lorentzian_length_in_ns = PULSE_LENGTH_NS
    parameters.waveform_template_length_in_ns = PULSE_LENGTH_NS
    parameters.lorentzian_peak_amplitude = WAVEFORM_PEAK_V
    parameters.min_amp_factor = 0.0
    parameters.max_amp_factor = 1.0
    parameters.amp_factor_points = NUM_AMPLITUDES
    parameters.amp_factor_step = 1.0 / (NUM_AMPLITUDES - 1)
    parameters.amp_factor_spacing = "linear"
    parameters.frequency_span_in_mhz = 1.0
    parameters.frequency_points = NUM_FREQUENCIES
    parameters.frequency_step_in_mhz = 1.0 / (NUM_FREQUENCIES - 1)
    parameters.fit_fwhm = False
    return parameters


def save_plot(calibration: EchoLorentzian, path: Path) -> None:
    figure = plot_raw_data(
        calibration.results["ds_raw"],
        calibration.namespace["qubits"],
        use_state_discrimination=True,
    )
    if isinstance(figure, (list, tuple)):
        figure = figure[0]
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    if not CAMPAIGN_DIR.is_dir():
        raise FileNotFoundError(f"Campaign directory does not exist: {CAMPAIGN_DIR}")

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
    machine = create_machine(qubit=QUBIT)
    manifest_path = CAMPAIGN_DIR / "manifest.csv"
    fieldnames = [
        "index", "name", "started_at", "finished_at", "status", "span_mhz",
        "cutoff", "echo", "frequency_points", "amplitude_points", "num_shots",
        "max_amp_factor", "waveform_peak_v", "run_directory", "figure", "error",
    ]

    with manifest_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        for position, (index, name, echo) in enumerate(SCAN_SETTINGS, start=1):
            row = {
                "index": index,
                "name": name,
                "started_at": datetime.now().astimezone().isoformat(),
                "finished_at": "",
                "status": "running",
                "span_mhz": 1.0,
                "cutoff": 0.005,
                "echo": echo,
                "frequency_points": NUM_FREQUENCIES,
                "amplitude_points": NUM_AMPLITUDES,
                "num_shots": NUM_SHOTS,
                "max_amp_factor": 1.0,
                "waveform_peak_v": WAVEFORM_PEAK_V,
                "run_directory": "",
                "figure": "",
                "error": "",
            }
            print(f"[{position}/2; scan {index}/6] Starting {name}...", flush=True)
            try:
                calibration = EchoLorentzian(
                    parameters=parameters_for(echo=echo),
                    options=options,
                    machine=machine,
                    qubit=QUBIT,
                    auto_connect=position == 1,
                    name=name,
                )
                calibration.run()
                figure_path = CAMPAIGN_DIR / f"{index:02d}_{name}.png"
                save_plot(calibration, figure_path)
                row["status"] = "ok"
                row["run_directory"] = str(
                    calibration.namespace.get("calibration_run_directory", "")
                )
                row["figure"] = str(figure_path)
                print(f"[{position}/2; scan {index}/6] Saved {figure_path}", flush=True)
            except BaseException as error:
                row["status"] = (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
                )
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"[{position}/2; scan {index}/6] {row['error']}", flush=True)
                raise
            finally:
                row["finished_at"] = datetime.now().astimezone().isoformat()
                writer.writerow(row)
                stream.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
