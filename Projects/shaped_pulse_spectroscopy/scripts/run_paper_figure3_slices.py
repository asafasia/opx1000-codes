"""Acquire and simulate the six fixed-amplitude slices used for a paper-style figure.

The default campaign mirrors the narrow root-Lorentzian settings of the recent
six-map spectroscopy campaign, but measures only 3, 20, and 40 MHz.  Each
amplitude is acquired once without the midpoint phase inversion and once with
it.  A matched vectorized three-level Lindblad calculation is overlaid on the
resulting 3x2 figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
SIMULATION_ROOT = PROJECT_ROOT / "simulation"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "paper_figure3_fixed_slices"
DEFAULT_RABI_MHZ = (3.0, 20.0, 40.0)
ECHO_MODES = (False, True)
# Lab-approved ceiling for the q6 OPX output and complete drive chain.
MAX_SAFE_AMPLITUDE_V = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", default="q6")
    parser.add_argument("--rabi-mhz", type=float, nargs=3, default=DEFAULT_RABI_MHZ)
    parser.add_argument("--num-shots", type=int, default=10_000)
    parser.add_argument("--frequency-span-mhz", type=float, default=1.0)
    parser.add_argument("--frequency-points", type=int, default=501)
    parser.add_argument("--pulse-length-us", type=float, default=20.0)
    parser.add_argument("--cutoff", type=float, default=0.005)
    parser.add_argument("--waveform-peak-v", type=float, default=0.7)
    parser.add_argument("--simulation-steps-per-half", type=int, default=8_000)
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hardware-only", action="store_true")
    parser.add_argument("--simulation-only", action="store_true")
    parser.add_argument(
        "--unmitigated-only",
        action="store_true",
        help="Replot an existing campaign from saved raw results without hardware.",
    )
    parser.add_argument(
        "--optimize-mitigation-only",
        action="store_true",
        help=(
            "Fit one readout-mitigation strength to all saved traces and replot "
            "without hardware."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    exclusive_modes = sum(
        bool(mode)
        for mode in (
            args.hardware_only,
            args.simulation_only,
            args.unmitigated_only,
            args.optimize_mitigation_only,
        )
    )
    if exclusive_modes > 1:
        raise ValueError(
            "--hardware-only, --simulation-only, --unmitigated-only, and "
            "--optimize-mitigation-only "
            "are mutually exclusive."
        )
    if (
        args.unmitigated_only or args.optimize_mitigation_only
    ) and args.campaign_dir is None:
        raise ValueError(
            "--unmitigated-only and --optimize-mitigation-only require --campaign-dir."
        )
    if args.num_shots <= 0:
        raise ValueError("num_shots must be positive.")
    if args.frequency_points < 2:
        raise ValueError("frequency_points must be at least 2.")
    if args.frequency_span_mhz <= 0:
        raise ValueError("frequency_span_mhz must be positive.")
    if args.pulse_length_us <= 0:
        raise ValueError("pulse_length_us must be positive.")
    if not 0 < args.cutoff <= 1:
        raise ValueError("cutoff must satisfy 0 < cutoff <= 1.")
    if not 0 < args.waveform_peak_v <= MAX_SAFE_AMPLITUDE_V:
        raise ValueError(f"waveform_peak_v must be in (0, {MAX_SAFE_AMPLITUDE_V:g}] V.")
    if args.simulation_steps_per_half < 1:
        raise ValueError("simulation_steps_per_half must be positive.")
    rabi_mhz = np.asarray(args.rabi_mhz, dtype=float)
    if not np.all(np.isfinite(rabi_mhz)) or np.any(rabi_mhz <= 0):
        raise ValueError("All Rabi frequencies must be finite and positive.")


def campaign_directory(args: argparse.Namespace) -> Path:
    if args.campaign_dir is not None:
        return args.campaign_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUTPUT_ROOT / timestamp).resolve()


def calibrated_amplitudes(
    rabi_mhz: np.ndarray,
    *,
    x180_amplitude_v: float,
    x180_length_ns: float,
    waveform_peak_v: float,
) -> tuple[np.ndarray, np.ndarray]:
    pi_rabi_hz = 1.0 / (2.0 * x180_length_ns * 1e-9)
    full_amplitude_v = rabi_mhz * 1e6 / pi_rabi_hz * x180_amplitude_v
    amp_prefactor = full_amplitude_v / waveform_peak_v
    if np.any(np.abs(full_amplitude_v) > MAX_SAFE_AMPLITUDE_V + 1e-12):
        raise ValueError(
            f"A requested Rabi frequency exceeds the calibrated "
            f"{MAX_SAFE_AMPLITUDE_V:g} V pulse limit: "
            f"{full_amplitude_v.tolist()} V."
        )
    if np.any(np.abs(amp_prefactor) >= 2):
        raise ValueError(
            "A requested Rabi frequency requires a QUA amplitude prefactor outside "
            f"[-2, 2): {amp_prefactor.tolist()}."
        )
    return full_amplitude_v, amp_prefactor


def machine_snapshot(machine: Any, args: argparse.Namespace) -> dict[str, Any]:
    qubit = machine.qubits[args.qubit]
    x180 = qubit.xy.operations["x180"]
    rabi_mhz = np.asarray(args.rabi_mhz, dtype=float)
    full_amplitude_v, amp_prefactor = calibrated_amplitudes(
        rabi_mhz,
        x180_amplitude_v=float(x180.amplitude),
        x180_length_ns=float(x180.length),
        waveform_peak_v=float(args.waveform_peak_v),
    )
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "qubit": args.qubit,
        "rf_frequency_hz": float(qubit.xy.RF_frequency),
        "anharmonicity_hz": float(qubit.anharmonicity),
        "t1_s": _optional_float(getattr(qubit, "T1", None)),
        "t2_star_s": _optional_float(getattr(qubit, "T2ramsey", None)),
        "x180_operation": "x180",
        "x180_type": type(x180).__name__,
        "x180_amplitude_v": float(x180.amplitude),
        "x180_length_ns": float(x180.length),
        "rabi_mhz": rabi_mhz.tolist(),
        "full_amplitude_v": full_amplitude_v.tolist(),
        "amp_prefactor": amp_prefactor.tolist(),
        "echo_modes": [False, True],
        "num_shots": int(args.num_shots),
        "frequency_span_mhz": float(args.frequency_span_mhz),
        "frequency_points": int(args.frequency_points),
        "pulse_shape": "root_lorentzian",
        "pulse_length_us": float(args.pulse_length_us),
        "waveform_template_length_us": float(args.pulse_length_us),
        "cutoff": float(args.cutoff),
        "waveform_peak_v": float(args.waveform_peak_v),
        "num_levels": 3,
        "simulation_steps_per_half": int(args.simulation_steps_per_half),
        "reset_type": "active",
        "use_state_discrimination": True,
        "use_readout_mitigation": 1.0,
        "ac_stark_correction": False,
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) and value > 0 else None


def hardware_parameters(snapshot: dict[str, Any], *, echo: bool, rabi_mhz: float):
    from shaped_pulse_spectroscopy.parameters import Parameters

    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.use_readout_mitigation = 1.0
    parameters.reset_type = "active"
    parameters.simulate = False
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = bool(echo)
    parameters.ac_stark_correction = False
    parameters.stark_kappa_mhz_inv = 0.0
    parameters.stark_chirp_max_error_hz = 0.0
    parameters.cutoff = float(snapshot["cutoff"])
    parameters.fixed_rabi_frequency_mhz = float(rabi_mhz)
    parameters.num_shots = int(snapshot["num_shots"])
    pulse_length_ns = int(round(float(snapshot["pulse_length_us"]) * 1000 / 4) * 4)
    parameters.lorentzian_length_in_ns = pulse_length_ns
    parameters.waveform_template_length_in_ns = pulse_length_ns
    parameters.lorentzian_peak_amplitude = float(snapshot["waveform_peak_v"])
    parameters.frequency_span_in_mhz = float(snapshot["frequency_span_mhz"])
    parameters.frequency_points = int(snapshot["frequency_points"])
    parameters.frequency_step_in_mhz = float(snapshot["frequency_span_mhz"]) / (
        int(snapshot["frequency_points"]) - 1
    )
    parameters.fit_fwhm = False
    return parameters


def empty_experiment_arrays(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    shape = (
        len(ECHO_MODES),
        len(snapshot["rabi_mhz"]),
        int(snapshot["frequency_points"]),
    )
    return {
        "detuning_mhz": np.linspace(
            -float(snapshot["frequency_span_mhz"]) / 2,
            float(snapshot["frequency_span_mhz"]) / 2,
            int(snapshot["frequency_points"]),
        ),
        "excited_probability": np.full(shape, np.nan, dtype=float),
    }


def load_or_initialize_experiment(
    campaign_dir: Path, snapshot: dict[str, Any]
) -> dict[str, np.ndarray]:
    path = campaign_dir / "experiment_traces.npz"
    if not path.is_file():
        return empty_experiment_arrays(snapshot)
    with np.load(path, allow_pickle=False) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def load_saved_unmitigated_experiment(
    campaign_dir: Path, snapshot: dict[str, Any]
) -> Path:
    """Collect the six raw saved traces without connecting to hardware."""
    manifest_path = campaign_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Campaign manifest does not exist: {manifest_path}")

    experiment = empty_experiment_arrays(snapshot)
    rabi_lookup = {
        float(rabi_mhz): index for index, rabi_mhz in enumerate(snapshot["rabi_mhz"])
    }
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["status"] == "ok"]

    for row in rows:
        echo_index = ECHO_MODES.index(row["echo"].strip().lower() == "true")
        rabi_index = rabi_lookup[float(row["rabi_mhz"])]
        run_directory = Path(row["run_directory"])
        with np.load(run_directory / "results.npz", allow_pickle=False) as results:
            values = np.asarray(results["state"], dtype=float).reshape(-1)
        with np.load(run_directory / "sweep.npz", allow_pickle=False) as sweep:
            detuning_mhz = np.asarray(sweep["detuning"], dtype=float) / 1e6
        if not np.allclose(detuning_mhz, experiment["detuning_mhz"]):
            raise ValueError(
                f"Saved detuning grid does not match the campaign: {run_directory}"
            )
        experiment["excited_probability"][echo_index, rabi_index] = values

    if not np.all(np.isfinite(experiment["excited_probability"])):
        raise ValueError("The campaign does not contain all six successful raw traces.")
    output_path = campaign_dir / "experiment_traces_unmitigated.npz"
    np.savez_compressed(output_path, **experiment)
    return output_path


def least_squares_mitigation_strength(
    raw: np.ndarray,
    fully_mitigated: np.ndarray,
    simulation: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Fit ``raw + strength * (fully_mitigated - raw)`` to simulation."""
    raw = np.asarray(raw, dtype=float)
    fully_mitigated = np.asarray(fully_mitigated, dtype=float)
    simulation = np.asarray(simulation, dtype=float)
    if raw.shape != fully_mitigated.shape or raw.shape != simulation.shape:
        raise ValueError(
            "Raw, mitigated, and simulation arrays must have equal shapes."
        )
    correction = fully_mitigated - raw
    valid = np.isfinite(raw) & np.isfinite(correction) & np.isfinite(simulation)
    denominator = float(np.sum(correction[valid] ** 2))
    if denominator <= 0:
        raise ValueError(
            "Cannot fit mitigation strength: correction is identically zero."
        )
    unconstrained = float(
        np.sum(correction[valid] * (simulation[valid] - raw[valid])) / denominator
    )
    return float(np.clip(unconstrained, *bounds)), unconstrained


def least_squares_affine_alignment(
    measured: np.ndarray, simulation: np.ndarray
) -> tuple[float, float]:
    """Fit ``gain * measured + offset`` to a simulated population."""
    measured = np.asarray(measured, dtype=float)
    simulation = np.asarray(simulation, dtype=float)
    if measured.shape != simulation.shape:
        raise ValueError("Measured and simulation arrays must have equal shapes.")
    valid = np.isfinite(measured) & np.isfinite(simulation)
    design = np.column_stack(
        (measured[valid], np.ones(np.count_nonzero(valid), dtype=float))
    )
    gain, offset = np.linalg.lstsq(design, simulation[valid], rcond=None)[0]
    return float(gain), float(offset)


def optimize_saved_mitigation(
    campaign_dir: Path, snapshot: dict[str, Any]
) -> tuple[Path, Path, Path]:
    """Fit and save one global mitigation strength for an existing campaign."""
    raw_path = campaign_dir / "experiment_traces_unmitigated.npz"
    if not raw_path.is_file():
        raw_path = load_saved_unmitigated_experiment(campaign_dir, snapshot)
    full_path = campaign_dir / "experiment_traces.npz"
    simulation_path = campaign_dir / "simulation_traces.npz"
    for path in (full_path, simulation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required campaign data does not exist: {path}")

    with np.load(raw_path, allow_pickle=False) as raw_archive:
        detuning_mhz = np.asarray(raw_archive["detuning_mhz"], dtype=float)
        raw = np.asarray(raw_archive["excited_probability"], dtype=float)
    with np.load(full_path, allow_pickle=False) as full_archive:
        fully_mitigated = np.asarray(full_archive["excited_probability"], dtype=float)
    with np.load(simulation_path, allow_pickle=False) as simulation_archive:
        simulation = np.asarray(
            simulation_archive["total_excited_probability"], dtype=float
        )

    strength, unconstrained = least_squares_mitigation_strength(
        raw, fully_mitigated, simulation
    )
    correction = fully_mitigated - raw
    optimized = raw + strength * correction
    affine_gain, affine_offset = least_squares_affine_alignment(raw, simulation)
    affine_aligned = affine_gain * raw + affine_offset
    mirrored_simulation = simulation[..., ::-1]
    mirrored_gain, mirrored_offset = least_squares_affine_alignment(
        raw, mirrored_simulation
    )
    mirrored_affine_aligned = mirrored_gain * raw + mirrored_offset
    per_panel_unconstrained = np.empty((len(ECHO_MODES), len(DEFAULT_RABI_MHZ)))
    for echo_index in range(len(ECHO_MODES)):
        for rabi_index in range(len(snapshot["rabi_mhz"])):
            _, panel_strength = least_squares_mitigation_strength(
                raw[echo_index, rabi_index],
                fully_mitigated[echo_index, rabi_index],
                simulation[echo_index, rabi_index],
            )
            per_panel_unconstrained[echo_index, rabi_index] = panel_strength

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean((values - simulation) ** 2)))

    report = {
        "objective": "unweighted RMSE over all six traces and all detuning points",
        "model": "raw + strength * (fully_mitigated - raw)",
        "bounds": [0.0, 1.0],
        "optimized_strength": strength,
        "unconstrained_strength": unconstrained,
        "rmse_raw": rmse(raw),
        "rmse_full_strength": rmse(fully_mitigated),
        "rmse_optimized": rmse(optimized),
        "empirical_affine_alignment": {
            "warning": (
                "Line-shape comparison only; this is not a calibrated or necessarily "
                "physical readout-assignment correction."
            ),
            "model": "gain * raw + offset",
            "gain": affine_gain,
            "offset": affine_offset,
            "rmse": rmse(affine_aligned),
        },
        "mirrored_simulation_affine_alignment": {
            "warning": (
                "Line-shape comparison only; the simulation detuning axis is mirrored "
                "and the empirical map is not calibrated readout mitigation."
            ),
            "model": "gain * raw + offset, compared with simulation(-detuning)",
            "gain": mirrored_gain,
            "offset": mirrored_offset,
            "rmse": float(
                np.sqrt(np.mean((mirrored_affine_aligned - mirrored_simulation) ** 2))
            ),
        },
        "per_panel_unconstrained_strength": {
            ("echo" if echo else "no_echo"): {
                f"{float(rabi_mhz):g}_mhz": float(
                    per_panel_unconstrained[echo_index, rabi_index]
                )
                for rabi_index, rabi_mhz in enumerate(snapshot["rabi_mhz"])
            }
            for echo_index, echo in enumerate(ECHO_MODES)
        },
    }
    (campaign_dir / "mitigation_strength_fit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    output_path = campaign_dir / "experiment_traces_optimized_mitigation.npz"
    np.savez_compressed(
        output_path,
        detuning_mhz=detuning_mhz,
        excited_probability=optimized,
        mitigation_strength=np.asarray(strength),
    )
    affine_path = campaign_dir / "experiment_traces_empirical_alignment.npz"
    np.savez_compressed(
        affine_path,
        detuning_mhz=detuning_mhz,
        excited_probability=affine_aligned,
        gain=np.asarray(affine_gain),
        offset=np.asarray(affine_offset),
    )
    mirrored_affine_path = (
        campaign_dir / "experiment_traces_empirical_alignment_mirrored_simulation.npz"
    )
    np.savez_compressed(
        mirrored_affine_path,
        detuning_mhz=detuning_mhz,
        excited_probability=mirrored_affine_aligned,
        gain=np.asarray(mirrored_gain),
        offset=np.asarray(mirrored_offset),
    )
    return output_path, affine_path, mirrored_affine_path


def trace_from_dataset(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    selected = dataset["state"].isel(qubit=0)
    if "amp_prefactor" in selected.dims:
        selected = selected.isel(amp_prefactor=0)
    values = np.asarray(selected.values, dtype=float).reshape(-1)
    detuning_mhz = np.asarray(dataset.detuning.values, dtype=float) / 1e6
    if values.size != detuning_mhz.size:
        raise ValueError(
            f"Unexpected fixed-slice shape: {values.shape} for {detuning_mhz.size} detunings."
        )
    return detuning_mhz, values


def run_hardware(
    campaign_dir: Path,
    snapshot: dict[str, Any],
    machine: Any,
    *,
    dry_run: bool,
) -> None:
    from calibrations.base import CalibrationOptions
    from experiments.fixed_amplitude_spectroscopy import EchoLorentzianFixedAmplitude

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
    experiment = load_or_initialize_experiment(campaign_dir, snapshot)
    manifest_path = campaign_dir / "manifest.csv"
    fieldnames = (
        "index",
        "started_at",
        "finished_at",
        "status",
        "echo",
        "rabi_mhz",
        "full_amplitude_v",
        "amp_prefactor",
        "run_directory",
        "error",
    )
    write_header = not manifest_path.is_file()
    rows = [
        (echo_index, rabi_index, echo, float(rabi_mhz))
        for rabi_index, rabi_mhz in enumerate(snapshot["rabi_mhz"])
        for echo_index, echo in enumerate(ECHO_MODES)
    ]
    needs_connect = True
    with manifest_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for index, (echo_index, rabi_index, echo, rabi_mhz) in enumerate(rows, start=1):
            mode = "echo" if echo else "no_echo"
            saved_values = experiment["excited_probability"][echo_index, rabi_index]
            if not dry_run and np.all(np.isfinite(saved_values)):
                print(
                    f"[{index}/6] {mode}, {rabi_mhz:g} MHz (already complete; skipping)",
                    flush=True,
                )
                continue
            row = {
                "index": index,
                "started_at": datetime.now().astimezone().isoformat(),
                "finished_at": "",
                "status": "dry-run" if dry_run else "running",
                "echo": echo,
                "rabi_mhz": rabi_mhz,
                "full_amplitude_v": snapshot["full_amplitude_v"][rabi_index],
                "amp_prefactor": snapshot["amp_prefactor"][rabi_index],
                "run_directory": "",
                "error": "",
            }
            print(f"[{index}/6] {mode}, {rabi_mhz:g} MHz", flush=True)
            try:
                if not dry_run:
                    calibration = EchoLorentzianFixedAmplitude(
                        parameters=hardware_parameters(
                            snapshot, echo=echo, rabi_mhz=rabi_mhz
                        ),
                        options=options,
                        machine=machine,
                        qubit=str(snapshot["qubit"]),
                        auto_connect=needs_connect,
                        name=f"paper_slice_{rabi_mhz:g}mhz_{mode}",
                    )
                    needs_connect = False
                    calibration.run()
                    detuning_mhz, values = trace_from_dataset(
                        calibration.results["ds_raw"]
                    )
                    if not np.allclose(detuning_mhz, experiment["detuning_mhz"]):
                        raise ValueError(
                            "Hardware detuning grid does not match the campaign grid."
                        )
                    experiment["excited_probability"][echo_index, rabi_index] = values
                    np.savez_compressed(
                        campaign_dir / "experiment_traces.npz", **experiment
                    )
                    row["run_directory"] = str(
                        calibration.namespace.get("calibration_run_directory", "")
                    )
                    row["status"] = "ok"
                    plot_figure(campaign_dir, snapshot)
            except BaseException as error:
                row["status"] = (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
                )
                row["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                row["finished_at"] = datetime.now().astimezone().isoformat()
                writer.writerow(row)
                stream.flush()


def run_simulation(campaign_dir: Path, snapshot: dict[str, Any]) -> None:
    if str(SIMULATION_ROOT) not in sys.path:
        sys.path.insert(0, str(SIMULATION_ROOT))
    from qutrit_slices import simulate_qutrit_slices

    detuning_mhz = np.linspace(
        -float(snapshot["frequency_span_mhz"]) / 2,
        float(snapshot["frequency_span_mhz"]) / 2,
        int(snapshot["frequency_points"]),
    )
    populations = np.empty(
        (len(ECHO_MODES), len(snapshot["rabi_mhz"]), 3, detuning_mhz.size),
        dtype=float,
    )
    for echo_index, echo in enumerate(ECHO_MODES):
        print(f"Simulating qutrit slices with echo={echo}...", flush=True)
        result = simulate_qutrit_slices(
            duration_us=float(snapshot["pulse_length_us"]),
            detuning_mhz=detuning_mhz,
            rabi_mhz=np.asarray(snapshot["rabi_mhz"], dtype=float),
            t1_us=float(snapshot["t1_s"]) * 1e6,
            t2_star_us=float(snapshot["t2_star_s"]) * 1e6,
            anharmonicity_mhz=-abs(float(snapshot["anharmonicity_hz"])) / 1e6,
            num_steps_per_half=int(snapshot["simulation_steps_per_half"]),
            cutoff=float(snapshot["cutoff"]),
            echo=bool(echo),
        )
        populations[echo_index] = np.stack(
            (result.ground, result.excited, result.second_excited), axis=1
        )
    np.savez_compressed(
        campaign_dir / "simulation_traces.npz",
        detuning_mhz=detuning_mhz,
        level_population=populations,
        total_excited_probability=populations[:, :, 1:, :].sum(axis=2),
    )
    plot_figure(campaign_dir, snapshot)


def plot_figure(
    campaign_dir: Path,
    snapshot: dict[str, Any],
    *,
    experiment_path: Path | None = None,
    output_stem: str = "paper_figure3_fixed_slices",
    experiment_label: str = "Experiment",
    mirror_simulation_x: bool = False,
) -> None:
    simulation_path = campaign_dir / "simulation_traces.npz"
    if experiment_path is None:
        experiment_path = campaign_dir / "experiment_traces.npz"
    if not simulation_path.is_file() and not experiment_path.is_file():
        return

    simulation = None
    experiment = None
    if simulation_path.is_file():
        simulation = np.load(simulation_path, allow_pickle=False)
    if experiment_path.is_file():
        experiment = np.load(experiment_path, allow_pickle=False)

    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.8,
            "axes.linewidth": 0.4,
            "axes.grid": False,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.fontsize": 6.0,
            "legend.frameon": True,
            "legend.handlelength": 1.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(3.38, 4.45),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel_labels = iter("abcdef")
    protocol_colors = (
        ("Root-Lorentzian", "#00838f", "#00474e"),
        ("Echo-root-Lorentzian", "#6a1b9a", "#350b4e"),
    )
    row_rabi_indices = np.argsort(np.asarray(snapshot["rabi_mhz"], dtype=float))
    for row_index, rabi_index in enumerate(row_rabi_indices):
        rabi_mhz = float(snapshot["rabi_mhz"][rabi_index])
        for echo_index, (title, marker_color, simulation_color) in enumerate(
            protocol_colors
        ):
            axis = axes[row_index, echo_index]
            if simulation is not None:
                simulated_values = simulation["total_excited_probability"][
                    echo_index, rabi_index
                ]
                if mirror_simulation_x:
                    simulated_values = simulated_values[::-1]
                axis.plot(
                    simulation["detuning_mhz"],
                    simulated_values,
                    color=simulation_color,
                    linewidth=0.8,
                    zorder=2,
                )
            if experiment is not None:
                values = experiment["excited_probability"][echo_index, rabi_index]
                valid = np.isfinite(values)
                if np.any(valid):
                    axis.scatter(
                        experiment["detuning_mhz"][valid],
                        values[valid],
                        s=1.0,
                        color=marker_color,
                        alpha=1.0,
                        edgecolors="none",
                        linewidths=0.0,
                        zorder=3,
                    )
            axis.axvline(0, color="0.45", linestyle="--", linewidth=0.55, zorder=0)
            axis.set_xlim(
                -float(snapshot["frequency_span_mhz"]) / 2,
                float(snapshot["frequency_span_mhz"]) / 2,
            )
            axis.set_ylim(0.0, 0.82)
            axis.set_box_aspect(1.0)
            axis.text(
                0.04,
                0.94,
                f"({next(panel_labels)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.5,
                },
            )
            if row_index == 0:
                axis.set_title(title)
            if row_index == 2:
                axis.set_xlabel(r"$\Delta/2\pi$ (MHz)")
            if echo_index == 0:
                axis.set_ylabel(r"$P_e$")
                axis.text(
                    0.96,
                    0.94,
                    rf"$\Omega_0/2\pi={rabi_mhz:g}$ MHz",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.0,
                )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            campaign_dir / f"{output_stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)
    if simulation is not None:
        simulation.close()
    if experiment is not None:
        experiment.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    output_dir = campaign_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "parameters.json"

    if args.optimize_mitigation_only:
        if not snapshot_path.is_file():
            raise FileNotFoundError(
                f"Campaign parameters do not exist: {snapshot_path}"
            )
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        optimized_path, affine_path, mirrored_affine_path = optimize_saved_mitigation(
            output_dir, snapshot
        )
        plot_figure(
            output_dir,
            snapshot,
            experiment_path=optimized_path,
            output_stem="paper_figure3_fixed_slices_optimized_mitigation",
            experiment_label="Experiment",
        )
        plot_figure(
            output_dir,
            snapshot,
            experiment_path=mirrored_affine_path,
            output_stem=(
                "paper_figure3_fixed_slices_empirical_alignment_mirrored_simulation"
            ),
            experiment_label="Experiment",
            mirror_simulation_x=True,
        )
        plot_figure(
            output_dir,
            snapshot,
            experiment_path=affine_path,
            output_stem="paper_figure3_fixed_slices_empirical_alignment",
            experiment_label="Experiment",
        )
        print(output_dir)
        return 0

    if args.unmitigated_only:
        if not snapshot_path.is_file():
            raise FileNotFoundError(
                f"Campaign parameters do not exist: {snapshot_path}"
            )
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        raw_path = load_saved_unmitigated_experiment(output_dir, snapshot)
        plot_figure(
            output_dir,
            snapshot,
            experiment_path=raw_path,
            output_stem="paper_figure3_fixed_slices_unmitigated",
            experiment_label="Experiment",
        )
        print(output_dir)
        return 0

    if args.simulation_only:
        if not snapshot_path.is_file():
            raise FileNotFoundError(
                f"Campaign parameters do not exist: {snapshot_path}"
            )
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        run_simulation(output_dir, snapshot)
        print(output_dir)
        return 0

    for path in (PROJECT_ROOT, REPOSITORY_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from quam_config import create_machine

    machine = create_machine(qubit=args.qubit)
    snapshot = machine_snapshot(machine, args)
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(
        "Calibrated full amplitudes: "
        + ", ".join(
            f"{rabi:g} MHz -> {amplitude:.6f} V"
            for rabi, amplitude in zip(
                snapshot["rabi_mhz"], snapshot["full_amplitude_v"], strict=True
            )
        ),
        flush=True,
    )

    if not args.hardware_only:
        run_simulation(output_dir, snapshot)
    if not args.simulation_only:
        run_hardware(output_dir, snapshot, machine, dry_run=args.dry_run)
    plot_figure(output_dir, snapshot)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
