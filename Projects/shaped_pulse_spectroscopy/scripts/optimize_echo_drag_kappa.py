"""Short kappa optimization after selecting DRAG beta from measured center RMS.

Defaults to offline QUA/waveform checks. --execute acquires a symmetric kappa
scan, two local kappa refinements, two nearby betas, and a denser baseline /
candidate / baseline comparison. Uses the existing EchoLorentzian calibration;
never updates profiles, closes another user's QM, or changes shared defaults.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Projects.shaped_pulse_spectroscopy.scripts import optimize_echo_drag_beta as beta_tools
from calibration_io import CalibrationSaver
from calibrations.output_safety import assert_outputs_allowed
from experiments.detuning_amplitude_sweep import EchoLorentzian
from shaped_pulse_spectroscopy.echo_parameters import ECHO_PULSE
from shaped_pulse_spectroscopy.lorentzian import ac_stark_chirp_parameters, build_waveform
from quam_config import create_machine
from utils.rabi_amplitude import amplitude_to_rabi_frequency_hz, pi_pulse_effective_length_ns

DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "echo_drag_kappa_optimization"


@dataclass(frozen=True)
class Settings:
    qubit: str = "q1"
    profile: str = "iqm_chip"
    beta: float = ECHO_PULSE.drag_beta
    kappa_half_span_mhz_inv: float = 0.0003
    kappa_points: int = 7
    beta_recheck_step: float = 0.1
    amplitude_points: int = 13
    frequency_points: int = 81
    shots: int = 200
    validation_amplitude_points: int = 25
    validation_frequency_points: int = 121
    validation_shots: int = 400

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values() if isinstance(v, (int, float))):
            raise ValueError("Settings must be finite.")
        if self.kappa_points < 3 or self.kappa_points % 2 != 1:
            raise ValueError("Use an odd kappa grid with at least three points.")
        if not 0 < self.kappa_half_span_mhz_inv <= 0.001 or not 0 < self.beta_recheck_step <= 0.25:
            raise ValueError("Require a short bounded kappa and beta refinement.")
        if min(self.amplitude_points, self.validation_amplitude_points) < 5:
            raise ValueError("At least five amplitudes are required.")
        if min(self.frequency_points, self.validation_frequency_points) < 41:
            raise ValueError("At least 41 frequency points are required.")
        if min(self.shots, self.validation_shots) < 1:
            raise ValueError("Shots must be positive.")

    @property
    def kappas(self):
        grid = np.linspace(-self.kappa_half_span_mhz_inv, self.kappa_half_span_mhz_inv, self.kappa_points)
        middle = self.kappa_points // 2
        # Zero first, followed by interleaved negative/positive values.
        return [float(grid[middle])] + [float(grid[middle + sign * j])
            for j in range(1, middle + 1) for sign in (-1, 1)]


def scoring_settings(settings, *, validation=False):
    return beta_tools.SweepSettings(qubit=settings.qubit,
        amplitude_min=0.2, amplitude_max=1.0, amplitude_spacing="linear",
        amplitude_points=settings.validation_amplitude_points if validation else settings.amplitude_points,
        frequency_points=settings.validation_frequency_points if validation else settings.frequency_points,
        num_shots=settings.validation_shots if validation else settings.shots)


def parameters_for_pair(settings, beta, kappa, *, validation=False):
    p = beta_tools.parameters_for_beta(scoring_settings(settings, validation=validation), beta)
    p.ac_stark_correction = kappa != 0.0
    p.stark_kappa_mhz_inv = float(kappa)
    p.stark_chirp_max_error_hz = 10.0
    # The experiment falls back explicitly to GE when calibrated GEF is absent.
    p.use_three_state_discrimination = True
    p.timeout = 30
    return p


def local_kappas(best, settings):
    step = 2 * settings.kappa_half_span_mhz_inv / (settings.kappa_points - 1)
    return [float(k) for k in (best - step / 2, best + step / 2)
            if abs(k) <= settings.kappa_half_span_mhz_inv + 1e-15]


def profile_hashes(settings):
    return {str(p.relative_to(REPOSITORY_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((REPOSITORY_ROOT / "profiles" / settings.profile).glob("*.json"))}


def check_chirp(settings, machine, kappa):
    """Reconstruct integer-rate QUA chirp independently on the 4 ns clock."""
    q = machine.qubits[settings.qubit]
    pulse = q.xy.operations["x180"]
    p = parameters_for_pair(settings, settings.beta, kappa)
    i = np.asarray(build_waveform(p))
    effective_length = pi_pulse_effective_length_ns(pulse)
    f_rabi = np.asarray(amplitude_to_rabi_frequency_hz(i, pulse.amplitude, effective_length)) / 1e6
    chirp = ac_stark_chirp_parameters(i.tolist(), amplitude_factor=1.0,
        pi_amplitude=pulse.amplitude, pi_length_ns=effective_length,
        pulse_length_ns=p.lorentzian_length_in_ns, kappa_mhz_inv=kappa, max_error_hz=10.0)
    clock = np.arange(p.lorentzian_length_in_ns // 4 + 1)
    desired = np.interp(clock, np.linspace(0, clock[-1], len(i)), kappa * f_rabi**2 * 1e6)
    starts = np.asarray(chirp["times_cycles"])
    # Include fixed-point amplitude scaling and integer truncation used in QUA.
    errors = []
    for amp in scoring_settings(settings, validation=True).amplitude_factors:
        scale = np.floor(amp**2 * 2**28) / 2**28
        rates = np.floor(np.asarray(chirp["rates"], float) * scale)
        rates *= 0.001 if chirp["units"] == "mHz/nsec" else 1.0
        segment = np.searchsorted(starts, clock[:-1], side="right") - 1
        actual = np.r_[np.floor(chirp["initial_frequency_offset_hz"] * scale),
            np.floor(chirp["initial_frequency_offset_hz"] * scale) + np.cumsum(4 * rates[segment])]
        errors.append(float(np.max(np.abs(actual - amp**2 * desired))))
    max_error = max(errors)
    if max_error > 2000:
        raise ValueError(f"QUA chirp reconstruction error {max_error:g} Hz exceeds 2 kHz.")
    return dict(kappa_mhz_inv=kappa, peak_drive_correction_hz=float(np.max(np.abs(desired))),
        max_chirp_error_hz=max_error, segments=chirp["segments"], units=chirp["units"])


def offline_check(settings, machine):
    from qm import QuantumMachinesManager, generate_qua_script
    active = list(machine.active_qubit_names)
    if active != [settings.qubit]:
        raise ValueError(f"Unexpected active qubits: {active}")
    s = replace(scoring_settings(settings), beta_min=settings.beta-settings.beta_recheck_step,
        beta_max=settings.beta+settings.beta_recheck_step, beta_points=3)
    headroom = beta_tools.check_waveforms(s, machine)
    QuantumMachinesManager.set_capabilities_offline()
    details = []
    for kappa in (-settings.kappa_half_span_mhz_inv, 0.0, settings.kappa_half_span_mhz_inv):
        p = parameters_for_pair(settings, s.beta_min, kappa)
        cal = EchoLorentzian(parameters=p, machine=machine, qubit=settings.qubit,
            auto_connect=False, options=beta_tools.calibration_options())
        program = cal.create_qua_program()
        script = generate_qua_script(program, machine.generate_config())
        if not script:
            raise ValueError("Offline QUA generation failed.")
        details.append({**check_chirp(settings, machine, kappa),
            "qua_script_generated": True,
            "leakage_available": cal.namespace["three_state_discrimination_available"]})
    bias = getattr(machine, "dc_bias", None)
    return dict(**headroom, chirps=details,
        dc_bias_v=dict(getattr(bias, "qubit_biases_v", {})),
        score="RMS dip centers about covariance-weighted mean, all amplitudes required",
        pulse=asdict(ECHO_PULSE), settings=asdict(settings), kappas=settings.kappas,
        profile_hashes=profile_hashes(settings))


def score_dataset(dataset, settings, beta, kappa, *, validation=False):
    record, fitted = beta_tools.fit_and_score(dataset, beta=beta,
        settings=scoring_settings(settings, validation=validation))
    record.update(kappa_mhz_inv=float(kappa), leakage_available="leakage" in dataset)
    if record["leakage_available"]:
        leakage = dataset.leakage.sel(qubit=settings.qubit)
        pf = [float(leakage.isel(amp_prefactor=j).sel(detuning=center, method="nearest"))
              if np.isfinite(center) else None for j, center in enumerate(record["center_hz"])]
        record["leakage_at_center"] = pf
    return record, fitted


def plot_summary(output, records):
    if not records:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for stage in ("coarse", "refine", "beta_recheck"):
        rows = [r for r in records if r["stage"] == stage]
        if rows:
            axes[0].plot([r["kappa_mhz_inv"] for r in rows],
                [r["center_rms_hz"]/1e3 if r["valid"] else np.nan for r in rows], "o", label=stage)
    for row in records:
        if row["stage"].startswith("validation") and row["valid"]:
            axes[1].errorbar(row["amplitudes"],
                (np.asarray(row["center_hz"])-row["weighted_center_hz"])/1e3,
                yerr=np.asarray(row["center_error_hz"])/1e3, fmt=".-", label=row["stage"])
    axes[0].set(xlabel="Drive chirp kappa (1/MHz)", ylabel="Center RMS (kHz)")
    axes[1].set(xlabel="Amplitude factor", ylabel="Centered dip frequency (kHz)")
    for axis in axes:
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.savefig(output / "summary.png", dpi=160)
    fig.savefig(output / "summary.pdf")
    plt.close(fig)


def run(settings, machine, output):
    assert_outputs_allowed()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = profile_hashes(settings)
    if not before:
        raise ValueError("The selected profile is missing; refusing acquisition.")
    snapshot_root = output / "input_profiles"
    shutil.copytree(REPOSITORY_ROOT / "profiles" / settings.profile, snapshot_root / settings.profile)
    beta_tools.write_json(output / "settings.json", {**asdict(settings), "profile_hashes_before": before,
        "pulse": asdict(ECHO_PULSE), "beta_source": "explicit settings in Gaussian DRAG convention",
        "drag_beta_convention": "gaussian_positive_derivative_v1"})
    saver = CalibrationSaver(output_root=output / "raw", profiles_root=snapshot_root)
    records = []
    summary = dict(complete=False, selected_pair=None)

    def acquire(beta, kappa, stage, validation=False):
        if profile_hashes(settings) != before:
            raise RuntimeError("Profile changed during optimization; stopping before the next acquisition.")
        # Check waveform/chirp for each adaptive candidate before any connection.
        s = scoring_settings(settings, validation=validation)
        beta_tools.check_waveforms(replace(s, beta_min=beta-1e-9, beta_max=beta+1e-9, beta_points=2), machine)
        chirp_check = check_chirp(settings, machine, kappa)
        name = f"echo_kappa_{len(records):02d}_{stage}"
        print(f"ACQUIRE {name}: beta={beta:+.9g}, kappa={kappa:+.9g}, "
              f"{s.amplitude_points} amplitudes x {s.frequency_points} frequencies x {s.num_shots} shots", flush=True)
        cal = EchoLorentzian(parameters=parameters_for_pair(settings, beta, kappa, validation=validation),
            machine=machine, qubit=settings.qubit, auto_connect=False, saver=saver,
            options=beta_tools.calibration_options(acquire=True), name=name)
        try:
            status = cal.run()
        except BaseException:
            # Retain fetched data even if the normal saver/analysis raises.
            if "ds_raw" in cal.results:
                ds = cal.results["ds_raw"]
                np.savez_compressed(output / f"{name}_recovery.npz",
                    **{key: value.values for key, value in ds.variables.items()})
            raise
        if status.interrupted:
            raise RuntimeError("Acquisition interrupted; partial results retained.")
        record, fitted = score_dataset(cal.results["ds_raw"], settings, beta, kappa, validation=validation)
        record.update(stage=stage, amplitudes=s.amplitude_factors.tolist(), chirp_check=chirp_check,
            raw_directory=str(cal.namespace["calibration_run_directory"]))
        records.append(record)
        beta_tools.write_json(output / "records.json", records)
        beta_tools.save_scan(fitted, record, s, output / f"{name}.npz")
        beta_tools.save_2d_plot(fitted, record, s, output / f"{name}.png")
        plot_summary(output, records)
        print(f"RESULT {name}: RMS={record['center_rms_hz']} Hz, "
              f"accepted={sum(record['accepted'])}/{len(record['accepted'])}", flush=True)
        return record

    try:
        for kappa in settings.kappas:
            acquire(settings.beta, kappa, "coarse")
        best = beta_tools.select_best(records)
        if best is None:
            raise RuntimeError("No kappa has all amplitude fits accepted; no optimum declared.")
        summary["coarse_at_boundary"] = bool(np.isclose(abs(best["kappa_mhz_inv"]), settings.kappa_half_span_mhz_inv))
        for kappa in local_kappas(best["kappa_mhz_inv"], settings):
            acquire(settings.beta, kappa, "refine")
        best = beta_tools.select_best(records)
        summary["fixed_beta_best"] = dict(best)
        for beta in (settings.beta-settings.beta_recheck_step, settings.beta+settings.beta_recheck_step):
            acquire(beta, best["kappa_mhz_inv"], "beta_recheck")
        best = beta_tools.select_best(records)
        summary["search_candidate"] = dict(best)
        baseline1 = acquire(settings.beta, 0.0, "validation_baseline_before", True)
        candidate = acquire(best["beta"], best["kappa_mhz_inv"], "validation_candidate", True)
        baseline2 = acquire(settings.beta, 0.0, "validation_baseline_after", True)
        validation = [baseline1, candidate, baseline2]
        summary["validation_all_valid"] = all(r["valid"] for r in validation)
        if summary["validation_all_valid"]:
            reference = (baseline1["center_rms_hz"]+baseline2["center_rms_hz"])/2
            summary["baseline_rms_hz"] = reference
            summary["candidate_rms_hz"] = candidate["center_rms_hz"]
            summary["improvement_fraction"] = 1-candidate["center_rms_hz"]/reference if reference else None
            summary["candidate_better_than_both_baselines"] = candidate["center_rms_hz"] < min(
                baseline1["center_rms_hz"], baseline2["center_rms_hz"])
            summary["selected_pair"] = dict(beta=best["beta"], kappa_mhz_inv=best["kappa_mhz_inv"])
        summary["complete"] = True
    finally:
        summary.update(measurements=len(records), profile_hashes_after=profile_hashes(settings),
            profile_files_unchanged=before == profile_hashes(settings), profiles_updated=False)
        beta_tools.write_json(output / "summary.json", summary)
        plot_summary(output, records)
        print(f"OUTPUT {output}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    for name, value in asdict(Settings()).items():
        parser.add_argument("--"+name.replace("_", "-"), type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    execute, output = args.pop("execute"), args.pop("output")
    settings = Settings(**args)
    machine = create_machine(profile_name=settings.profile, qubit=settings.qubit)
    if list(machine.active_qubit_names) != [settings.qubit]:
        raise ValueError("Machine target differs from requested qubit.")
    if execute:
        output = output or DEFAULT_OUTPUT / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        print(json.dumps(run(settings, machine, output), indent=2), flush=True)
    else:
        checked = offline_check(settings, machine)
        if output:
            output.mkdir(parents=True, exist_ok=True)
            beta_tools.write_json(output / "offline_check.json", checked)
        print(json.dumps(checked, indent=2), flush=True)


if __name__ == "__main__":
    main()
