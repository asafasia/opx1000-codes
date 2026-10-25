"""Find DRAG beta from configurable echo spectroscopy amplitude slices.

Defaults to an offline waveform/QUA check. Use --execute to acquire 40 betas
from -0.7 to +0.1, each at amplitude prefactors 0.25, 0.5 and 1.0. The best
sampled beta minimizes fitted-center RMS about their covariance-weighted mean.
All amplitude-slice dip fits must pass. Results are saved under data/echo_drag_beta_1d;
device profiles are never updated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np

from calibration_io import CalibrationSaver
from calibrations.base import CalibrationOptions
from experiments.detuning_amplitude_sweep import EchoLorentzian
from experiments.stark_kappa_sweep import add_simple_negative_gaussian_fits
from shaped_pulse_spectroscopy.lorentzian import (
    amplitude_prefactors, build_waveform, drag_quadrature_waveform,
)
from shaped_pulse_spectroscopy.parameters import Parameters
from shaped_pulse_spectroscopy.echo_parameters import DEFAULT_QUBIT, ECHO_PULSE, echo_parameters
from quam_config import create_machine

AMPLITUDE_FACTORS = (0.25, 0.5, 1.0)
MAX_AMPLITUDE_V = 1.0
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "echo_drag_beta_1d"


@dataclass(frozen=True)
class SweepSettings:
    qubit: str = DEFAULT_QUBIT
    beta_min: float = -0.7
    beta_max: float = 0.1
    beta_points: int = 40
    beta_order: str = "ascending"
    amplitude_min: float = 0.25
    amplitude_max: float = 1.0
    amplitude_points: int = 3
    amplitude_spacing: str = "log"
    pulse_length_ns: int = ECHO_PULSE.lorentzian_length_in_ns
    template_length_ns: int = ECHO_PULSE.waveform_template_length_in_ns
    cutoff: float = ECHO_PULSE.cutoff
    peak_amplitude_v: float = ECHO_PULSE.lorentzian_peak_amplitude
    transition_ns: float = ECHO_PULSE.echo_transition_time_ns
    edge_smoothing: str = ECHO_PULSE.edge_smoothing
    edge_smoothing_power: int = ECHO_PULSE.edge_smoothing_power
    frequency_span_mhz: float = 1.0
    frequency_points: int = 201
    num_shots: int = 200
    min_contrast: float = 0.02
    min_r_squared: float = 0.1

    def __post_init__(self):
        numeric = [v for v in asdict(self).values() if isinstance(v, (float, int))]
        if not all(np.isfinite(numeric)):
            raise ValueError("Sweep settings must be finite.")
        if not self.qubit or self.beta_points < 2 or self.beta_min >= self.beta_max:
            raise ValueError("Specify a qubit and at least two betas with beta_min < beta_max.")
        if self.beta_order not in ("ascending", "descending"):
            raise ValueError("beta_order must be ascending or descending.")
        if self.amplitude_spacing not in ("linear", "log"):
            raise ValueError("amplitude_spacing must be linear or log.")
        if (self.amplitude_points < 2 or not 0 < self.amplitude_min < self.amplitude_max < 2):
            raise ValueError("Require at least two amplitudes and 0 < amplitude_min < amplitude_max < 2.")
        if self.pulse_length_ns < 16 or self.pulse_length_ns % 4:
            raise ValueError("Pulse length must be at least 16 ns and divisible by 4 ns.")
        if not 0 < self.peak_amplitude_v <= MAX_AMPLITUDE_V:
            raise ValueError(f"peak_amplitude_v must be in (0, {MAX_AMPLITUDE_V:g}] V.")
        if self.transition_ns <= 0:
            raise ValueError("This DRAG sweep requires a positive echo transition.")
        if self.frequency_span_mhz <= 0 or self.frequency_points < 9 or self.num_shots < 1:
            raise ValueError("Require positive span/shots and at least nine frequency points.")
        if not 0 <= self.min_r_squared <= 1 or self.min_contrast <= 0:
            raise ValueError("Require 0 <= min_r_squared <= 1 and positive min_contrast.")

    @property
    def betas(self):
        values = np.linspace(self.beta_min, self.beta_max, self.beta_points)
        return values[::-1] if self.beta_order == "descending" else values

    @property
    def amplitude_factors(self):
        spacing = np.geomspace if self.amplitude_spacing == "log" else np.linspace
        return spacing(self.amplitude_min, self.amplitude_max, self.amplitude_points)


def parameters_for_beta(settings: SweepSettings, beta: float) -> Parameters:
    p = echo_parameters(
        echo=True, drag_beta=float(beta),
        echo_transition_time_ns=settings.transition_ns,
        edge_smoothing=settings.edge_smoothing,
        edge_smoothing_power=settings.edge_smoothing_power,
        lorentzian_length_in_ns=settings.pulse_length_ns,
        waveform_template_length_in_ns=min(settings.template_length_ns, settings.pulse_length_ns),
        lorentzian_peak_amplitude=settings.peak_amplitude_v, cutoff=settings.cutoff,
    )
    p.min_amp_factor, p.max_amp_factor = settings.amplitude_min, settings.amplitude_max
    p.amp_factor_points = settings.amplitude_points
    p.amp_factor_spacing = settings.amplitude_spacing
    p.amp_factor_step = (settings.amplitude_max-settings.amplitude_min)/(settings.amplitude_points-1)
    p.frequency_span_in_mhz = settings.frequency_span_mhz
    p.frequency_points = settings.frequency_points
    p.frequency_step_in_mhz = settings.frequency_span_mhz/(settings.frequency_points-1)
    p.num_shots = settings.num_shots
    p.fit_fwhm = False  # The same existing negative-Gaussian fit is applied below.
    return p


def calibration_options(*, acquire=False):
    return CalibrationOptions(
        save_raw_data=acquire, save_analysis_result=False, save_figures=False,
        analyse_data=acquire, plot_data=False, update_state=False,
        propose_profile_update=False, apply_profile_update=False, ai_review=False,
        report_runtime_estimate=acquire,
    )


def check_waveforms(settings, machine):
    """Validate every beta, including its complete I/Q voltage, before connecting."""
    qubit = machine.qubits[settings.qubit]
    alpha_mhz = abs(float(qubit.anharmonicity)) / 1e6
    largest = 0.0
    worst_beta = None
    for beta in settings.betas:
        p = parameters_for_beta(settings, beta)
        i = np.asarray(build_waveform(p))
        q = np.asarray(drag_quadrature_waveform(
            i, drag_beta=float(beta), anharmonicity_mhz=alpha_mhz,
            pulse_length_ns=settings.pulse_length_ns,
        ))
        peak = float(np.max(np.abs(i + 1j*q))) * max(settings.amplitude_factors)
        if not np.isfinite(peak) or peak > MAX_AMPLITUDE_V:
            raise ValueError(f"beta={beta:g}: complete I/Q peak {peak:g} V exceeds {MAX_AMPLITUDE_V:g} V.")
        if peak > largest:
            largest, worst_beta = peak, float(beta)
    return {"max_complex_amplitude_v": largest, "worst_beta": worst_beta}


def offline_check(settings, machine):
    from qm import QuantumMachinesManager, generate_qua_script
    metrics = check_waveforms(settings, machine)
    QuantumMachinesManager.set_capabilities_offline()
    calibration = EchoLorentzian(
        parameters=parameters_for_beta(settings, metrics["worst_beta"]),
        machine=machine, qubit=settings.qubit, auto_connect=False,
        options=calibration_options(),
    )
    program = calibration.create_qua_program()
    script = generate_qua_script(program, machine.generate_config())
    np.testing.assert_array_equal(calibration.namespace["sweep_axes"]["amp_prefactor"],
                                  settings.amplitude_factors)
    return {**metrics, "qua_script_generated": bool(script)}


def fit_and_score(dataset, *, beta, settings):
    """Reuse the existing dip fitter; reject a beta if any amplitude fit fails."""
    np.testing.assert_allclose(dataset.amp_prefactor, settings.amplitude_factors, rtol=0, atol=1e-12)
    fitted = add_simple_negative_gaussian_fits(dataset)
    def values(suffix):
        return np.asarray(fitted[f"gaussian_negative_{suffix}"].sel(qubit=settings.qubit), float)
    centers, errors = values("center_hz"), values("center_error_hz")
    widths, contrasts = values("fwhm_hz"), values("fit_contrast")
    r_squared = values("fit_r_squared")
    accepted = (np.isfinite(centers) & np.isfinite(errors) & (errors > 0)
                & np.isfinite(widths) & (widths > 0)
                & np.isfinite(contrasts) & (contrasts >= settings.min_contrast)
                & np.isfinite(r_squared) & (r_squared >= settings.min_r_squared))
    valid = bool(np.all(accepted))
    reference, score = None, None
    if valid:
        weights = (errors.min() / errors)**2
        reference = float(np.average(centers, weights=weights))
        score = float(np.sqrt(np.mean((centers-reference)**2)))
    record = dict(beta=float(beta), accepted=accepted.tolist(), valid=valid,
                  center_hz=centers.tolist(), center_error_hz=errors.tolist(),
                  fwhm_hz=widths.tolist(), contrast=contrasts.tolist(),
                  r_squared=r_squared.tolist(), weighted_center_hz=reference,
                  center_rms_hz=score)
    return record, fitted


def select_best(records):
    eligible = [r for r in records if r["valid"] and r["center_rms_hz"] is not None
                and np.isfinite(r["center_rms_hz"])]
    return min(eligible, key=lambda r: r["center_rms_hz"]) if eligible else None


def write_json(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_trace_plot(fitted, record, settings, path):
    count = settings.amplitude_points
    columns = min(count, 3)
    rows = (count + columns - 1) // columns
    fig, grid = plt.subplots(rows, columns, figsize=(4*columns, 3.6*rows),
                             layout="constrained", squeeze=False)
    axes = grid.ravel()
    detuning = np.asarray(fitted.detuning, float)
    for index, (axis, amp) in enumerate(zip(axes, settings.amplitude_factors)):
        trace = fitted.sel(qubit=settings.qubit).isel(amp_prefactor=index)
        axis.plot(detuning/1e3, trace.state, ".", markersize=3, label="Measured")
        center = record["center_hz"][index]
        if np.isfinite(center):
            sigma = record["fwhm_hz"][index] / (2*np.sqrt(2*np.log(2)))
            curve = float(trace.gaussian_negative_offset) - record["contrast"][index] * np.exp(
                -0.5*((detuning-center)/sigma)**2)
            axis.plot(detuning/1e3, curve, label="Existing Gaussian fit")
            axis.axvline(center/1e3, color="gray", linestyle=":")
        status = "accepted" if record["accepted"][index] else "rejected"
        axis.set(title=f"Amplitude ×{amp:g}: {status}", xlabel="Detuning (kHz)", ylabel="Excited probability")
        axis.grid(alpha=.2)
    for axis in axes[count:]:
        axis.set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{settings.qubit}: echo DRAG beta = {record['beta']:+.6f}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_2d_plot(fitted, record, settings, path):
    """Map and fitted resonance line for one beta, on the same amplitude axis."""
    trace = fitted.sel(qubit=settings.qubit)
    amplitudes = settings.amplitude_factors
    centers = np.asarray(record["center_hz"], float) / 1e3
    errors = np.asarray(record["center_error_hz"], float) / 1e3
    accepted = np.asarray(record["accepted"], bool)
    centers = np.where(accepted, centers, np.nan)
    errors = np.where(accepted, errors, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, layout="constrained")
    mesh = axes[0].pcolormesh(np.asarray(trace.detuning)/1e3, amplitudes,
                             trace.state.transpose("amp_prefactor", "detuning"),
                             shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=axes[0], label="Excited probability")
    axes[0].plot(centers, amplitudes, "w.-", label="Fitted dip center")
    axes[0].set(xlabel="Detuning (kHz)", ylabel="Amplitude factor")
    axes[0].legend(fontsize=8)
    axes[1].errorbar(centers, amplitudes, xerr=errors, fmt="o-", capsize=2)
    reference = record["weighted_center_hz"]
    if reference is not None:
        axes[1].axvline(reference/1e3, color="gray", linestyle="--", label="Weighted mean")
        axes[1].legend(fontsize=8)
    axes[1].set(xlabel="Fitted resonance detuning (kHz)")
    axes[1].grid(alpha=.25)
    score = record["center_rms_hz"]
    label = f"center RMS = {score/1e3:.2f} kHz" if score is not None else "rejected: incomplete fits"
    fig.suptitle(f"{settings.qubit}: echo beta = {record['beta']:+.6f}; {label}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_scan(fitted, record, settings, path):
    """Compact 2D data and fitted centers, for reanalysis without hardware."""
    trace = fitted.sel(qubit=settings.qubit)
    np.savez_compressed(path, beta=record["beta"],
        detuning_hz=np.asarray(trace.detuning), amplitude_factor=settings.amplitude_factors,
        state=np.asarray(trace.state.transpose("amp_prefactor", "detuning")),
        center_hz=record["center_hz"], center_error_hz=record["center_error_hz"],
        accepted=record["accepted"])


def save_summary(output, records, settings, *, complete):
    best = select_best(records)
    summary = dict(complete=complete, measured_betas=len(records), requested_betas=settings.beta_points,
                   best_beta=best["beta"] if complete and best else None,
                   best_record=best, score="RMS centers about covariance-weighted mean (Hz)",
                   best_at_boundary=bool(best and best["beta"] in (settings.beta_min, settings.beta_max)),
                   amplitude_factors=settings.amplitude_factors.tolist())
    write_json(output / "summary.json", summary)
    write_json(output / "records.json", records)
    with (output / "fits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["beta", "amplitude_factor", "accepted", "center_hz", "center_error_hz",
                         "fwhm_hz", "contrast", "r_squared", "center_rms_hz"])
        for r in records:
            for i, amp in enumerate(settings.amplitude_factors):
                writer.writerow([r["beta"], amp, r["accepted"][i], r["center_hz"][i],
                                 r["center_error_hz"][i], r["fwhm_hz"][i], r["contrast"][i],
                                 r["r_squared"][i], r["center_rms_hz"]])
    if records:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), layout="constrained")
        betas = [r["beta"] for r in records]
        for i, amp in enumerate(settings.amplitude_factors):
            centers = [r["center_hz"][i]/1e3 if r["accepted"][i] else np.nan for r in records]
            errors = [r["center_error_hz"][i]/1e3 if r["accepted"][i] else np.nan for r in records]
            axes[0].errorbar(betas, centers, yerr=errors, fmt=".-", label=f"Amplitude ×{amp:g}")
        axes[0].set(ylabel="Fitted resonance detuning (kHz)")
        axes[0].legend(fontsize=8, ncols=3 if settings.amplitude_points > 5 else 1)
        axes[1].plot(betas, [r["center_rms_hz"]/1e3 if r["valid"] else np.nan for r in records], "o-")
        axes[1].set(xlabel="DRAG beta", ylabel="Center RMS across amplitudes (kHz)")
        for axis in axes:
            axis.grid(alpha=.2)
            if best:
                axis.axvline(best["beta"], color="black", linestyle="--")
        fig.suptitle(f"{settings.qubit}: {'complete' if complete else 'partial'} beta sweep")
        fig.savefig(output / "beta_summary.png", dpi=150)
        plt.close(fig)
    return summary


def run_sweep(settings, machine, output_root=DEFAULT_OUTPUT_ROOT, *, plot_mode="slices"):
    if plot_mode not in ("slices", "2d"):
        raise ValueError("plot_mode must be slices or 2d.")
    metrics = check_waveforms(settings, machine)
    output = Path(output_root) / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "settings.json", {**asdict(settings), **metrics,
        "plot_mode": plot_mode,
        "parameters_at_first_beta": parameters_for_beta(settings, settings.betas[0]).model_dump(mode="json"),
        "betas": settings.betas.tolist(), "amplitude_factors": settings.amplitude_factors.tolist()})
    saver = CalibrationSaver(output_root=output / "raw")
    records = []
    complete = False
    try:
        for index, beta in enumerate(settings.betas):
            print(f"[{index+1}/{settings.beta_points}] beta={beta:+.6f}: "
                  f"{settings.amplitude_points} slices", flush=True)
            calibration = EchoLorentzian(
                parameters=parameters_for_beta(settings, beta), machine=machine,
                qubit=settings.qubit, auto_connect=False, saver=saver,
                options=calibration_options(acquire=True), name=f"echo_beta_{index:03d}",
            )
            status = calibration.run()
            if status.interrupted:
                raise RuntimeError("Acquisition interrupted; partial results retained, no optimum declared.")
            record, fitted = fit_and_score(calibration.results["ds_raw"], beta=beta, settings=settings)
            record["raw_directory"] = str(calibration.namespace["calibration_run_directory"])
            records.append(record)
            save_summary(output, records, settings, complete=False)
            if plot_mode == "2d":
                save_scan(fitted, record, settings, output / f"beta_{index:03d}_scan.npz")
                save_2d_plot(fitted, record, settings, output / f"beta_{index:03d}_map.png")
            else:
                save_trace_plot(fitted, record, settings, output / f"beta_{index:03d}_slices.png")
            print(f"  center RMS: {record['center_rms_hz']} Hz; fits accepted: {record['accepted']}", flush=True)
        complete = True
    finally:
        summary = save_summary(output, records, settings, complete=complete)
        print(f"Saved results: {output}", flush=True)
    if summary["best_beta"] is None:
        raise RuntimeError("No beta has all amplitude fits accepted; inspect saved slices and fit quality.")
    return output, summary


def build_parser(*, defaults=None, output_root=DEFAULT_OUTPUT_ROOT, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Acquire on hardware.")
    mode.add_argument("--dry-run", action="store_true", help="Offline checks only (default).")
    parser.add_argument("--output-root", type=Path, default=output_root)
    for name, value in asdict(defaults or SweepSettings()).items():
        kwargs = dict(default=value, type=type(value))
        if name == "edge_smoothing":
            kwargs["choices"] = ("none", "polynomial")
        elif name == "beta_order":
            kwargs["choices"] = ("ascending", "descending")
        elif name == "amplitude_spacing":
            kwargs["choices"] = ("linear", "log")
        parser.add_argument("--" + name.replace("_", "-"), **kwargs)
    return parser


def main(argv=None, *, defaults=None, output_root=DEFAULT_OUTPUT_ROOT,
         plot_mode="slices", description=__doc__):
    args = vars(build_parser(defaults=defaults, output_root=output_root,
                             description=description).parse_args(argv))
    execute = args.pop("execute")
    args.pop("dry_run")
    output_root = args.pop("output_root")
    settings = SweepSettings(**args)
    machine = create_machine(qubit=settings.qubit)
    if not execute:
        print(json.dumps({"mode": "dry-run", **asdict(settings),
            "amplitude_factors": settings.amplitude_factors.tolist(),
            "total_slices": settings.amplitude_points*settings.beta_points,
            "points_per_beta": settings.amplitude_points*settings.frequency_points,
            "total_shots": settings.beta_points*settings.amplitude_points*settings.frequency_points*settings.num_shots,
            "pulse_parameters_at_first_beta": parameters_for_beta(settings, settings.betas[0]).model_dump(mode="json"),
            "beta_step": float(settings.betas[1]-settings.betas[0]),
            **offline_check(settings, machine)}, indent=2))
        return
    output, summary = run_sweep(settings, machine, output_root, plot_mode=plot_mode)
    print(f"Best sampled beta: {summary['best_beta']:+.6f}; "
          f"center RMS = {summary['best_record']['center_rms_hz']:.3f} Hz")
    if summary["best_at_boundary"]:
        print("The minimum is at the sweep boundary; a wider beta range may be needed.")


if __name__ == "__main__":
    main()
