"""Scan cutoff at fixed pulse length; measure dip FWHM and contrast (Figure 4).

Defaults to offline waveform/QUA validation. --execute acquires one detuning
versus amplitude map per cutoff with the existing BaseCalibration subclass.
Current shared echo settings are used; no profile or shared defaults are updated.
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
from shaped_pulse_spectroscopy.cutoff_metrics import measure_dip
from shaped_pulse_spectroscopy.echo_parameters import DEFAULT_QUBIT, ECHO_PULSE, echo_parameters
from shaped_pulse_spectroscopy.lorentzian import (
    amplitude_prefactors, build_waveform, drag_quadrature_waveform,
)
from quam_config import create_machine
from utils.rabi_amplitude import amplitude_to_rabi_frequency_hz, pi_pulse_effective_length_ns

DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "cutoff_dip_scan"


@dataclass(frozen=True)
class ScanSettings:
    profile: str = "iqm_chip"
    qubit: str = DEFAULT_QUBIT
    pulse_length_ns: int = 30_000
    cutoff_min: float = 0.001
    cutoff_max: float = 1.0
    cutoff_points: int = 31
    amplitude_min: float = 0.2
    amplitude_max: float = 1.0
    amplitude_points: int = 13
    peak_amplitude_v: float = ECHO_PULSE.lorentzian_peak_amplitude
    drag_beta: float = ECHO_PULSE.drag_beta
    transition_ns: float = ECHO_PULSE.echo_transition_time_ns
    frequency_span_mhz: float = 1.0
    frequency_points: int = 201
    num_shots: int = 1000
    min_contrast: float = 0.001
    min_width_intervals: int = 6
    t2_reference_us: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values() if isinstance(v, (int, float))):
            raise ValueError("Scan settings must be finite.")
        for name in ("pulse_length_ns", "cutoff_points", "amplitude_points", "frequency_points",
                     "num_shots", "min_width_intervals"):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f"{name} must be an integer.")
        if not self.qubit or self.pulse_length_ns < 16 or self.pulse_length_ns % 4:
            raise ValueError("Specify a qubit and pulse length >=16 ns divisible by 4.")
        if not 0 < self.cutoff_min < self.cutoff_max <= 1 or self.cutoff_points < 2:
            raise ValueError("Require 0 < cutoff_min < cutoff_max <= 1 and at least two cutoffs.")
        if not 0 < self.amplitude_min <= self.amplitude_max < 2 or self.amplitude_points < 1:
            raise ValueError("Require positive amplitude factors below 2 and at least one amplitude.")
        if (self.amplitude_points == 1) != (self.amplitude_min == self.amplitude_max):
            raise ValueError("For one amplitude use equal min/max; for multiple amplitudes use min < max.")
        if not 0 < self.peak_amplitude_v <= 1 or not 0 < self.transition_ns <= self.pulse_length_ns:
            raise ValueError("Require nominal amplitude in (0, 1] V and a positive transition within the pulse.")
        if self.frequency_span_mhz < 1 or self.frequency_points < 21 or self.num_shots < 1:
            raise ValueError("Require span >=1 MHz, >=21 frequencies, and positive shots.")
        if self.min_contrast <= 0 or self.min_width_intervals < 1 or self.t2_reference_us < 0:
            raise ValueError("Require positive quality thresholds and nonnegative T2 reference.")
        # Full-length templates sample every ns. Reject sub-grid pulse cores.
        sigma_ns = self.pulse_length_ns / 2 * self.cutoff_min / np.sqrt(1-self.cutoff_min**2)
        if sigma_ns < 2:
            raise ValueError("Smallest cutoff gives a core width below 2 ns; increase cutoff or pulse length.")
        # Validate analysis-window coverage before any connection.
        measure_dip(self.detunings_hz, np.full(self.frequency_points, 0.5))

    @property
    def cutoffs(self):
        return np.geomspace(self.cutoff_min, self.cutoff_max, self.cutoff_points)

    @property
    def amplitudes(self):
        return np.linspace(self.amplitude_min, self.amplitude_max, self.amplitude_points)

    @property
    def detunings_hz(self):
        span_hz = round(self.frequency_span_mhz * 1e6)
        return np.rint(np.linspace(-span_hz / 2, span_hz / 2, self.frequency_points)).astype(int)


def parameters_for_cutoff(settings, cutoff):
    return echo_parameters(
        pulse_shape="root_lorentzian", echo=True, cutoff=float(cutoff),
        lorentzian_length_in_ns=settings.pulse_length_ns,
        waveform_template_length_in_ns=settings.pulse_length_ns,
        lorentzian_peak_amplitude=settings.peak_amplitude_v,
        drag_beta=settings.drag_beta, echo_transition_time_ns=settings.transition_ns,
        edge_smoothing="polynomial", edge_smoothing_power=5,
        ac_stark_correction=False, stark_kappa_mhz_inv=0.0,
        min_amp_factor=settings.amplitude_min, max_amp_factor=settings.amplitude_max,
        amp_factor_points=settings.amplitude_points, amp_factor_spacing="linear",
        frequency_span_in_mhz=settings.frequency_span_mhz,
        frequency_points=settings.frequency_points, num_shots=settings.num_shots,
        fit_fwhm=False, plot_sweep_diagnostics=False,
        use_state_discrimination=True, use_three_state_discrimination=False,
        use_readout_mitigation=0, simulate=False,
    )


def calibration_options(*, acquire=False):
    return CalibrationOptions(
        save_raw_data=acquire, save_analysis_result=False, save_figures=False,
        analyse_data=False, plot_data=False, update_state=False,
        propose_profile_update=False, apply_profile_update=False, ai_review=False,
        report_runtime_estimate=acquire,
    )


def validate_offline(settings, machine):
    """Check every I/Q waveform and generate QUA/config offline for each cutoff."""
    from qm import QuantumMachinesManager, generate_qua_script
    QuantumMachinesManager.set_capabilities_offline()
    qubit = machine.qubits[settings.qubit]
    pi = qubit.xy.operations["x180"]
    effective_ns = pi_pulse_effective_length_ns(pi)
    checks = []
    for index, cutoff in enumerate(settings.cutoffs):
        p = parameters_for_cutoff(settings, cutoff)
        i = np.asarray(build_waveform(p))
        q = np.asarray(drag_quadrature_waveform(
            i, drag_beta=settings.drag_beta, anharmonicity_mhz=abs(qubit.anharmonicity)/1e6,
            pulse_length_ns=settings.pulse_length_ns))
        peak = float(np.max(np.abs(i+1j*q)))
        if not np.isfinite(peak) or peak * settings.amplitude_max > 1:
            raise ValueError(f"cutoff={cutoff:g}: complete I/Q peak exceeds 1 V.")
        calibration = EchoLorentzian(parameters=p, machine=machine, qubit=settings.qubit,
                                    auto_connect=False, options=calibration_options())
        program = calibration.create_qua_program()
        if not generate_qua_script(program, machine.generate_config()):
            raise RuntimeError("Offline QUA generation failed.")
        np.testing.assert_allclose(amplitude_prefactors(p), settings.amplitudes)
        checks.append(dict(cutoff=float(cutoff), peak_complex_amplitude_v=peak,
                           nominal_rabi_mhz_per_factor=float(amplitude_to_rabi_frequency_hz(
                               settings.peak_amplitude_v, abs(pi.amplitude), effective_ns)/1e6),
                           peak_rabi_mhz_per_factor=float(amplitude_to_rabi_frequency_hz(
                               peak, abs(pi.amplitude), effective_ns)/1e6)))
        print(f"  checked cutoff {index+1}/{settings.cutoff_points}: {cutoff:.6g}", flush=True)
    return checks


def analyze_dataset(dataset, settings, check):
    data = dataset.sel(qubit=settings.qubit)
    np.testing.assert_allclose(data.amp_prefactor, settings.amplitudes, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(data.detuning, settings.detunings_hz)
    records = []
    for index, amplitude in enumerate(settings.amplitudes):
        metrics = measure_dip(data.detuning, data.state.isel(amp_prefactor=index),
                              min_contrast=settings.min_contrast,
                              min_intervals=settings.min_width_intervals)
        records.append(dict(cutoff=check["cutoff"], pulse_length_ns=settings.pulse_length_ns,
                            amp_prefactor=float(amplitude),
                            nominal_rabi_mhz=amplitude*check["nominal_rabi_mhz_per_factor"],
                            peak_rabi_mhz=amplitude*check["peak_rabi_mhz_per_factor"],
                            peak_complex_amplitude_v=amplitude*check["peak_complex_amplitude_v"],
                            **metrics,
                            fwhm_over_t2_limit=(metrics["fwhm_hz"]*np.pi*settings.t2_reference_us*1e-6
                                                if settings.t2_reference_us else np.nan)))
    return records


def write_json(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
        return v
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def save_summary(output, records, settings, *, complete):
    write_json(output / "summary.json", dict(complete=complete, measured_spectra=len(records),
               accepted_widths=sum(r["accepted"] for r in records), records=records))
    if not records:
        return
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    cutoffs = np.unique([r["cutoff"] for r in records])
    amps = settings.amplitudes
    width_key = "fwhm_over_t2_limit" if settings.t2_reference_us else "fwhm_hz"
    width = np.array([r[width_key] for r in records]).reshape(-1, len(amps)).T
    if not settings.t2_reference_us:
        width = width / 1e3
    contrast = np.array([r["contrast"] for r in records]).reshape(-1, len(amps)).T
    y = np.array([r["nominal_rabi_mhz"] for r in records[:len(amps)]])
    width_label = "FWHM / [1/(pi T2)]" if settings.t2_reference_us else "FWHM (kHz)"
    for axis, values, label, cmap in zip(axes[0], (width, contrast),
                                        (width_label, "Contrast (probability difference)"),
                                        ("magma", "viridis")):
        if len(cutoffs) > 1 and len(amps) > 1:
            log_x = np.log(cutoffs)
            x_edges = np.exp(np.r_[log_x[0]-(log_x[1]-log_x[0])/2,
                                  (log_x[:-1]+log_x[1:])/2,
                                  log_x[-1]+(log_x[-1]-log_x[-2])/2])
            y_edges = np.r_[y[0]-(y[1]-y[0])/2, (y[:-1]+y[1:])/2,
                            y[-1]+(y[-1]-y[-2])/2]
            plot = axis.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values), cmap=cmap)
            finite_contrast = contrast[np.isfinite(contrast)]
            if finite_contrast.size and finite_contrast.min() < 0.2 < finite_contrast.max():
                axis.contour(cutoffs, y, contrast, levels=[0.2], colors="white")
        else:
            plot = axis.scatter(np.repeat(cutoffs, len(amps)), np.tile(y, len(cutoffs)),
                                c=values.T.ravel(), cmap=cmap)
        fig.colorbar(plot, ax=axis, label=label)
        axis.set_ylabel("Nominal Rabi frequency (MHz)")
        if len(amps) > 1:
            axis.set_ylim(y[0], y[-1])
    # Show a few amplitude slices; all individual values are retained in CSV.
    for index in np.unique(np.linspace(0, len(amps)-1, min(5, len(amps))).astype(int)):
        axes[1, 0].plot(cutoffs, width[index], ".-", label=f"{y[index]:.2f} MHz")
        axes[1, 1].plot(cutoffs, contrast[index], ".-")
    axes[1, 0].set_ylabel(width_label)
    axes[1, 0].legend(fontsize="small")
    axes[1, 1].set_ylabel("Contrast (probability difference)")
    axes[1, 1].axhline(0.2, color="gray", linestyle=":")
    if settings.t2_reference_us:
        axes[1, 0].axhline(1, color="gray", linestyle=":")
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xlabel("Cutoff c")
        if len(cutoffs) > 1:
            axis.set_xlim(cutoffs[0], cutoffs[-1])
    fig.suptitle(f"{settings.qubit}: {settings.pulse_length_ns/1000:g} us echo; "
                 f"{'complete' if complete else 'partial'} scan")
    fig.savefig(output / "cutoff_fwhm_contrast.png", dpi=160)
    plt.close(fig)


def run_scan(settings, machine, checks, output_root):
    output = Path(output_root) / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "settings.json", dict(
        **asdict(settings), cutoffs=settings.cutoffs.tolist(),
        amplitude_factors=settings.amplitudes.tolist(), waveform_checks=checks,
        first_pulse_parameters=parameters_for_cutoff(settings, settings.cutoffs[0]).model_dump(mode="json"),
        signal="unmitigated two-state readout; not separately resolved Pe and Pf",
        metric="linear background 0.4 <= |detuning MHz| <= 0.5; central search +/-0.25 MHz; nearest half-depth crossings",
        reference="amplitude-robust-spectrscopy/paper/figures/04_duration_cutoff_scan_float.tex",
    ))
    saver = CalibrationSaver(output_root=output / "raw")
    records, complete = [], False
    try:
        for index, check in enumerate(checks):
            print(f"[{index+1}/{len(checks)}] acquiring cutoff={check['cutoff']:.6g}", flush=True)
            calibration = EchoLorentzian(
                parameters=parameters_for_cutoff(settings, check["cutoff"]), machine=machine,
                qubit=settings.qubit, auto_connect=False, options=calibration_options(acquire=True),
                saver=saver, name=f"cutoff_dip_{index:03d}")
            status = calibration.run()
            if status.interrupted:
                raise RuntimeError("Acquisition interrupted; completed cutoff results retained.")
            dataset = calibration.results["ds_raw"]
            rows = analyze_dataset(dataset, settings, check)
            for row in rows:
                row["raw_directory"] = str(calibration.namespace["calibration_run_directory"])
            records.extend(rows)
            data = dataset.sel(qubit=settings.qubit)
            np.savez_compressed(output / f"cutoff_{index:03d}_scan.npz",
                                cutoff=check["cutoff"], detuning_hz=data.detuning.values,
                                amplitude_factors=settings.amplitudes,
                                state=data.state.transpose("amp_prefactor", "detuning").values)
            save_summary(output, records, settings, complete=False)
        complete = True
    finally:
        save_summary(output, records, settings, complete=complete)
        print(f"Results: {output}", flush=True)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Acquire on hardware after offline validation.")
    mode.add_argument("--dry-run", action="store_true", help="Offline validation only (default).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    for name, value in asdict(ScanSettings()).items():
        parser.add_argument("--"+name.replace("_", "-"), type=type(value), default=value)
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    execute, output_root = args.pop("execute"), args.pop("output_root")
    args.pop("dry_run")
    settings = ScanSettings(**args)
    print(f"{'EXECUTE' if execute else 'DRY RUN'}: {settings.pulse_length_ns/1000:g} us, "
          f"{settings.cutoff_points} cutoffs x {settings.amplitude_points} amplitudes x "
          f"{settings.frequency_points} detunings x {settings.num_shots} shots", flush=True)
    machine = create_machine(profile_name=settings.profile, qubit=settings.qubit)
    checks = validate_offline(settings, machine)
    if execute:
        run_scan(settings, machine, checks, output_root)
    else:
        print("Offline waveform and QUA checks passed. Add --execute to acquire.")


if __name__ == "__main__":
    main()
