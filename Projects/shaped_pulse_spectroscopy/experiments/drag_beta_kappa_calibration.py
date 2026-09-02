"""Approval-gated DRAG-beta and AC-Stark-kappa calibration workflow.

The workflow has two separately approved hardware stages. Stage 1 measures the
legacy beta=0 reference and a symmetric beta scan at fixed kappa. Once those
results identify the best fixed-kappa beta, stage 2 materializes an exact local
beta-kappa grid around it and requires a new plan hash before acquisition.
Neither stage updates or proposes a device-profile change.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibrations.base import CalibrationOptions
from experiments.detuning_amplitude_sweep import EchoLorentzian
from experiments.stark_kappa_sweep import add_simple_negative_gaussian_fits
from shaped_pulse_spectroscopy.parameters import Parameters
from quam_config import Quam, create_machine
from utils.rabi_amplitude import amplitude_to_rabi_frequency_hz

DEFAULT_BETAS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
DEFAULT_JOINT_BETA_OFFSETS = (-0.25, 0.0, 0.25)
DEFAULT_JOINT_KAPPA_OFFSETS_MHZ_INV = (-0.0005, 0.0, 0.0005)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "drag_beta_kappa_calibration"


@dataclass(frozen=True)
class SpectroscopyGrid:
    """The detuning x amplitude grid reused for every beta-kappa pair."""

    detuning_span_mhz: float = 1.0
    detuning_points: int = 201
    min_amp_factor: float = 0.05
    max_amp_factor: float = 1.0
    amplitude_points: int = 20
    num_shots: int = 500


@dataclass(frozen=True)
class WaveformSpec:
    """Fixed waveform settings for the approval-gated calibration."""

    pulse_shape: str = "root_lorentzian"
    duration_ns: int = 20_000
    template_length_ns: int = 20_000
    cutoff: float = 0.0025
    peak_amplitude_v: float = 0.2
    echo_transition_time_ns: float = 16.0
    complex_envelope_sign: str = "I+iQ"
    derivative_convention: str = (
        "Q=-beta*dI/dt/(2*pi*alpha_MHz), physical t in microseconds"
    )
    kappa_convention: str = "real-time detuning via QUA chirp"


@dataclass(frozen=True)
class CalibrationPoint:
    label: str
    category: Literal[
        "kappa_only",
        "drag_same_kappa",
        "drag_retuned_kappa",
        "no_correction",
    ]
    drag_beta: float
    stark_kappa_mhz_inv: float
    ac_stark_correction: bool = True


@dataclass(frozen=True)
class CalibrationPlan:
    stage: Literal["coarse", "joint"]
    target_qubit: str
    waveform: WaveformSpec
    grid: SpectroscopyGrid
    points: tuple[CalibrationPoint, ...]
    acceptable_center_rms_hz: float = 50_000.0
    minimum_contrast: float = 0.02


def coarse_plan(
    *,
    target_qubit: str,
    existing_kappa_mhz_inv: float,
    betas: Iterable[float] = DEFAULT_BETAS,
    include_no_correction: bool = False,
    waveform: WaveformSpec = WaveformSpec(),
    grid: SpectroscopyGrid = SpectroscopyGrid(),
) -> CalibrationPlan:
    """Build exact stage-A/B points, with beta=0 serving as the A reference."""
    beta_values = _finite_unique(betas, "betas")
    if not any(beta == 0.0 for beta in beta_values):
        raise ValueError("The symmetric beta scan must contain beta=0.")
    if not np.allclose(beta_values, -np.asarray(beta_values)[::-1]):
        raise ValueError("betas must be symmetric about zero and sorted ascending.")
    points = [
        CalibrationPoint(
            label=("kappa_only" if beta == 0.0 else f"drag_beta_{beta:+.9g}"),
            category=("kappa_only" if beta == 0.0 else "drag_same_kappa"),
            drag_beta=beta,
            stark_kappa_mhz_inv=float(existing_kappa_mhz_inv),
        )
        for beta in beta_values
    ]
    if include_no_correction:
        points.append(
            CalibrationPoint(
                label="no_correction",
                category="no_correction",
                drag_beta=0.0,
                stark_kappa_mhz_inv=0.0,
                ac_stark_correction=False,
            )
        )
    return CalibrationPlan(
        stage="coarse",
        target_qubit=_validated_target(target_qubit),
        waveform=waveform,
        grid=grid,
        points=tuple(points),
    )


def joint_plan(
    coarse: CalibrationPlan,
    *,
    best_fixed_kappa_beta: float,
    beta_offsets: Iterable[float] = DEFAULT_JOINT_BETA_OFFSETS,
    kappa_offsets_mhz_inv: Iterable[float] = DEFAULT_JOINT_KAPPA_OFFSETS_MHZ_INV,
) -> CalibrationPlan:
    """Materialize exact stage-C points around the measured stage-B winner."""
    if coarse.stage != "coarse":
        raise ValueError("joint_plan requires the approved coarse plan.")
    beta_offsets = _finite_unique(beta_offsets, "beta_offsets")
    kappa_offsets = _finite_unique(kappa_offsets_mhz_inv, "kappa_offsets_mhz_inv")
    baseline = next(point for point in coarse.points if point.category == "kappa_only")
    points = tuple(
        CalibrationPoint(
            label=f"joint_beta_{best_fixed_kappa_beta + db:+.9g}_kappa_{baseline.stark_kappa_mhz_inv + dk:+.9g}",
            category=("drag_same_kappa" if dk == 0.0 else "drag_retuned_kappa"),
            drag_beta=float(best_fixed_kappa_beta + db),
            stark_kappa_mhz_inv=float(baseline.stark_kappa_mhz_inv + dk),
        )
        for db in beta_offsets
        for dk in kappa_offsets
    )
    return CalibrationPlan(
        stage="joint",
        target_qubit=coarse.target_qubit,
        waveform=coarse.waveform,
        grid=coarse.grid,
        points=points,
        acceptable_center_rms_hz=coarse.acceptable_center_rms_hz,
        minimum_contrast=coarse.minimum_contrast,
    )


def plan_sha256(plan: CalibrationPlan) -> str:
    """Return the approval token for the exact target, waveform, and grids."""
    payload = json.dumps(asdict(plan), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_plan(plan: CalibrationPlan, path: Path) -> Path:
    """Write a reviewable plan manifest; this never connects to hardware."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = asdict(plan)
    manifest["plan_sha256"] = plan_sha256(plan)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def parameters_for_point(plan: CalibrationPlan, point: CalibrationPoint) -> Parameters:
    """Translate one approved point into the shared spectroscopy parameters."""
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.use_three_state_discrimination = True
    parameters.reset_type = "active"
    parameters.use_readout_mitigation = 0
    parameters.simulate = False
    parameters.pulse_shape = plan.waveform.pulse_shape
    parameters.echo = True
    parameters.ac_stark_correction = point.ac_stark_correction
    parameters.stark_kappa_mhz_inv = point.stark_kappa_mhz_inv
    parameters.drag_beta = point.drag_beta
    parameters.echo_transition_time_ns = plan.waveform.echo_transition_time_ns
    parameters.stark_chirp_max_error_hz = 100.0
    parameters.cutoff = plan.waveform.cutoff
    parameters.num_shots = plan.grid.num_shots
    parameters.lorentzian_length_in_ns = plan.waveform.duration_ns
    parameters.waveform_template_length_in_ns = plan.waveform.template_length_ns
    parameters.lorentzian_peak_amplitude = plan.waveform.peak_amplitude_v
    parameters.min_amp_factor = plan.grid.min_amp_factor
    parameters.max_amp_factor = plan.grid.max_amp_factor
    parameters.amp_factor_points = plan.grid.amplitude_points
    parameters.amp_factor_spacing = "linear"
    parameters.frequency_span_in_mhz = plan.grid.detuning_span_mhz
    parameters.frequency_points = plan.grid.detuning_points
    parameters.fit_fwhm = False
    return parameters


def evaluate_pair(
    dataset: xr.Dataset,
    *,
    qubit: str,
    rabi_frequency_mhz: np.ndarray,
) -> dict[str, Any]:
    """Extract center-vs-amplitude, RMS, contrast, and measured P_f."""
    fitted = add_simple_negative_gaussian_fits(dataset)
    centers = np.asarray(
        fitted.gaussian_negative_center_hz.sel(qubit=qubit), dtype=float
    )
    errors = np.asarray(
        fitted.gaussian_negative_center_error_hz.sel(qubit=qubit), dtype=float
    )
    contrasts = np.asarray(
        fitted.gaussian_negative_fit_contrast.sel(qubit=qubit), dtype=float
    )
    scores = np.asarray(
        fitted.gaussian_negative_fit_r_squared.sel(qubit=qubit), dtype=float
    )
    accepted = (
        np.isfinite(centers)
        & np.isfinite(errors)
        & (errors > 0)
        & np.isfinite(contrasts)
        & (contrasts > 0)
        & (scores >= 0.1)
    )
    if int(accepted.sum()) < 2:
        raise ValueError("Fewer than two valid center fits for this beta-kappa pair.")
    weights = 1.0 / errors[accepted] ** 2
    weighted_center = float(np.sum(weights * centers[accepted]) / np.sum(weights))
    center_rms = float(np.sqrt(np.mean((centers[accepted] - weighted_center) ** 2)))
    contrast = float(np.nanmedian(contrasts[accepted]))

    leakage_at_center: list[float] = []
    if "leakage" in dataset:
        leakage = dataset.leakage.sel(qubit=qubit)
        amplitudes = np.asarray(fitted.amp_prefactor.values, dtype=float)
        for index in np.flatnonzero(accepted):
            value = leakage.sel(
                amp_prefactor=amplitudes[index],
                detuning=centers[index],
                method="nearest",
            )
            leakage_at_center.append(float(value))

    return {
        "center_hz_vs_amplitude": centers.tolist(),
        "center_fit_accepted": accepted.tolist(),
        "rabi_frequency_mhz": np.asarray(rabi_frequency_mhz, dtype=float).tolist(),
        "center_rms_hz": center_rms,
        "weighted_center_hz": weighted_center,
        "spectroscopy_contrast": contrast,
        "leakage_available": bool(leakage_at_center),
        "mean_p_f_at_center": (
            float(np.mean(leakage_at_center)) if leakage_at_center else None
        ),
        "max_p_f_at_center": (
            float(np.max(leakage_at_center)) if leakage_at_center else None
        ),
    }


def select_best(
    records: Iterable[dict[str, Any]],
    *,
    acceptable_center_rms_hz: float,
    minimum_contrast: float,
) -> dict[str, Any]:
    """Minimize measured leakage subject to center-RMS and contrast gates."""
    eligible = [
        record
        for record in records
        if float(record["center_rms_hz"]) <= acceptable_center_rms_hz
        and float(record["spectroscopy_contrast"]) >= minimum_contrast
    ]
    if not eligible:
        raise ValueError("No pair satisfies the center-RMS and contrast constraints.")
    with_leakage = [
        record for record in eligible if record["max_p_f_at_center"] is not None
    ]
    if with_leakage:
        return min(
            with_leakage,
            key=lambda record: (
                float(record["max_p_f_at_center"]),
                float(record["center_rms_hz"]),
                -float(record["spectroscopy_contrast"]),
            ),
        )
    return min(
        eligible,
        key=lambda record: (
            float(record["center_rms_hz"]),
            -float(record["spectroscopy_contrast"]),
        ),
    )


def select_comparison_records(
    coarse_records: Iterable[dict[str, Any]],
    joint_records: Iterable[dict[str, Any]],
    *,
    acceptable_center_rms_hz: float,
    minimum_contrast: float,
) -> list[dict[str, Any]]:
    """Choose the four requested comparison traces with constrained scoring."""
    coarse_records = list(coarse_records)
    joint_records = list(joint_records)
    selected = [
        next(record for record in coarse_records if record["category"] == "kappa_only")
    ]
    fixed_drag = [
        record
        for record in coarse_records
        if record["category"] == "drag_same_kappa" and float(record["drag_beta"]) != 0.0
    ]
    if fixed_drag:
        selected.append(
            select_best(
                fixed_drag,
                acceptable_center_rms_hz=acceptable_center_rms_hz,
                minimum_contrast=minimum_contrast,
            )
        )
    if joint_records:
        selected.append(
            select_best(
                joint_records,
                acceptable_center_rms_hz=acceptable_center_rms_hz,
                minimum_contrast=minimum_contrast,
            )
        )
    no_correction = [
        record for record in coarse_records if record["category"] == "no_correction"
    ]
    if no_correction:
        selected.append(no_correction[0])
    return selected


def plot_comparison(records: Iterable[dict[str, Any]]):
    """Compare kappa-only, fixed-kappa DRAG, retuned DRAG, and no correction."""
    records = list(records)
    figure, axes_grid = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axes = axes_grid.ravel()
    for record in records:
        label = str(record["label"])
        rabi = np.asarray(record["rabi_frequency_mhz"], dtype=float)
        centers = np.asarray(record["center_hz_vs_amplitude"], dtype=float) / 1e3
        axes[0].plot(rabi, centers, "o-", markersize=3, label=label)
    axes[0].set(xlabel="Rabi frequency [MHz]", ylabel="Fitted center [kHz]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    labels = [str(record["label"]) for record in records]
    positions = np.arange(len(records))
    axes[1].bar(positions, [float(r["center_rms_hz"]) / 1e3 for r in records])
    axes[1].set(ylabel="Center RMS [kHz]", xticks=positions, xticklabels=labels)
    axes[2].bar(positions, [float(r["spectroscopy_contrast"]) for r in records])
    axes[2].set(
        ylabel="Median spectroscopy contrast",
        xticks=positions,
        xticklabels=labels,
    )
    axes[3].bar(
        positions,
        [
            np.nan if r["max_p_f_at_center"] is None else r["max_p_f_at_center"]
            for r in records
        ],
    )
    axes[3].set(
        ylabel="Max measured P_f at center",
        xticks=positions,
        xticklabels=labels,
    )
    for axis in axes[1:]:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def run_approved_plan(
    plan: CalibrationPlan,
    *,
    approved_plan_sha256: str,
    machine: Quam | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Acquire one exact approved stage without any persisted profile update."""
    expected_hash = plan_sha256(plan)
    if approved_plan_sha256 != expected_hash:
        raise PermissionError(
            "Hardware acquisition is locked: approved_plan_sha256 does not match "
            f"the exact plan ({expected_hash})."
        )
    output_dir = Path(output_root) / datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_plan(plan, output_dir / "approved_plan.json")
    machine = machine or create_machine(qubit=plan.target_qubit)
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
    records: list[dict[str, Any]] = []
    for index, point in enumerate(plan.points, start=1):
        parameters = parameters_for_point(plan, point)
        calibration = EchoLorentzian(
            parameters=parameters,
            options=options,
            machine=machine,
            auto_connect=True,
            name=f"drag_kappa_{plan.stage}_{index:02d}",
        )
        calibration.run()
        dataset = calibration.results["ds_raw"]
        qubit = next(
            q for q in calibration.namespace["qubits"] if q.name == plan.target_qubit
        )
        amplitudes_v = np.asarray(dataset.full_amp.sel(qubit=qubit.name), dtype=float)
        pi_pulse = qubit.xy.operations["x180"]
        rabi_mhz = (
            np.asarray(
                amplitude_to_rabi_frequency_hz(
                    amplitudes_v,
                    float(pi_pulse.amplitude),
                    float(pi_pulse.length),
                )
            )
            / 1e6
        )
        metric = evaluate_pair(dataset, qubit=qubit.name, rabi_frequency_mhz=rabi_mhz)
        records.append({**asdict(point), **metric})
        (output_dir / "records.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )
    return {"output_dir": output_dir, "records": records, "plan_sha256": expected_hash}


def _finite_unique(values: Iterable[float], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a non-empty finite sequence.")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates.")
    return result


def _validated_target(target: str) -> str:
    if not target or target.strip() != target:
        raise ValueError("target_qubit must be an explicit non-empty qubit name.")
    return target
