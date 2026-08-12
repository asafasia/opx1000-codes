"""Find the AC-Stark kappa that minimizes Echo-Lorentzian oscillations.

Each kappa value runs the full detuning-versus-amplitude experiment.  The
preferred oscillation score follows the amplitude-robust spectroscopy analysis:
fit the resonance center at every amplitude, quality-filter the fits in the
operating Rabi-frequency window, and calculate the RMS center residual about a
covariance-weighted mean center.  A smaller score means a more stable resonance.
"""

from __future__ import annotations

import csv
import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibrations.base import BaseCalibration, CalibrationOptions
from experiments.detuning_amplitude_sweep import EchoLorentzian
from shaped_pulse_spectroscopy.parameters import Parameters
from quam_config import Quam, create_machine

DEFAULT_KAPPAS_MHZ_INV = np.linspace(0.0, -0.03, 50)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "stark_kappa_sweep"
SIMPLE_GAUSSIAN_MAXFEV = 800
MIN_CENTER_CONTRAST = 0.02
MAX_CENTER_FWHM_HZ = 120e3
OPERATING_RABI_MIN_MHZ = 2.475
OPERATING_RABI_MAX_MHZ = 8.973


def adjacent_amplitude_rms(
    dataset: xr.Dataset,
    *,
    variable: str = "state",
) -> tuple[float, xr.DataArray]:
    """Return the global and per-amplitude RMS adjacent-response differences."""
    if variable not in dataset:
        raise ValueError(
            f"Dataset has no {variable!r} variable; available variables are "
            f"{list(dataset.data_vars)}."
        )
    response = dataset[variable]
    if "amp_prefactor" not in response.dims:
        raise ValueError(
            f"{variable!r} must have an 'amp_prefactor' dimension; got {response.dims}."
        )
    if response.sizes["amp_prefactor"] < 2:
        raise ValueError("At least two amplitude points are required for an RMS score.")

    differences = response.diff("amp_prefactor")
    reduction_dims = [dim for dim in differences.dims if dim != "amp_prefactor"]
    per_amplitude = np.sqrt((differences**2).mean(dim=reduction_dims, skipna=True))
    per_amplitude.name = "adjacent_amplitude_rms"
    per_amplitude.attrs.update(
        {
            "long_name": "RMS response difference from previous amplitude point",
            "response_variable": variable,
        }
    )
    global_rms = float(np.sqrt((differences**2).mean(skipna=True)))
    if not np.isfinite(global_rms):
        raise ValueError("The adjacent-amplitude RMS score is not finite.")
    return global_rms, per_amplitude


def add_simple_negative_gaussian_fits(dataset: xr.Dataset) -> xr.Dataset:
    """Fit one bounded negative Gaussian per trace with a strict work limit."""
    from scipy.optimize import curve_fit

    detuning = np.asarray(dataset.detuning.values, dtype=float)
    qubits = list(dataset.qubit.values)
    amplitudes = list(dataset.amp_prefactor.values)
    shape = (len(qubits), len(amplitudes))
    centers = np.full(shape, np.nan)
    widths = np.full(shape, np.nan)
    scores = np.full(shape, np.nan)
    contrasts = np.full(shape, np.nan)
    center_errors = np.full(shape, np.nan)
    if detuning.size < 5:
        return dataset

    span = float(np.ptp(detuning))
    step = float(np.median(np.diff(np.sort(detuning))))

    def negative_gaussian(x, offset, depth, center, sigma):
        return offset - depth * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    for qubit_index, qubit_name in enumerate(qubits):
        for amp_index, amp in enumerate(amplitudes):
            signal = np.asarray(
                dataset["state"].sel(qubit=qubit_name, amp_prefactor=amp).values,
                dtype=float,
            )
            finite = np.isfinite(detuning) & np.isfinite(signal)
            x = detuning[finite]
            y = signal[finite]
            if x.size < 5 or float(np.ptp(y)) <= 0:
                continue
            edge_points = min(40, max(0, (x.size - 7) // 4))
            if edge_points:
                x = x[edge_points:-edge_points]
                y = y[edge_points:-edge_points]
            from scipy.ndimage import gaussian_filter1d

            y = gaussian_filter1d(y, 1.0)
            local_span = float(np.ptp(x))
            local_step = float(np.median(np.diff(np.sort(x))))
            offset_guess = float(np.percentile(y, 80))
            depth_guess = max(offset_guess - float(np.min(y)), 0.1 * float(np.ptp(y)))
            center_guess = float(x[np.argmin(y)])
            sigma_guess = max(40e3, local_step)
            y_padding = float(np.ptp(y))
            try:
                params, covariance = curve_fit(
                    negative_gaussian,
                    x,
                    y,
                    p0=(offset_guess, depth_guess, center_guess, sigma_guess),
                    bounds=(
                        (
                            float(np.min(y)) - y_padding,
                            0,
                            float(np.min(x)),
                            local_step / 2,
                        ),
                        (
                            float(np.max(y)) + y_padding,
                            2 * y_padding,
                            float(np.max(x)),
                            local_span,
                        ),
                    ),
                    maxfev=SIMPLE_GAUSSIAN_MAXFEV,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            fitted = negative_gaussian(x, *params)
            residual_sum = float(np.sum((y - fitted) ** 2))
            total_sum = float(np.sum((y - np.mean(y)) ** 2))
            r_squared = 1 - residual_sum / total_sum if total_sum > 0 else np.nan
            fitted_fwhm = float(2 * np.sqrt(2 * np.log(2)) * abs(params[3]))
            center_error = float(np.sqrt(np.diag(covariance))[2])
            if (
                not np.isfinite(r_squared)
                or r_squared < 0.1
                or not np.isfinite(center_error)
                or center_error <= 0
            ):
                continue
            centers[qubit_index, amp_index] = float(params[2])
            widths[qubit_index, amp_index] = fitted_fwhm
            scores[qubit_index, amp_index] = r_squared
            contrasts[qubit_index, amp_index] = float(params[1])
            center_errors[qubit_index, amp_index] = center_error

    return dataset.assign(
        gaussian_negative_center_hz=(("qubit", "amp_prefactor"), centers),
        gaussian_negative_fwhm_hz=(("qubit", "amp_prefactor"), widths),
        gaussian_negative_fit_r_squared=(("qubit", "amp_prefactor"), scores),
        gaussian_negative_fit_contrast=(("qubit", "amp_prefactor"), contrasts),
        gaussian_negative_center_error_hz=(
            ("qubit", "amp_prefactor"),
            center_errors,
        ),
    )


def amplitude_robust_center_rms(
    dataset: xr.Dataset,
    rabi_frequency_mhz: np.ndarray,
    *,
    qubit: str = "q1",
) -> dict[str, Any]:
    """RMS of fitted-center residuals, matching the amplitude-robust analysis."""
    fitted = add_simple_negative_gaussian_fits(dataset)
    centers = np.asarray(fitted.gaussian_negative_center_hz.sel(qubit=qubit), dtype=float)
    errors = np.asarray(
        fitted.gaussian_negative_center_error_hz.sel(qubit=qubit), dtype=float
    )
    widths = np.asarray(fitted.gaussian_negative_fwhm_hz.sel(qubit=qubit), dtype=float)
    contrasts = np.asarray(
        fitted.gaussian_negative_fit_contrast.sel(qubit=qubit), dtype=float
    )
    scores = np.asarray(
        fitted.gaussian_negative_fit_r_squared.sel(qubit=qubit), dtype=float
    )
    rabi_frequency_mhz = np.asarray(rabi_frequency_mhz, dtype=float)
    accepted = (
        np.isfinite(centers)
        & np.isfinite(errors)
        & (errors > 0)
        & (contrasts >= MIN_CENTER_CONTRAST)
        & (widths <= MAX_CENTER_FWHM_HZ)
        & (scores >= 0.1)
        & (rabi_frequency_mhz >= OPERATING_RABI_MIN_MHZ)
        & (rabi_frequency_mhz <= OPERATING_RABI_MAX_MHZ)
    )
    if int(accepted.sum()) < 2:
        raise ValueError("Fewer than two accepted center fits in the operating window.")
    weights = 1 / errors[accepted] ** 2
    weighted_mean_hz = float(np.sum(weights * centers[accepted]) / np.sum(weights))
    residuals_hz = centers[accepted] - weighted_mean_hz
    return {
        "center_residual_rms_hz": float(np.sqrt(np.mean(residuals_hz**2))),
        "weighted_center_hz": weighted_mean_hz,
        "weighted_center_error_hz": float(1 / np.sqrt(np.sum(weights))),
        "max_abs_center_residual_hz": float(np.max(np.abs(residuals_hz))),
        "accepted_points": int(accepted.sum()),
        "fitted_dataset": fitted,
        "accepted": accepted,
    }


def run_stark_kappa_sweep(
    *,
    kappas_mhz_inv: Iterable[float] = DEFAULT_KAPPAS_MHZ_INV,
    machine: Quam | None = None,
    qubit: str = "q1",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output_dir: Path | None = None,
    max_attempts_per_kappa: int = 3,
) -> dict[str, Any]:
    """Run the hardware sweep and save a CSV plus RMS-versus-kappa plot."""
    kappas = np.asarray(list(kappas_mhz_inv), dtype=float)
    if kappas.ndim != 1 or kappas.size == 0 or not np.all(np.isfinite(kappas)):
        raise ValueError(
            "kappas_mhz_inv must be a non-empty 1D sequence of finite values."
        )

    if max_attempts_per_kappa < 1:
        raise ValueError("max_attempts_per_kappa must be at least one.")
    if output_dir is None:
        output_dir = output_root / datetime.now().astimezone().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    machine = machine or create_machine(qubit=qubit)
    records: list[dict[str, float | int | str]] = _read_csv(
        output_dir / "kappa_rms.csv"
    )
    per_amplitude_records: list[dict[str, float | int | str]] = _read_csv(
        output_dir / "kappa_rms_per_amplitude.csv"
    )
    completed_indices = {int(record["run_index"]) for record in records}
    for record in records:
        index = int(record["run_index"])
        if (
            index < 1
            or index > len(kappas)
            or not np.isclose(float(record["stark_kappa_mhz_inv"]), kappas[index - 1])
        ):
            raise ValueError(
                "Checkpointed kappa_rms.csv does not match the requested kappa sweep."
            )

    # Preserve every raw hardware result, but suppress ten interactive plots and
    # prohibit this diagnostic from changing or proposing changes to the profile.
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

    try:
        for run_index, kappa in enumerate(kappas, start=1):
            if run_index in completed_indices:
                print(
                    f"Kappa sweep {run_index}/{len(kappas)} already completed; "
                    "using checkpointed result."
                )
                continue
            print(
                f"Kappa sweep {run_index}/{len(kappas)}: "
                f"stark_kappa_mhz_inv={kappa:.9g}"
            )
            parameters = _experiment_parameters(float(kappa))
            for attempt in range(1, max_attempts_per_kappa + 1):
                calibration = EchoLorentzian(
                    parameters=parameters,
                    options=options,
                    machine=machine,
                    auto_connect=not parameters.simulate,
                    name=f"echo_lorentzian_kappa_{run_index:02d}",
                )
                try:
                    calibration.run()
                    break
                except Exception as error:
                    if attempt == max_attempts_per_kappa:
                        raise
                    print(
                        f"  acquisition attempt {attempt} failed with "
                        f"{type(error).__name__}: {error}; retrying."
                    )

            dataset = calibration.results["ds_raw"]
            rms, per_amplitude = adjacent_amplitude_rms(dataset)
            run_directory = calibration.namespace.get("calibration_run_directory")
            response_figure_path = output_dir / f"kappa_{run_index:02d}_response.png"
            response_figure = _plot_kappa_response(
                dataset,
                per_amplitude,
                kappa=float(kappa),
                global_rms=rms,
                qubit=qubit,
            )
            response_figure.savefig(response_figure_path, dpi=200, bbox_inches="tight")
            plt.close(response_figure)
            records.append(
                {
                    "run_index": run_index,
                    "stark_kappa_mhz_inv": float(kappa),
                    "adjacent_amplitude_rms": rms,
                    "run_directory": (
                        "" if run_directory is None else str(run_directory)
                    ),
                    "response_figure": str(response_figure_path),
                }
            )
            amp_values = np.asarray(per_amplitude.amp_prefactor.values, dtype=float)
            for amp_prefactor, amp_rms in zip(
                amp_values, np.asarray(per_amplitude.values, dtype=float), strict=True
            ):
                per_amplitude_records.append(
                    {
                        "run_index": run_index,
                        "stark_kappa_mhz_inv": float(kappa),
                        "amp_prefactor": float(amp_prefactor),
                        "adjacent_amplitude_rms": float(amp_rms),
                    }
                )
            _write_csv(output_dir / "kappa_rms.csv", records)
            _write_csv(
                output_dir / "kappa_rms_per_amplitude.csv", per_amplitude_records
            )
            print(f"  adjacent-amplitude RMS = {rms:.6g}")
    except KeyboardInterrupt:
        print("\nKappa sweep interrupted; plotting all completed points.")

    if not records:
        raise RuntimeError("No kappa points completed, so no RMS plot can be made.")

    best_record = min(
        records, key=lambda record: float(record["adjacent_amplitude_rms"])
    )
    figure_path = output_dir / "kappa_rms.png"
    figure = _plot_summary(records, best_record)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(
        "Best measured kappa: "
        f"{float(best_record['stark_kappa_mhz_inv']):.9g} MHz^-1 "
        f"(RMS={float(best_record['adjacent_amplitude_rms']):.6g})"
    )
    print(f"Summary saved to {output_dir}")
    return {
        "output_dir": output_dir,
        "records": records,
        "best_record": best_record,
        "figure": figure,
        "figure_path": figure_path,
    }


def _experiment_parameters(kappa_mhz_inv: float) -> Parameters:
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.use_readout_mitigation = True

    parameters.simulate = False
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = True
    parameters.ac_stark_correction = True
    parameters.stark_kappa_mhz_inv = kappa_mhz_inv
    parameters.stark_chirp_max_error_hz = 100.0
    parameters.cutoff = 0.005
    parameters.num_shots = 50
    parameters.lorentzian_length_in_ns = 10000
    parameters.waveform_template_length_in_ns = 10000
    parameters.lorentzian_peak_amplitude = 0.2
    parameters.min_amp_factor = 0.0
    parameters.max_amp_factor = 1
    parameters.amp_factor_step = 0.01
    parameters.amp_factor_points = None
    parameters.amp_factor_spacing = "linear"
    parameters.frequency_span_in_mhz = 1
    parameters.frequency_step_in_mhz = 0.005
    parameters.frequency_points = None
    parameters.fit_fwhm = False
    return parameters


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _plot_summary(
    records: list[dict[str, float | int | str]],
    best_record: dict[str, float | int | str],
):
    kappas = np.asarray(
        [record["stark_kappa_mhz_inv"] for record in records], dtype=float
    )
    rms_values = np.asarray(
        [record["adjacent_amplitude_rms"] for record in records], dtype=float
    )
    best_kappa = float(best_record["stark_kappa_mhz_inv"])
    best_rms = float(best_record["adjacent_amplitude_rms"])

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(kappas, rms_values, "o-", linewidth=1.5)
    for kappa, rms in zip(kappas, rms_values, strict=True):
        axis.annotate(
            f"{kappa:.3g}\n{rms:.4g}",
            (kappa, rms),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    axis.scatter([best_kappa], [best_rms], marker="*", s=150, zorder=3, label="minimum")
    axis.axvline(best_kappa, color="tab:red", linestyle="--", alpha=0.6)
    axis.set_xlabel(r"AC-Stark $\kappa$ [MHz$^{-1}$]")
    axis.set_ylabel("RMS adjacent-amplitude response difference")
    axis.set_title("Echo root-Lorentzian oscillation score")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_kappa_response(
    dataset: xr.Dataset,
    per_amplitude_rms: xr.DataArray,
    *,
    kappa: float,
    global_rms: float,
    qubit: str,
    fitted_dataset: xr.Dataset | None = None,
):
    """Plot the response map and simple-fit centers/FWHM for one kappa."""
    if fitted_dataset is None:
        fitted_dataset = add_simple_negative_gaussian_fits(dataset)
    response = dataset["state"]
    if "qubit" in response.dims:
        response = response.sel(qubit=qubit)
    response = response.transpose("detuning", "amp_prefactor")
    detuning_mhz = np.asarray(response.detuning.values, dtype=float) / 1e6
    amplitudes = np.asarray(response.amp_prefactor.values, dtype=float)
    values = np.asarray(response.values, dtype=float)
    centers = fitted_dataset["gaussian_negative_center_hz"]
    widths = fitted_dataset["gaussian_negative_fwhm_hz"]
    scores = fitted_dataset["gaussian_negative_fit_r_squared"]
    if "qubit" in centers.dims:
        centers = centers.sel(qubit=qubit)
        widths = widths.sel(qubit=qubit)
        scores = scores.sel(qubit=qubit)
    center_mhz = np.asarray(centers.values, dtype=float) / 1e6
    fwhm_mhz = np.asarray(widths.values, dtype=float) / 1e6
    fit_scores = np.asarray(scores.values, dtype=float)
    valid = np.isfinite(center_mhz) & np.isfinite(fwhm_mhz)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    image = axes[0].pcolormesh(
        detuning_mhz, amplitudes, values.T, shading="auto", cmap="viridis"
    )
    figure.colorbar(image, ax=axes[0], label="Measured state")
    axes[0].errorbar(
        center_mhz[valid],
        amplitudes[valid],
        xerr=fwhm_mhz[valid] / 2,
        fmt="none",
        ecolor="magenta",
        elinewidth=0.8,
        alpha=0.65,
    )
    axes[0].scatter(
        center_mhz[valid],
        amplitudes[valid],
        marker="x",
        s=28,
        linewidths=1.3,
        color="magenta",
        label="center ± FWHM/2",
    )
    axes[0].set_xlabel("Detuning [MHz]")
    axes[0].set_ylabel("Amplitude prefactor")
    axes[0].set_title("Measured response and simple Gaussian fit")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].errorbar(
        amplitudes[valid],
        center_mhz[valid],
        yerr=fwhm_mhz[valid] / 2,
        fmt="none",
        ecolor="0.65",
        elinewidth=0.8,
        alpha=0.7,
        zorder=0,
    )
    center_plot = axes[1].scatter(
        amplitudes[valid],
        center_mhz[valid],
        c=fit_scores[valid],
        cmap="plasma",
        vmin=0,
        vmax=1,
        s=24,
        zorder=2,
    )
    axes[1].plot(
        amplitudes[valid], center_mhz[valid], color="0.55", linewidth=0.8, zorder=1
    )
    figure.colorbar(center_plot, ax=axes[1], label=r"Simple fit $R^2$")
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Amplitude prefactor")
    axes[1].set_ylabel("Center detuning [MHz]")
    axes[1].set_title(f"Centers and FWHM ({int(valid.sum())}/{len(amplitudes)})")
    axes[1].grid(alpha=0.25)

    figure.suptitle(
        f"{qubit} Echo root-Lorentzian: kappa={kappa:.9g} MHz^-1, "
        f"RMS={global_rms:.6g}"
    )
    figure.tight_layout()
    return figure


def regenerate_saved_response_figures(
    summary_directory: Path,
    *,
    qubit: str = "q1",
) -> Path:
    """Refit saved hardware data and replace per-kappa response figures."""
    summary_directory = Path(summary_directory)
    records = _read_csv(summary_directory / "kappa_rms.csv")
    if not records:
        raise FileNotFoundError(
            f"No kappa_rms.csv records found in {summary_directory}."
        )

    center_records: list[dict[str, float | int]] = []
    for record in records:
        run_index = int(record["run_index"])
        dataset = _load_saved_dataset(Path(record["run_directory"]))
        _, per_amplitude = adjacent_amplitude_rms(dataset)
        fitted_dataset = add_simple_negative_gaussian_fits(dataset)
        centers = fitted_dataset["gaussian_negative_center_hz"]
        scores = fitted_dataset["gaussian_negative_fit_r_squared"]
        widths = fitted_dataset["gaussian_negative_fwhm_hz"]
        if "qubit" in centers.dims:
            centers = centers.sel(qubit=qubit)
            scores = scores.sel(qubit=qubit)
            widths = widths.sel(qubit=qubit)
        for amp, center, width, score in zip(
            np.asarray(centers.amp_prefactor.values, dtype=float),
            np.asarray(centers.values, dtype=float),
            np.asarray(widths.values, dtype=float),
            np.asarray(scores.values, dtype=float),
            strict=True,
        ):
            center_records.append(
                {
                    "run_index": run_index,
                    "stark_kappa_mhz_inv": float(record["stark_kappa_mhz_inv"]),
                    "amp_prefactor": float(amp),
                    "negative_gaussian_center_hz": float(center),
                    "negative_gaussian_fwhm_hz": float(width),
                    "negative_gaussian_r_squared": float(score),
                }
            )

        figure = _plot_kappa_response(
            dataset,
            per_amplitude,
            kappa=float(record["stark_kappa_mhz_inv"]),
            global_rms=float(record["adjacent_amplitude_rms"]),
            qubit=qubit,
            fitted_dataset=fitted_dataset,
        )
        figure.savefig(
            summary_directory / f"kappa_{run_index:02d}_response.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)
        print(f"Regenerated response figure {run_index}/{len(records)}")

    centers_path = summary_directory / "negative_gaussian_centers.csv"
    _write_csv(centers_path, center_records)
    return centers_path


def analyze_saved_center_stability(
    summary_directory: Path,
    *,
    qubit: str = "q1",
) -> Path:
    """Calculate the amplitude-robust center-residual RMS for every kappa."""
    summary_directory = Path(summary_directory)
    records = _read_csv(summary_directory / "kappa_rms.csv")
    output_records: list[dict[str, float | int]] = []
    for record in records:
        run_directory = Path(record["run_directory"])
        dataset = _load_saved_dataset(run_directory)
        parameters = json.loads(
            (run_directory / "parameters.json").read_text(encoding="utf-8")
        )
        pulses = json.loads(
            (run_directory / "profile" / "pulses.json").read_text(encoding="utf-8")
        )
        x180 = pulses["pulses"][qubit]["x180_const"]
        pi_rabi_mhz = 1000 / (2 * float(x180["length_ns"]))
        rabi_frequency_mhz = (
            np.asarray(dataset.amp_prefactor.values, dtype=float)
            * float(parameters["lorentzian_peak_amplitude"])
            / float(x180["amplitude"])
            * pi_rabi_mhz
        )
        metric = amplitude_robust_center_rms(
            dataset,
            rabi_frequency_mhz,
            qubit=qubit,
        )
        output_records.append(
            {
                "run_index": int(record["run_index"]),
                "stark_kappa_mhz_inv": float(record["stark_kappa_mhz_inv"]),
                "center_residual_rms_khz": metric["center_residual_rms_hz"] / 1e3,
                "weighted_center_khz": metric["weighted_center_hz"] / 1e3,
                "weighted_center_error_khz": metric["weighted_center_error_hz"] / 1e3,
                "max_abs_center_residual_khz": metric[
                    "max_abs_center_residual_hz"
                ]
                / 1e3,
                "accepted_points": metric["accepted_points"],
            }
        )
        print(f"Center-stability metric {len(output_records)}/{len(records)}")
    output_path = summary_directory / "kappa_center_stability.csv"
    _write_csv(output_path, output_records)
    theoretical_kappa = _theoretical_kappa_from_saved_run(
        Path(records[0]["run_directory"]), qubit=qubit
    )
    figure = _plot_center_stability_summary(
        output_records, theoretical_kappa_mhz_inv=theoretical_kappa
    )
    figure.savefig(
        summary_directory / "kappa_center_stability.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)
    return output_path


def _theoretical_kappa_from_saved_run(run_directory: Path, *, qubit: str) -> float:
    """Return -1/|alpha| in MHz^-1 using the saved positive anharmonicity.

    Kappa multiplies a Rabi frequency expressed in MHz, so no extra 2*pi is
    needed.  The minus sign matches the implemented correction convention.
    """
    qubits = json.loads(
        (run_directory / "profile" / "qubits.json").read_text(encoding="utf-8")
    )
    anharmonicity_mhz = (
        float(qubits["qubits"][qubit]["transmon"]["anharmonicity_hz"]) / 1e6
    )
    return -1.0 / anharmonicity_mhz


def _plot_center_stability_summary(
    records: list[dict[str, float | int]],
    *,
    theoretical_kappa_mhz_inv: float | None = None,
):
    """Plot the amplitude-robust center-stability score versus kappa."""
    kappas = np.asarray(
        [record["stark_kappa_mhz_inv"] for record in records], dtype=float
    )
    rms_khz = np.asarray(
        [record["center_residual_rms_khz"] for record in records], dtype=float
    )
    best_index = int(np.nanargmin(rms_khz))

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(kappas, rms_khz, "o-", linewidth=1.2, markersize=4)
    axis.scatter(
        [kappas[best_index]],
        [rms_khz[best_index]],
        marker="*",
        s=170,
        color="tab:red",
        zorder=3,
        label=(
            f"minimum: {kappas[best_index]:.6g} MHz$^{{-1}}$, "
            f"{rms_khz[best_index]:.3f} kHz"
        ),
    )
    axis.axvline(kappas[best_index], color="tab:red", linestyle="--", alpha=0.55)
    if theoretical_kappa_mhz_inv is not None:
        axis.axvline(
            theoretical_kappa_mhz_inv,
            color="tab:green",
            linestyle=":",
            linewidth=2,
            label=(
                r"theory $-1/|\alpha|$: "
                f"{theoretical_kappa_mhz_inv:.6g} MHz$^{{-1}}$"
            ),
        )
    axis.set_xlabel(r"AC-Stark $\kappa$ [MHz$^{-1}$]")
    axis.set_ylabel("RMS resonance-center residual [kHz]")
    axis.set_title("Echo root-Lorentzian center stability")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def _load_saved_dataset(run_directory: Path) -> xr.Dataset:
    with np.load(run_directory / "sweep.npz", allow_pickle=False) as sweep_file:
        coordinates = {name: np.array(sweep_file[name]) for name in sweep_file.files}
    with np.load(run_directory / "results.npz", allow_pickle=False) as results_file:
        data_vars = {
            name: BaseCalibration._array_to_data_var(
                np.array(results_file[name]), coordinates
            )
            for name in results_file.files
        }
    return xr.Dataset(data_vars=data_vars, coords=coordinates)


def embed_response_images_in_html(html_path: Path) -> Path:
    """Embed every response PNG so the explorer is a single shareable file."""
    html_path = Path(html_path)
    html = html_path.read_text(encoding="utf-8")
    if "const responseImages=" in html:
        raise ValueError("The explorer already contains embedded response images.")

    records = _read_csv(html_path.parent / "kappa_center_stability.csv")
    response_images: dict[int, str] = {}
    for record in records:
        run_index = int(record["run_index"])
        image_path = html_path.parent / f"kappa_{run_index:02d}_response.png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response_images[run_index] = f"data:image/png;base64,{encoded}"

    image_javascript = json.dumps(response_images, separators=(",", ":"))
    html = html.replace(
        "const slider=document.getElementById('kappa')",
        f"const responseImages={image_javascript};\n"
        "const slider=document.getElementById('kappa')",
        1,
    )
    html = html.replace(
        "image.src='./kappa_'+String(d.i).padStart(2,'0')+'_response.png';",
        "image.src=responseImages[d.i];",
        1,
    )
    html_path.write_text(html, encoding="utf-8")
    return html_path


if __name__ == "__main__":
    run_stark_kappa_sweep()
