import xarray as xr
import numpy as np

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Prepare measured data for plotting and analysis."""
    if node.parameters.use_state_discrimination:
        return ds
    return convert_IQ_to_V(ds, node.namespace["qubits"])


def analyze_fine_rabi(ds: xr.Dataset, node: QualibrationNode) -> tuple[xr.Dataset, dict[str, dict[str, float]]]:
    """Fit the two Fourier-ridge branches and return their intersection."""
    data_name = "state" if node.parameters.use_state_discrimination else "I"
    if data_name not in ds:
        raise RuntimeError(f"Fine-Rabi analysis expected {data_name!r}, but dataset contains {list(ds.data_vars)}")

    data = ds[data_name]
    repetition_counts = data.repetition_group_count.values
    amp_prefactors = data.amp_prefactor.values
    if repetition_counts.size < 4:
        raise ValueError("Fine-Rabi Fourier analysis requires at least four repetition points.")
    spacing = float(np.mean(np.diff(repetition_counts)))
    frequencies = np.fft.rfftfreq(repetition_counts.size, d=spacing)

    centered = data - data.mean(dim="repetition_group_count")
    fourier_amplitude = np.abs(np.fft.rfft(centered.values, axis=data.get_axis_num("repetition_group_count")))
    fourier_amplitude = np.moveaxis(
        fourier_amplitude,
        data.get_axis_num("repetition_group_count"),
        1,
    )

    ridge_frequency = []
    ridge_amplitude = []
    left_mask = []
    right_mask = []
    left_coefficients = []
    right_coefficients = []
    optimum = []
    optimum_frequency = []
    results = {}

    for qubit_index, qubit in enumerate(ds.qubit.values):
        qubit_fft = fourier_amplitude[qubit_index]
        fit_result = fit_fourier_branches(amp_prefactors, frequencies, qubit_fft)
        ridge_frequency.append(fit_result["ridge_frequency"])
        ridge_amplitude.append(fit_result["ridge_amplitude"])
        left_mask.append(fit_result["left_mask"])
        right_mask.append(fit_result["right_mask"])
        left_coefficients.append(fit_result["left_coefficients"])
        right_coefficients.append(fit_result["right_coefficients"])
        optimum.append(fit_result["optimal_amp_prefactor"])
        optimum_frequency.append(fit_result["optimal_frequency"])
        results[str(qubit)] = {
            "optimal_amp_prefactor": float(fit_result["optimal_amp_prefactor"]),
            "optimal_frequency": float(fit_result["optimal_frequency"]),
            "left_slope": float(fit_result["left_coefficients"][0]),
            "left_intercept": float(fit_result["left_coefficients"][1]),
            "right_slope": float(fit_result["right_coefficients"][0]),
            "right_intercept": float(fit_result["right_coefficients"][1]),
        }

    fit = ds.copy()
    fit["fourier_amplitude"] = (
        ("qubit", "fourier_frequency", "amp_prefactor"),
        fourier_amplitude,
    )
    fit = fit.assign_coords(fourier_frequency=frequencies)
    fit["ridge_frequency"] = (
        ("qubit", "amp_prefactor"),
        np.asarray(ridge_frequency),
    )
    fit["ridge_amplitude"] = (
        ("qubit", "amp_prefactor"),
        np.asarray(ridge_amplitude),
    )
    fit["left_branch_mask"] = (
        ("qubit", "amp_prefactor"),
        np.asarray(left_mask, dtype=bool),
    )
    fit["right_branch_mask"] = (
        ("qubit", "amp_prefactor"),
        np.asarray(right_mask, dtype=bool),
    )
    fit["branch_line_coefficients"] = (
        ("qubit", "branch", "line_parameter"),
        np.stack([left_coefficients, right_coefficients], axis=1),
    )
    fit = fit.assign_coords(branch=["left", "right"], line_parameter=["slope", "intercept"])
    fit["optimal_amp_prefactor"] = ("qubit", np.asarray(optimum))
    fit["optimal_frequency"] = ("qubit", np.asarray(optimum_frequency))

    return fit, results


def fit_fourier_branches(
    amp_prefactors: np.ndarray,
    frequencies: np.ndarray,
    fourier_amplitude: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Detect the dominant nonzero Fourier ridge and fit its left/right branches."""
    if fourier_amplitude.shape != (frequencies.size, amp_prefactors.size):
        raise ValueError("fourier_amplitude must have shape (frequency, amplitude).")
    if frequencies.size < 2:
        raise ValueError("At least one nonzero Fourier frequency is required.")

    nonzero_fft = fourier_amplitude[1:, :]
    nonzero_frequencies = frequencies[1:]
    ridge_indices = np.argmax(nonzero_fft, axis=0)
    ridge_frequency = nonzero_frequencies[ridge_indices]
    ridge_amplitude = nonzero_fft[ridge_indices, np.arange(amp_prefactors.size)]

    coarse_optimum_index = int(np.argmin(ridge_frequency))
    left_mask = amp_prefactors < amp_prefactors[coarse_optimum_index]
    right_mask = amp_prefactors > amp_prefactors[coarse_optimum_index]

    left_mask = _keep_strong_branch_points(left_mask, ridge_amplitude)
    right_mask = _keep_strong_branch_points(right_mask, ridge_amplitude)
    if left_mask.sum() < 2 or right_mask.sum() < 2:
        raise ValueError("Could not identify two Fourier ridge branches.")

    left_coefficients = np.polyfit(
        amp_prefactors[left_mask],
        ridge_frequency[left_mask],
        deg=1,
        w=ridge_amplitude[left_mask],
    )
    right_coefficients = np.polyfit(
        amp_prefactors[right_mask],
        ridge_frequency[right_mask],
        deg=1,
        w=ridge_amplitude[right_mask],
    )
    left_slope, left_intercept = left_coefficients
    right_slope, right_intercept = right_coefficients
    if np.isclose(left_slope, right_slope):
        raise ValueError("Fourier ridge branch fits are nearly parallel.")

    optimal_amp_prefactor = (left_intercept - right_intercept) / (right_slope - left_slope)
    optimal_frequency = left_slope * optimal_amp_prefactor + left_intercept

    return {
        "ridge_frequency": ridge_frequency,
        "ridge_amplitude": ridge_amplitude,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "left_coefficients": left_coefficients,
        "right_coefficients": right_coefficients,
        "optimal_amp_prefactor": float(optimal_amp_prefactor),
        "optimal_frequency": float(optimal_frequency),
    }


def _keep_strong_branch_points(mask: np.ndarray, ridge_amplitude: np.ndarray) -> np.ndarray:
    branch_mask = np.asarray(mask, dtype=bool).copy()
    if branch_mask.sum() <= 4:
        return branch_mask
    threshold = np.percentile(ridge_amplitude[branch_mask], 25)
    filtered = branch_mask & (ridge_amplitude >= threshold)
    return filtered if filtered.sum() >= 2 else branch_mask


def log_analysis_results(fit_results: dict[str, dict[str, float]], log_callable) -> None:
    for qubit, result in fit_results.items():
        log_callable(
            f"Fine Rabi {qubit}: optimal amplitude factor "
            f"{result['optimal_amp_prefactor']:.6g}, "
            f"Fourier branch intersection frequency {result['optimal_frequency']:.4g} cycles/group"
        )
