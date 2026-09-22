"""Robust local parabolic analysis for external DC-bias spectroscopy."""

from itertools import combinations
import math

import numpy as np
import xarray as xr
from qualibration_libs.analysis import peaks_dips


def robust_parabola(voltage, frequency, frequency_step_hz):
    """Fit a majority-supported quadratic; return result, fitted curve and mask.

    Use normalized voltages and frequency offsets for numerical conditioning.
    Triplet candidates are scored on the best ~60% of residuals, so one false
    peak (including a high-leverage endpoint) cannot initialize the fit alone.
    MAD clipping and a least-squares refit retain the consistent peak track.
    At least five inliers, significant curvature and a bracketed vertex are
    required before the result can be proposed as a new bias setting.
    """
    voltage = np.asarray(voltage, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    if voltage.ndim != 1 or frequency.shape != voltage.shape:
        raise ValueError("Voltage and frequency must be matching 1D arrays.")
    if not np.isfinite(frequency_step_hz) or frequency_step_hz <= 0:
        raise ValueError("A positive frequency resolution is required.")
    valid = np.isfinite(voltage) & np.isfinite(frequency)
    v, y = voltage[valid], frequency[valid]
    if len(v) < 5 or len(np.unique(v)) < 5:
        raise ValueError("At least five distinct bias points with valid peaks are required.")
    center = float((v.min() + v.max()) / 2)
    scale = float(np.ptp(v) / 2)
    x = (v - center) / scale
    baseline = float(np.median(y))
    y = y - baseline
    design = np.column_stack((x*x, x, np.ones_like(x)))
    n = len(x)
    if math.comb(n, 3) <= 1024:
        samples = combinations(range(n), 3)
    else:
        rng = np.random.default_rng(0)
        samples = sorted({tuple(sorted(rng.choice(n, 3, replace=False))) for _ in range(1024)})
    keep = max(4, int(np.ceil(0.6 * n)))
    best_score, coefficients = np.inf, None
    for sample in samples:
        sub = design[list(sample)]
        if np.linalg.cond(sub) > 1e8:
            continue
        candidate = np.linalg.solve(sub, y[list(sample)])
        squared = (y - design @ candidate)**2
        score = float(np.mean(np.partition(squared, keep - 1)[:keep]))
        if score < best_score:
            best_score, coefficients = score, candidate
    if coefficients is None:
        raise ValueError("Bias points do not support a stable quadratic fit.")
    residual = y - design @ coefficients
    mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    # Two frequency bins accommodate quantization of the extracted peaks.
    threshold = max(2 * frequency_step_hz, 3 * mad)
    inlier = np.abs(residual) <= threshold
    minimum = max(5, int(np.ceil(0.6 * n)))
    for _ in range(10):
        if inlier.sum() < minimum:
            raise ValueError("Too few consistent peaks for a robust parabola.")
        coefficients = np.linalg.lstsq(design[inlier], y[inlier], rcond=None)[0]
        updated = np.abs(y - design @ coefficients) <= threshold
        if np.array_equal(updated, inlier):
            break
        inlier = updated
    else:
        raise ValueError("Parabola outlier selection did not converge.")
    a, b, c = coefficients
    residual = y[inlier] - design[inlier] @ coefficients
    rms = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((y[inlier] - np.mean(y[inlier]))**2))
    if total <= 0:
        raise ValueError("Curvature is too weak: peak frequencies have no variation for an R-squared score.")
    r_squared = float(1 - (residual @ residual) / total)
    all_total = float(np.sum((y - np.mean(y))**2))
    r_squared_all = float(1 - np.sum((y - design @ coefficients)**2) / all_total) if all_total > 0 else None
    noise_variance = max(float(residual @ residual / (inlier.sum() - 3)), frequency_step_hz**2 / 12)
    covariance = noise_variance * np.linalg.inv(design[inlier].T @ design[inlier])
    if abs(a) <= 3 * np.sqrt(covariance[0, 0]):
        raise ValueError("Curvature is too weak to locate a reliable extremum.")
    vertex_x = float(-b / (2*a))
    vertex_v = float(center + scale * vertex_x)
    vertex_hz = float(baseline + c - b*b / (4*a))
    gradient = np.array([b / (2*a*a), -1 / (2*a), 0])
    vertex_std = float(scale * np.sqrt(max(0, gradient @ covariance @ gradient)))
    bracketed = bool((v[inlier] < vertex_v).sum() >= 2 and (v[inlier] > vertex_v).sum() >= 2)
    success = bracketed and vertex_std < scale / 2
    reason = "" if success else "Extremum is not bracketed by enough inliers or is too uncertain."
    mask = np.zeros(voltage.shape, dtype=bool)
    mask[valid] = inlier
    xx = (voltage - center) / scale
    curve = baseline + np.polyval(coefficients, xx)
    result = dict(
        success=bool(success), reason=reason, fit_model="robust_parabola",
        idle_offset=vertex_v, extremum_voltage_v=vertex_v,
        frequency_shift=vertex_hz, extremum_type="maximum" if a < 0 else "minimum",
        extremum_voltage_std_v=vertex_std,
        curvature_hz_per_v2=float(a / scale**2),
        fit_center_v=center, fit_scale_v=scale,
        coefficients_hz=[float(a), float(b), float(c + baseline)],
        residual_rms_hz=rms, outlier_threshold_hz=float(threshold),
        r_squared=r_squared, r_squared_all_peaks=r_squared_all,
        r_squared_basis="inlier peak frequencies",
        num_inliers=int(inlier.sum()), num_outliers=int((~inlier).sum()),
        num_missing=int((~valid).sum()),
    )
    return result, curve, mask


def fit_external_flux(ds):
    """Fit I/Q separately; select the valid parabola with highest inlier R².

    Keep both candidates and their peak masks for inspection. R² is evaluated
    after robust rejection so a spurious peak does not dominate selection.
    The all-finite-peaks score and point counts are retained for transparency.
    """
    channels = ("I", "Q")
    positions = {}
    extraction_errors = {}
    for channel in channels:
        try:
            positions[channel] = peaks_dips(ds[channel], dim="detuning", prominence_factor=5).position
        except (ValueError, RuntimeError, IndexError, np.linalg.LinAlgError) as exc:
            extraction_errors[channel] = str(exc)
            positions[channel] = xr.DataArray(
                np.full((ds.sizes["qubit"], ds.sizes["flux_bias"]), np.nan),
                dims=("qubit", "flux_bias"),
                coords={"qubit": ds.qubit, "flux_bias": ds.flux_bias},
            )
    step = float(np.median(np.abs(np.diff(ds.detuning.values))))
    results, selected_peaks, selected_curves, selected_masks = {}, [], [], []
    all_peaks, all_curves, all_masks = [], [], []
    for name in ds.qubit.values:
        candidates, curves, masks, peaks = {}, [], [], []
        for channel in channels:
            peak = positions[channel].sel(qubit=name).values
            try:
                if channel in extraction_errors:
                    raise ValueError(extraction_errors[channel])
                result, curve, mask = robust_parabola(ds.flux_bias.values, peak, step)
                result["qubit_frequency"] = float(ds.drive_frequency_hz.sel(qubit=name)) + result["frequency_shift"]
                if not float(ds.detuning.min()) <= result["frequency_shift"] <= float(ds.detuning.max()):
                    result.update(success=False, reason="Fitted extremum frequency lies outside the measured frequency range.")
            except (ValueError, np.linalg.LinAlgError) as exc:
                result = dict(success=False, reason=str(exc), fit_model="robust_parabola", r_squared=None)
                curve = np.full(ds.sizes["flux_bias"], np.nan)
                mask = np.zeros(ds.sizes["flux_bias"], dtype=bool)
            result["quadrature"] = channel
            candidates[channel] = result
            peaks.append(peak)
            curves.append(curve)
            masks.append(mask)
        eligible = [channel for channel in channels if candidates[channel]["success"]
                    and candidates[channel]["r_squared"] is not None
                    and np.isfinite(candidates[channel]["r_squared"])]
        if eligible:
            chosen = max(eligible, key=lambda channel: candidates[channel]["r_squared"])
            selected = dict(candidates[chosen])
            index = channels.index(chosen)
            selected_peaks.append(peaks[index])
            selected_curves.append(curves[index])
            selected_masks.append(masks[index])
        else:
            chosen = None
            selected = dict(success=False, fit_model="robust_parabola", r_squared=None,
                            reason="; ".join(f"{ch}: {candidates[ch]['reason']}" for ch in channels))
            selected_peaks.append(np.full(ds.sizes["flux_bias"], np.nan))
            selected_curves.append(np.full(ds.sizes["flux_bias"], np.nan))
            selected_masks.append(np.zeros(ds.sizes["flux_bias"], dtype=bool))
        selected["selected_quadrature"] = chosen
        selected["quadrature_fit_results"] = candidates
        results[str(name)] = selected
        all_peaks.append(peaks)
        all_curves.append(curves)
        all_masks.append(masks)
    fits = xr.Dataset(
        {
            "peak_freq": (("qubit", "flux_bias"), np.asarray(selected_peaks)),
            "fit_frequency": (("qubit", "flux_bias"), np.asarray(selected_curves)),
            "fit_inlier": (("qubit", "flux_bias"), np.asarray(selected_masks)),
            "quadrature_peak_freq": (("qubit", "quadrature", "flux_bias"), np.asarray(all_peaks)),
            "quadrature_fit_frequency": (("qubit", "quadrature", "flux_bias"), np.asarray(all_curves)),
            "quadrature_fit_inlier": (("qubit", "quadrature", "flux_bias"), np.asarray(all_masks)),
        },
        coords={"qubit": ds.qubit, "flux_bias": ds.flux_bias, "quadrature": list(channels)},
    )
    for key in ("peak_freq", "fit_frequency", "quadrature_peak_freq", "quadrature_fit_frequency"):
        fits[key].attrs["units"] = "Hz"
    return fits, results
