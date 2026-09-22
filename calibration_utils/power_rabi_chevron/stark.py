"""Gaussian spectral-center fits for the fixed-duration Rabi chevron.

This measures the frequency of the observed maximum, not a direct dressed-state
measurement. Rabi fringes, shaped pulses and power broadening can bias that
maximum. All frequencies, including Rabi frequency, are in cycles/second (Hz).
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from scipy.optimize import OptimizeWarning, curve_fit, least_squares
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d, median_filter


def _noise(y):
    """Robust point-noise estimate, largely insensitive to smooth curvature."""
    triples = np.isfinite(y[:-2]) & np.isfinite(y[1:-1]) & np.isfinite(y[2:])
    differences = np.diff(y, n=2)[triples]
    if differences.size < 3:
        return np.nan
    mad = np.median(np.abs(differences - np.median(differences)))
    return max(float(mad / (0.67448975 * np.sqrt(6))), np.finfo(float).eps)


def _gaussian_model(x, parameters):
    offset, slope, height, center, width = parameters
    return offset + slope * x + height * np.exp(-0.5 * ((x - center) / width) ** 2)


def _gaussian_guesses(x, y, lower, upper, minimum_width):
    """Prominence/FWHM seeds; smoothing is used only for initialization."""
    smooth = gaussian_filter1d(median_filter(y, size=3, mode="nearest"), 1.0)
    peaks, properties = find_peaks(smooth, prominence=0)
    allowed = (x[peaks] > lower) & (x[peaks] < upper)
    peaks, prominences = peaks[allowed], properties["prominences"][allowed]
    strongest = np.argsort(prominences)[-3:][::-1]
    guesses = []
    if strongest.size:
        peaks = peaks[strongest]
        widths = peak_widths(smooth, peaks, rel_height=0.5)
        for index, left, right in zip(peaks, widths[2], widths[3]):
            fwhm = np.interp(right, np.arange(x.size), x) - np.interp(left, np.arange(x.size), x)
            width = np.clip(fwhm / 2.35482, 2 * minimum_width, 0.4)
            for factor in (0.7, 1.4):
                guesses.append((float(x[index]), float(np.clip(width * factor, minimum_width * 1.1, 0.8))))
    # A broad central seed remains available when noise obscures local maxima.
    guesses.append(((lower + upper) / 2, max(2 * minimum_width, 0.15)))
    return guesses, smooth


def _spectral_peak(frequency, signal, fit_points, min_snr, *,
                   center_bounds=None, seed_signal=None, return_diagnostics=False):
    """Bounded multistart robust Gaussian plus linear-background fit.

    Fit unsmoothed samples, using peak prominence/FWHM and optional nearby-trace
    seeds. The covariance is a local approximation from the robust Jacobian,
    inflated by the residual MAD; it is not a bootstrap confidence interval.
    """
    details = {}

    def result(center=np.nan, error=np.nan, status="gaussian_fit_failed"):
        value = (center, error, status)
        return (*value, details) if return_diagnostics else value

    frequency = np.asarray(frequency, dtype=float)
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(signal) & np.isfinite(frequency)
    if finite.sum() < max(fit_points, 8):
        return result(status="nonfinite")
    noise = _noise(signal)
    if not np.isfinite(noise):
        return result(status="nonfinite")
    frequency, signal = frequency[finite], signal[finite]
    if np.any(np.diff(frequency) <= 0):
        return result(status="invalid_frequency_axis")
    step = float(np.median(np.diff(frequency)))
    span = float(np.ptp(frequency))
    origin = float((frequency[0] + frequency[-1]) / 2)
    x = (frequency - origin) / span
    lower, upper = x[0], x[-1]
    if center_bounds is not None:
        lower = max(lower, (center_bounds[0] - origin) / span)
        upper = min(upper, (center_bounds[1] - origin) / span)
    if lower >= upper:
        return result(status="invalid_frequency_axis")
    edge_count = max(3, len(signal) // 8)
    left, right = np.median(signal[:edge_count]), np.median(signal[-edge_count:])
    slope = (right - left) / (np.median(x[-edge_count:]) - np.median(x[:edge_count]))
    offset = left - slope * np.median(x[:edge_count])
    detrended = signal - offset - slope * x
    scale = max(float(np.percentile(detrended, 95)), float(noise))
    if np.ptp(signal) <= 1e-12 or scale <= 1e-12:
        return result(status="weak_or_flat")
    y = detrended / scale
    noise_scaled = max(noise / scale, 1e-6)
    minimum_width = 0.5 * step / span
    guesses, smooth = _gaussian_guesses(x, y, lower, upper, minimum_width)
    if seed_signal is not None:
        seed = np.asarray(seed_signal, dtype=float)[finite]
        if np.all(np.isfinite(seed)):
            seed = (seed - offset - slope * x) / scale
            guesses = _gaussian_guesses(x, seed, lower, upper, minimum_width)[0][:2] + guesses
    # Multiple resolved, comparable lobes cannot identify a unique center.
    peaks, properties = find_peaks(smooth, prominence=max(3 * noise_scaled, 0.2))
    peaks = peaks[(x[peaks] >= lower) & (x[peaks] <= upper)]
    if len(peaks) > 1:
        strongest = peaks[np.argmax(smooth[peaks])]
        widths = peak_widths(smooth, [strongest])[0][0]
        if any(abs(int(p) - strongest) >= max(fit_points, widths)
               and smooth[p] >= 0.85 * smooth[strongest] for p in peaks if p != strongest):
            return result(status="ambiguous")
    best = None
    for center, width in guesses:
        initial = [0.0, 0.0, max(float(np.interp(center, x, smooth)), 0.1),
                   np.clip(center, lower + 1e-9, upper - 1e-9), width]
        try:
            fitted = least_squares(
                lambda p: (_gaussian_model(x, p) - y) / noise_scaled,
                initial, bounds=([-np.inf, -np.inf, 0, lower, minimum_width],
                                 [np.inf, np.inf, np.inf, upper, 1.0]),
                loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=2000,
            )
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        if fitted.success and np.isfinite(fitted.cost) and (best is None or fitted.cost < best[0].cost - 1e-8 * max(1.0, best[0].cost)):
            best = fitted, initial
    if best is None:
        return result()
    fitted, initial = best
    parameters = fitted.x
    _, _, height, center, width = parameters
    residual = (y - _gaussian_model(x, parameters)) / noise_scaled
    residual_scale = max(1.0, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
    # Rank checks prevent pseudoinverse zeros from implying false certainty.
    _, singular, vt = np.linalg.svd(fitted.jac, full_matrices=False)
    if singular[-1] <= singular[0] * 1e-10:
        return result(status="uncertain_peak")
    covariance = (vt.T / singular**2) @ vt * residual_scale**2
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    details.update(
        gaussian_center_hz=float(origin + center * span),
        gaussian_sigma_hz=float(width * span),
        gaussian_height=float(height * scale),
        gaussian_offset=float(offset + parameters[0] * scale),
        gaussian_slope_per_hz=float((slope + parameters[1] * scale) / span),
        gaussian_origin_hz=origin,
        gaussian_center_std_hz=max(float(errors[3] * span), 0.1 * step),
        gaussian_height_snr=float(height / max(errors[2], 1e-15)),
        gaussian_initial_center_hz=float(origin + initial[3] * span),
        gaussian_initial_sigma_hz=float(initial[4] * span),
        gaussian_residual_scale=float(residual_scale),
    )
    if not np.all(np.isfinite(errors)):
        return result(status="uncertain_peak")
    if center - x[0] < max(step / span, width) or x[-1] - center < max(step / span, width):
        return result(status="edge")
    if min(center - lower, upper - center) < 0.1 * step / span:
        return result(status="edge")
    if height <= min_snr * errors[2]:
        return result(status="weak_or_flat")
    if width <= minimum_width * 1.01 or errors[3] > width / 2:
        return result(status="uncertain_peak")
    return result(details["gaussian_center_hz"], details["gaussian_center_std_hz"], "valid")


def _line_fit(x, y, sigma):
    """Weighted y=b+m*x with scaled columns and residual-inflated errors."""
    scale = float(np.ptp(x))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("No amplitude range remains for the fit.")
    origin = float(np.min(x))
    u = (x - origin) / scale
    design = np.column_stack((np.ones_like(u), u))
    weighted = design / sigma[:, None]
    parameters, _, rank, _ = np.linalg.lstsq(weighted, y / sigma, rcond=None)
    if rank < 2:
        raise ValueError("Amplitude fit is singular.")
    prediction = design @ parameters
    chi2 = float(np.sum(((y - prediction) / sigma) ** 2))
    covariance = np.linalg.inv(weighted.T @ weighted)
    covariance *= max(1.0, chi2 / max(1, len(y) - 2))
    transform = np.array([[1.0, -origin / scale], [0.0, 1 / scale]])
    intercept, slope = transform @ parameters
    covariance = transform @ covariance @ transform.T
    weights = 1 / sigma ** 2
    total = float(np.sum(weights * (y - np.average(y, weights=weights)) ** 2))
    r_squared = 1 - chi2 / total if total > 0 else 0.0
    return (
        float(intercept), float(slope),
        float(np.sqrt(covariance[0, 0])), float(np.sqrt(covariance[1, 1])),
        float(r_squared), float(chi2 / max(1, len(y) - 2)),
    )


def _exponent_fit(amplitude, y, sigma, intercept, slope):
    """Test a free power law; do not assume the quadratic hypothesis."""
    if np.unique(np.abs(amplitude)).size < 5:
        return None, None
    maximum = float(np.max(np.abs(amplitude)))
    x = np.abs(amplitude) / maximum

    def model(x, offset, change, exponent):
        return offset + change * x ** exponent

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            fit, covariance = curve_fit(
                model, x, y / 1e6, sigma=sigma / 1e6, absolute_sigma=True,
                p0=[intercept / 1e6, slope * maximum ** 2 / 1e6, 2.0],
                bounds=([-np.inf, -np.inf, 0.2], [np.inf, np.inf, 6.0]),
                maxfev=20000,
            )
        residual_chi2 = float(np.sum(((y / 1e6 - model(x, *fit)) / (sigma / 1e6)) ** 2))
        covariance *= max(1.0, residual_chi2 / max(1, len(y) - 3))
        error = float(np.sqrt(covariance[2, 2]))
        if not np.isfinite(error) or fit[2] < 0.201 or fit[2] > 5.999:
            return None, None
        return float(fit[2]), error
    except (ValueError, RuntimeError, OptimizeWarning, FloatingPointError):
        return None, None


def fit_stark_shift(ds: xr.Dataset, node):
    """Fit each qubit's spectral maximum vs amplitude squared, without updates.

    Returns ``(ds_fit, reports)``. Report success means a usable empirical fit,
    not agreement with either the quadratic law or the ideal transmon prefactor.
    ``theory_shift_hz`` is relative to the free fitted zero-power intercept.
    """
    params = node.parameters
    shape = (ds.sizes["qubit"], ds.sizes["amp_prefactor"])
    dims = ("qubit", "amp_prefactor")
    fit = xr.Dataset(coords={name: ds.coords[name] for name in ds.coords})
    for name in (
        "peak_detuning_hz", "peak_frequency_hz", "peak_frequency_std_hz",
        "quadratic_fit_detuning_hz", "theory_shift_hz",
    ):
        fit[name] = (dims, np.full(shape, np.nan))
        fit[name].attrs["units"] = "Hz"
    fit["peak_valid"] = (dims, np.zeros(shape, dtype=bool))
    fit["peak_fit_used"] = (dims, np.zeros(shape, dtype=bool))
    fit["peak_status"] = (dims, np.full(shape, "not_processed", dtype="U32"))
    fit["fit_signal"] = (
        ("qubit", "detuning", "amp_prefactor"),
        np.full((shape[0], ds.sizes["detuning"], shape[1]), np.nan),
    )
    frequency = np.asarray(ds.detuning, dtype=float)
    order = np.argsort(frequency)
    frequency = frequency[order]
    window = getattr(params, "stark_peak_window_mhz", 50.0)
    fit_points = max(3, int(getattr(params, "stark_peak_fit_points", 5)))
    fit_points += 1 - fit_points % 2
    min_points = max(4, int(getattr(params, "stark_min_valid_points", 6)))
    discrimination = getattr(params, "use_state_discrimination", "state" in ds)
    reports = {}
    for qi, qubit in enumerate(ds.qubit.values):
        qubit_name = str(qubit)
        amplitude = np.asarray(ds.full_amp.isel(qubit=qi), dtype=float)
        prefactor = np.asarray(ds.amp_prefactor, dtype=float)
        rabi = (
            np.asarray(ds.rabi_frequency_hz.isel(qubit=qi), dtype=float)
            if "rabi_frequency_hz" in ds else np.full(shape[1], np.nan)
        )
        anharmonicity = (
            float(ds.signed_anharmonicity_hz.isel(qubit=qi))
            if "signed_anharmonicity_hz" in ds else np.nan
        )
        has_anharmonicity = bool(np.isfinite(anharmonicity) and anharmonicity != 0)
        theory_applicable = (
            bool(ds.stark_theory_applicable.isel(qubit=qi))
            if "stark_theory_applicable" in ds else False
        )
        report = {
            "success": False, "reason": "Insufficient valid Gaussian spectral centers.",
            "peak_method": "Robust multistart Gaussian center with linear background",
            "n_valid_peaks": 0, "n_fit_points": 0,
            "intercept_hz": None, "intercept_std_hz": None,
            "quadratic_coefficient_hz_per_amplitude2": None,
            "quadratic_coefficient_std_hz_per_amplitude2": None,
            "r_squared": None, "quadratic_reduced_chi_squared": None,
            "exponent": None, "exponent_std": None,
            "quadratic_consistency": "inconclusive",
            "signed_anharmonicity_hz": float(anharmonicity) if has_anharmonicity else None,
            "theory_applicable": theory_applicable and has_anharmonicity,
            "theory_coefficient": None, "theory_coefficient_std": None,
            "ratio_to_ideal_half": None,
            "theory_reason": "Ideal +0.5 applies to weak square pulses with negative transmon anharmonicity.",
        }
        reports[qubit_name] = report
        if not has_anharmonicity:
            report["theory_reason"] = "Missing or invalid signed anharmonicity; empirical amplitude fit only."
        elif not theory_applicable:
            report["theory_reason"] = "Ideal +0.5 comparison is not applicable to this pulse (e.g. shaped or DRAG)."
        if discrimination:
            signals = np.asarray(ds.state.isel(qubit=qi).transpose("detuning", "amp_prefactor"), dtype=float)[order]
        else:
            iq = (
                np.asarray(ds.I.isel(qubit=qi).transpose("detuning", "amp_prefactor"), dtype=float)
                + 1j * np.asarray(ds.Q.isel(qubit=qi).transpose("detuning", "amp_prefactor"), dtype=float)
            )[order]
            edge = max(2, len(frequency) // 10)
            outer = np.concatenate((iq[:edge], iq[-edge:]), axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                baseline = np.nanmedian(outer.real, axis=0) + 1j * np.nanmedian(outer.imag, axis=0)
            signals = np.abs(iq - baseline)
        fit.fit_signal.values[qi, order, :] = signals
        absolute_frequency = np.asarray(ds.full_freq.isel(qubit=qi), dtype=float)[order]
        reference = float(np.nanmedian(absolute_frequency - frequency))
        for ai in range(shape[1]):
            if not np.isfinite(amplitude[ai]) or not np.isfinite(prefactor[ai]):
                status = "nonfinite_amplitude"
                peak, error = np.nan, np.nan
            elif abs(amplitude[ai]) <= np.finfo(float).eps:
                status = "zero_amplitude"
                peak, error = np.nan, np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    seed = np.nanmedian(signals[:, max(0, ai - 2):ai + 3], axis=1)
                peak, error, status, diagnostics = _spectral_peak(
                    frequency, signals[:, ai], fit_points,
                    float(getattr(params, "stark_min_peak_snr", 5.0)),
                    center_bounds=None if window is None else (-window * 1e6, window * 1e6),
                    seed_signal=seed, return_diagnostics=True,
                )
                for name, value in diagnostics.items():
                    if name not in fit:
                        fit[name] = (dims, np.full(shape, np.nan))
                    fit[name].values[qi, ai] = value
            fit.peak_status.values[qi, ai] = status
            if status == "valid":
                fit.peak_valid.values[qi, ai] = True
                fit.peak_detuning_hz.values[qi, ai] = peak
                fit.peak_frequency_hz.values[qi, ai] = reference + peak
                fit.peak_frequency_std_hz.values[qi, ai] = error
        used = fit.peak_valid.values[qi].copy()
        minimum = getattr(params, "stark_min_amp_factor", None)
        maximum = getattr(params, "stark_max_amp_factor", None)
        if minimum is not None:
            used &= np.abs(prefactor) >= minimum
        if maximum is not None:
            used &= np.abs(prefactor) <= maximum
        max_ratio = getattr(params, "stark_max_rabi_to_anharmonicity", 0.2)
        if has_anharmonicity and max_ratio is not None:
            # Preserve an empirical fit if physical amplitude calibration is unavailable.
            used &= ~np.isfinite(rabi) | (np.abs(rabi / anharmonicity) <= max_ratio)
        fit.peak_fit_used.values[qi] = used
        report["n_valid_peaks"] = int(fit.peak_valid.values[qi].sum())
        report["n_fit_points"] = int(used.sum())
        if used.sum() < min_points:
            continue
        y = fit.peak_detuning_hz.values[qi, used]
        uncertainty = fit.peak_frequency_std_hz.values[qi, used]
        try:
            intercept, slope, intercept_error, slope_error, r2, reduced_chi2 = _line_fit(amplitude[used] ** 2, y, uncertainty)
        except (ValueError, np.linalg.LinAlgError) as exc:
            report["reason"] = str(exc)
            continue
        exponent, exponent_error = _exponent_fit(amplitude[used], y, uncertainty, intercept, slope)
        report.update(
            success=True, reason="Empirical spectral-maximum fit completed; this does not alone identify an AC Stark shift.",
            intercept_hz=intercept, intercept_std_hz=intercept_error,
            quadratic_coefficient_hz_per_amplitude2=slope,
            quadratic_coefficient_std_hz_per_amplitude2=slope_error,
            r_squared=r2, quadratic_reduced_chi_squared=reduced_chi2,
            exponent=exponent, exponent_std=exponent_error,
        )
        fit.quadratic_fit_detuning_hz.values[qi] = intercept + slope * amplitude ** 2
        resolved = abs(slope) > 3 * slope_error and abs(slope) * np.ptp(amplitude[used] ** 2) > 3 * np.median(uncertainty)
        if resolved and exponent is not None and exponent_error is not None and exponent_error < 0.75:
            report["quadratic_consistency"] = (
                "consistent" if abs(exponent - 2.0) <= 3 * exponent_error and reduced_chi2 <= 3
                else "inconsistent"
            )
        if has_anharmonicity:
            theory_used = used & np.isfinite(rabi)
            if np.any(np.abs(rabi[theory_used] / anharmonicity) > 0.2):
                report["theory_applicable"] = False
                report["theory_reason"] = "Fit includes drive above f_R/|Delta_f|=0.2; weak-drive +0.5 benchmark disabled."
            if theory_used.sum() >= min_points:
                x = rabi[theory_used] ** 2 / abs(anharmonicity)
                try:
                    _, coefficient, _, error, _, _ = _line_fit(
                        x, fit.peak_detuning_hz.values[qi, theory_used],
                        fit.peak_frequency_std_hz.values[qi, theory_used],
                    )
                    report["theory_coefficient"] = coefficient
                    report["theory_coefficient_std"] = error
                    if report["theory_applicable"]:
                        report["ratio_to_ideal_half"] = coefficient / 0.5
                        fit.theory_shift_hz.values[qi] = 0.5 * rabi ** 2 / abs(anharmonicity)
                except (ValueError, np.linalg.LinAlgError):
                    report["theory_reason"] = "Physical Rabi frequencies do not span a usable fit range."
            else:
                report["theory_applicable"] = False
                report["theory_reason"] = "Too few physically calibrated Rabi-frequency points; empirical amplitude fit only."
    fit.attrs["peak_method"] = "Robust multistart Gaussian center with linear background"
    fit.attrs["stark_fit_caveat"] = (
        "A fixed-duration excitation maximum is a proxy for resonance. Rabi side lobes, "
        "pulse shaping and power broadening can invalidate an AC Stark interpretation."
    )
    return fit, reports
