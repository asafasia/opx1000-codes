"""Shared Gaussian FWHM analysis for measured and simulated spectroscopy."""

from __future__ import annotations

import numpy as np
import xarray as xr

MIN_GAUSSIAN_FWHM_R_SQUARED = 0.1
MAX_GAUSSIAN_CENTER_FRACTION_OF_SPAN = 0.5
MAX_GAUSSIAN_FWHM_FRACTION_OF_SPAN = 0.3
MAX_SUPERPOSITION_COMPONENT_FWHM_FRACTION_OF_SPAN = 0.8
GAUSSIAN_FIT_CANDIDATES = 5
MIN_POLARITY_PROMINENCE_FRACTION = 0.05
GAUSSIAN_INITIAL_FWHM_GUESSES = 24
MIN_FIT_SIGNAL_TO_NOISE = 3.5


def add_gaussian_fwhm_analysis(
    ds: xr.Dataset,
    *,
    use_state_discrimination: bool,
) -> xr.Dataset:
    """Fit a Gaussian spectroscopy trace for each amplitude and store FWHM."""
    if not {"qubit", "detuning", "amp_prefactor"}.issubset(ds.coords):
        return ds

    qubits = list(ds.qubit.values)
    amps = list(ds.amp_prefactor.values)
    shape = (len(qubits), len(amps))
    centers = np.full(shape, np.nan, dtype=float)
    fwhm = np.full(shape, np.nan, dtype=float)
    fit_amplitudes = np.full(shape, np.nan, dtype=float)
    r_squared = np.full(shape, np.nan, dtype=float)
    positive_centers = np.full(shape, np.nan, dtype=float)
    positive_fwhm = np.full(shape, np.nan, dtype=float)
    positive_amplitudes = np.full(shape, np.nan, dtype=float)
    positive_r_squared = np.full(shape, np.nan, dtype=float)
    negative_centers = np.full(shape, np.nan, dtype=float)
    negative_fwhm = np.full(shape, np.nan, dtype=float)
    negative_amplitudes = np.full(shape, np.nan, dtype=float)
    negative_r_squared = np.full(shape, np.nan, dtype=float)
    fit_model = np.full(shape, "none", dtype=object)

    detuning = np.asarray(ds.detuning.values, dtype=float)
    for qubit_index, qubit_name in enumerate(qubits):
        selected_qubit = ds.sel(qubit=qubit_name)
        for amp_index, amp in enumerate(amps):
            trace = selected_qubit.sel(amp_prefactor=amp)
            signal = _spectroscopy_trace_for_fwhm(
                trace,
                use_state_discrimination=use_state_discrimination,
            )
            positive, negative, model = _fit_gaussian_superposition_components(
                detuning,
                signal,
            )
            if model == "none":
                positive = _fit_gaussian_center_fwhm(
                    detuning,
                    signal,
                    polarity="positive",
                )
                negative = _fit_gaussian_center_fwhm(
                    detuning,
                    signal,
                    polarity="negative",
                )
                model = "separate"
            center, width, fit_amplitude, score = _select_min_fwhm_fit(
                positive,
                negative,
            )
            centers[qubit_index, amp_index] = center
            fwhm[qubit_index, amp_index] = width
            fit_amplitudes[qubit_index, amp_index] = fit_amplitude
            r_squared[qubit_index, amp_index] = score
            (
                positive_centers[qubit_index, amp_index],
                positive_fwhm[qubit_index, amp_index],
                positive_amplitudes[qubit_index, amp_index],
                positive_r_squared[qubit_index, amp_index],
            ) = positive
            (
                negative_centers[qubit_index, amp_index],
                negative_fwhm[qubit_index, amp_index],
                negative_amplitudes[qubit_index, amp_index],
                negative_r_squared[qubit_index, amp_index],
            ) = negative
            fit_model[qubit_index, amp_index] = model

    left = centers - fwhm / 2
    right = centers + fwhm / 2
    positive_left = positive_centers - positive_fwhm / 2
    positive_right = positive_centers + positive_fwhm / 2
    negative_left = negative_centers - negative_fwhm / 2
    negative_right = negative_centers + negative_fwhm / 2
    ds = ds.assign(
        gaussian_center_hz=(["qubit", "amp_prefactor"], centers),
        gaussian_fwhm_hz=(["qubit", "amp_prefactor"], fwhm),
        gaussian_fwhm_left_hz=(["qubit", "amp_prefactor"], left),
        gaussian_fwhm_right_hz=(["qubit", "amp_prefactor"], right),
        gaussian_fit_amplitude=(["qubit", "amp_prefactor"], fit_amplitudes),
        gaussian_fit_abs_amplitude=(
            ["qubit", "amp_prefactor"],
            np.abs(fit_amplitudes),
        ),
        gaussian_fit_r_squared=(["qubit", "amp_prefactor"], r_squared),
        gaussian_fit_model=(["qubit", "amp_prefactor"], fit_model),
        gaussian_positive_center_hz=(["qubit", "amp_prefactor"], positive_centers),
        gaussian_positive_fwhm_hz=(["qubit", "amp_prefactor"], positive_fwhm),
        gaussian_positive_fwhm_left_hz=(["qubit", "amp_prefactor"], positive_left),
        gaussian_positive_fwhm_right_hz=(["qubit", "amp_prefactor"], positive_right),
        gaussian_positive_fit_amplitude=(
            ["qubit", "amp_prefactor"],
            positive_amplitudes,
        ),
        gaussian_positive_fit_abs_amplitude=(
            ["qubit", "amp_prefactor"],
            np.abs(positive_amplitudes),
        ),
        gaussian_positive_fit_r_squared=(
            ["qubit", "amp_prefactor"],
            positive_r_squared,
        ),
        gaussian_negative_center_hz=(["qubit", "amp_prefactor"], negative_centers),
        gaussian_negative_fwhm_hz=(["qubit", "amp_prefactor"], negative_fwhm),
        gaussian_negative_fwhm_left_hz=(["qubit", "amp_prefactor"], negative_left),
        gaussian_negative_fwhm_right_hz=(["qubit", "amp_prefactor"], negative_right),
        gaussian_negative_fit_amplitude=(
            ["qubit", "amp_prefactor"],
            negative_amplitudes,
        ),
        gaussian_negative_fit_abs_amplitude=(
            ["qubit", "amp_prefactor"],
            np.abs(negative_amplitudes),
        ),
        gaussian_negative_fit_r_squared=(
            ["qubit", "amp_prefactor"],
            negative_r_squared,
        ),
    )
    ds.gaussian_center_hz.attrs = {
        "long_name": "Gaussian center detuning",
        "units": "Hz",
    }
    ds.gaussian_fwhm_hz.attrs = {
        "long_name": "Gaussian FWHM",
        "units": "Hz",
    }
    ds.gaussian_fwhm_left_hz.attrs = {
        "long_name": "Gaussian FWHM left edge",
        "units": "Hz",
    }
    ds.gaussian_fwhm_right_hz.attrs = {
        "long_name": "Gaussian FWHM right edge",
        "units": "Hz",
    }
    ds.gaussian_fit_amplitude.attrs = {
        "long_name": "Signed Gaussian fit amplitude",
    }
    ds.gaussian_fit_abs_amplitude.attrs = {
        "long_name": "Absolute Gaussian fit amplitude",
    }
    ds.gaussian_fit_r_squared.attrs = {
        "long_name": "Gaussian fit R squared",
    }
    ds.gaussian_fit_model.attrs = {
        "long_name": "Gaussian fit model source",
    }
    _set_polarity_fit_attrs(ds, "positive", "Positive Gaussian")
    _set_polarity_fit_attrs(ds, "negative", "Negative Gaussian")
    if "t2_star_fwhm_limit_hz" in ds.coords:
        ds["gaussian_fwhm_t2_star_units"] = (
            ds.gaussian_fwhm_hz / ds.t2_star_fwhm_limit_hz
        )
        ds.gaussian_fwhm_t2_star_units.attrs = {
            "long_name": "Gaussian FWHM in Ramsey T2* limit units",
        }
    return ds


def _set_polarity_fit_attrs(ds: xr.Dataset, polarity: str, label: str) -> None:
    ds[f"gaussian_{polarity}_center_hz"].attrs = {
        "long_name": f"{label} center detuning",
        "units": "Hz",
    }
    ds[f"gaussian_{polarity}_fwhm_hz"].attrs = {
        "long_name": f"{label} FWHM",
        "units": "Hz",
    }
    ds[f"gaussian_{polarity}_fwhm_left_hz"].attrs = {
        "long_name": f"{label} FWHM left edge",
        "units": "Hz",
    }
    ds[f"gaussian_{polarity}_fwhm_right_hz"].attrs = {
        "long_name": f"{label} FWHM right edge",
        "units": "Hz",
    }
    ds[f"gaussian_{polarity}_fit_amplitude"].attrs = {
        "long_name": f"{label} signed fit amplitude",
    }
    ds[f"gaussian_{polarity}_fit_abs_amplitude"].attrs = {
        "long_name": f"{label} absolute fit amplitude",
    }
    ds[f"gaussian_{polarity}_fit_r_squared"].attrs = {
        "long_name": f"{label} fit R squared",
    }


def _spectroscopy_trace_for_fwhm(
    trace: xr.Dataset,
    *,
    use_state_discrimination: bool,
) -> np.ndarray:
    if use_state_discrimination:
        return np.asarray(trace["state"].values, dtype=float)
    return np.sqrt(
        np.asarray(trace["I"].values, dtype=float) ** 2
        + np.asarray(trace["Q"].values, dtype=float) ** 2
    )


def _fit_gaussian_center_fwhm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    polarity: str | None = None,
) -> tuple[float, float, float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size < 5 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return np.nan, np.nan, np.nan, np.nan

    from scipy.optimize import curve_fit

    baseline = _robust_linear_baseline(x, y)
    residual = y - baseline
    noise_scale = _robust_noise_scale(residual)
    peak_delta = float(np.max(residual))
    dip_delta = float(-np.min(residual))
    if polarity is None:
        positive = _fit_gaussian_center_fwhm(x, y, polarity="positive")
        negative = _fit_gaussian_center_fwhm(x, y, polarity="negative")
        return _select_min_fwhm_fit(positive, negative)

    if polarity not in {"positive", "negative"}:
        raise ValueError("polarity must be 'positive', 'negative', or None.")

    is_peak = polarity == "positive"
    if is_peak and peak_delta <= 0:
        return np.nan, np.nan, np.nan, np.nan
    if not is_peak and dip_delta <= 0:
        return np.nan, np.nan, np.nan, np.nan
    polarity_delta = peak_delta if is_peak else dip_delta
    if polarity_delta < MIN_POLARITY_PROMINENCE_FRACTION * float(np.ptp(y)):
        return np.nan, np.nan, np.nan, np.nan

    fits = []
    for center, amplitude in _gaussian_initial_candidates(
        x,
        residual,
        is_peak=is_peak,
        noise_scale=noise_scale,
    ):
        for sigma in _gaussian_sigma_guesses(x):
            fit_x, fit_y = _gaussian_fit_window(x, y, center, sigma, is_peak=is_peak)
            if fit_x.size < 5 or np.ptp(fit_y) <= 0:
                continue
            fit_baseline = float(np.median(fit_y))
            fit_amplitude = _initial_signed_amplitude(
                fit_x,
                fit_y,
                fit_baseline,
                center,
                is_peak=is_peak,
            )
            if not np.isfinite(fit_amplitude) or fit_amplitude == 0:
                continue
            try:
                fit = _fit_gaussian_attempt(
                    curve_fit,
                    fit_x,
                    fit_y,
                    baseline=fit_baseline,
                    amplitude=fit_amplitude,
                    center=center,
                    sigma=sigma,
                    is_peak=is_peak,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            if _is_valid_gaussian_fit(x, fit) and _has_sufficient_fit_snr(
                fit,
                noise_scale,
            ):
                fits.append(fit)

    if fits:
        return max(fits, key=_gaussian_fit_quality)
    return np.nan, np.nan, np.nan, np.nan


def _fit_gaussian_superposition_components(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    str,
]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    nan_fit = (np.nan, np.nan, np.nan, np.nan)
    if x.size < 8 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return nan_fit, nan_fit, "none"

    from scipy.optimize import curve_fit

    baseline = _robust_linear_baseline(x, y)
    residual = y - baseline
    noise_scale = _robust_noise_scale(residual)
    positive_candidates = _gaussian_initial_candidates(
        x,
        residual,
        is_peak=True,
        noise_scale=noise_scale,
    )
    negative_candidates = _gaussian_initial_candidates(
        x,
        residual,
        is_peak=False,
        noise_scale=noise_scale,
    )
    if not positive_candidates or not negative_candidates:
        return nan_fit, nan_fit, "none"

    sigma_guesses = _combined_gaussian_sigma_guesses(x)
    fits = []
    baseline_offset = float(np.median(y))
    for positive_center, positive_amplitude in positive_candidates:
        for negative_center, negative_amplitude in negative_candidates:
            if abs(positive_center - negative_center) < _minimum_component_spacing(x):
                continue
            for positive_sigma in sigma_guesses:
                for negative_sigma in sigma_guesses:
                    try:
                        params = curve_fit(
                            _gaussian_superposition_with_linear_baseline,
                            x,
                            y,
                            p0=[
                                baseline_offset,
                                0.0,
                                abs(float(positive_amplitude)),
                                float(positive_center),
                                float(positive_sigma),
                                -abs(float(negative_amplitude)),
                                float(negative_center),
                                float(negative_sigma),
                            ],
                            bounds=(
                                [
                                    -np.inf,
                                    -np.inf,
                                    0.0,
                                    float(np.min(x)),
                                    0.0,
                                    -np.inf,
                                    float(np.min(x)),
                                    0.0,
                                ],
                                [
                                    np.inf,
                                    np.inf,
                                    np.inf,
                                    float(np.max(x)),
                                    np.inf,
                                    0.0,
                                    float(np.max(x)),
                                    np.inf,
                                ],
                            ),
                            maxfev=30000,
                        )[0]
                    except (RuntimeError, ValueError, FloatingPointError):
                        continue
                    positive, negative, r_squared = _superposition_fit_components(
                        x,
                        y,
                        params,
                    )
                    if (
                        _is_valid_gaussian_fit(
                            x,
                            positive,
                            max_fwhm_fraction=(
                                MAX_SUPERPOSITION_COMPONENT_FWHM_FRACTION_OF_SPAN
                            ),
                        )
                        and _is_valid_gaussian_fit(
                            x,
                            negative,
                            max_fwhm_fraction=(
                                MAX_SUPERPOSITION_COMPONENT_FWHM_FRACTION_OF_SPAN
                            ),
                        )
                        and _has_sufficient_fit_snr(positive, noise_scale)
                        and _has_sufficient_fit_snr(negative, noise_scale)
                    ):
                        fits.append((positive, negative, r_squared))

    if not fits:
        return nan_fit, nan_fit, "none"

    positive, negative, _ = max(fits, key=_superposition_fit_quality)
    return positive, negative, "superposition"


def _combined_gaussian_sigma_guesses(x: np.ndarray) -> list[float]:
    guesses = _gaussian_sigma_guesses(x)
    if not guesses:
        return []
    indices = np.linspace(0, len(guesses) - 1, min(6, len(guesses)), dtype=int)
    return [float(guesses[index]) for index in np.unique(indices)]


def _minimum_component_spacing(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return max(float(np.median(np.diff(np.sort(x)))) * 2, 0.02 * float(np.ptp(x)))


def _superposition_fit_components(
    x: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    float,
]:
    (
        offset,
        slope,
        positive_amplitude,
        positive_center,
        positive_sigma,
        negative_amplitude,
        negative_center,
        negative_sigma,
    ) = params
    fitted = _gaussian_superposition_with_linear_baseline(x, *params)
    residual_sum_squares = float(np.sum((y - fitted) ** 2))
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = (
        1 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else np.nan
    )
    positive = (
        float(positive_center),
        float(2 * np.sqrt(2 * np.log(2)) * abs(positive_sigma)),
        float(positive_amplitude),
        r_squared,
    )
    negative = (
        float(negative_center),
        float(2 * np.sqrt(2 * np.log(2)) * abs(negative_sigma)),
        float(negative_amplitude),
        r_squared,
    )
    return positive, negative, r_squared


def _superposition_fit_quality(
    fit: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        float,
    ],
) -> tuple[float, float]:
    positive, negative, r_squared = fit
    positive_signal_density = abs(positive[2]) / positive[1]
    negative_signal_density = abs(negative[2]) / negative[1]
    return float(r_squared), float(
        min(positive_signal_density, negative_signal_density)
    )


def _gaussian_initial_candidates(
    x: np.ndarray,
    residual: np.ndarray,
    is_peak: bool,
    noise_scale: float,
) -> list[tuple[float, float]]:
    """Find physically resolved extrema after removing the slow baseline.

    The old implementation seeded fits from any large sample.  That made a
    random off-resonance fluctuation look like a narrow peak.  ``find_peaks``
    gives us a prominence and spacing requirement before nonlinear fitting.
    """
    from scipy.signal import find_peaks

    signed = residual if is_peak else -residual
    prominence = max(
        3 * noise_scale,
        MIN_POLARITY_PROMINENCE_FRACTION * float(np.ptp(residual)),
    )
    indices, properties = find_peaks(
        signed,
        prominence=prominence,
        distance=max(2, x.size // 30),
    )
    order = indices[np.argsort(properties["prominences"])[::-1]]
    candidates: list[tuple[float, float]] = []
    for index in order:
        if signed[index] < prominence:
            continue
        amplitude = float(residual[index])
        candidates.append((int(index), amplitude))
        if len(candidates) >= GAUSSIAN_FIT_CANDIDATES:
            break

    if not candidates:
        return []
    return [(float(x[index]), amplitude) for index, amplitude in candidates]


def _robust_linear_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate the off-resonant baseline from the outer fifth of a sweep."""
    edge_count = max(3, int(np.ceil(x.size * 0.2)))
    edge_indices = np.r_[np.arange(edge_count), np.arange(x.size - edge_count, x.size)]
    slope, offset = np.polyfit(x[edge_indices], y[edge_indices], deg=1)
    return slope * x + offset


def _robust_noise_scale(residual: np.ndarray) -> float:
    """Use adjacent differences so broad spectroscopy features are not noise."""
    differences = np.diff(residual)
    if differences.size == 0:
        return 0.0
    mad = float(np.median(np.abs(differences - np.median(differences))))
    return mad / (0.67448975 * np.sqrt(2))


def _has_sufficient_fit_snr(
    fit: tuple[float, float, float, float],
    noise_scale: float,
) -> bool:
    if noise_scale <= np.finfo(float).eps:
        return True
    return bool(abs(fit[2]) >= MIN_FIT_SIGNAL_TO_NOISE * noise_scale)


def _initial_signed_amplitude(
    x: np.ndarray,
    y: np.ndarray,
    baseline: float,
    center: float,
    *,
    is_peak: bool,
) -> float:
    index = int(np.argmin(np.abs(x - center)))
    if is_peak:
        return float(max(np.max(y) - baseline, y[index] - baseline))
    return float(min(np.min(y) - baseline, y[index] - baseline))


def _gaussian_fit_window(
    x: np.ndarray,
    y: np.ndarray,
    center: float,
    sigma: float,
    *,
    is_peak: bool,
) -> tuple[np.ndarray, np.ndarray]:
    radius = max(4 * float(sigma), 0.08 * float(np.ptp(x)))
    mask = np.abs(x - center) <= radius
    if np.count_nonzero(mask) < 7:
        nearest = int(np.argmin(np.abs(x - center)))
        left = max(0, nearest - 3)
        right = min(x.size, nearest + 4)
        mask = np.zeros_like(x, dtype=bool)
        mask[left:right] = True
    return x[mask], y[mask]


def _gaussian_sigma_guesses(x: np.ndarray) -> list[float]:
    span = float(np.ptp(x))
    if span <= 0:
        return []
    min_fwhm = span / max(20, x.size // 4)
    max_fwhm = MAX_GAUSSIAN_FWHM_FRACTION_OF_SPAN * span
    if min_fwhm >= max_fwhm:
        min_fwhm = max_fwhm / 4
    fwhm_guesses = np.geomspace(
        min_fwhm,
        max_fwhm,
        GAUSSIAN_INITIAL_FWHM_GUESSES,
    )
    return (fwhm_guesses / (2 * np.sqrt(2 * np.log(2)))).tolist()


def _fit_gaussian_attempt(
    curve_fit,
    x: np.ndarray,
    y: np.ndarray,
    *,
    baseline: float,
    amplitude: float,
    center: float,
    sigma: float,
    is_peak: bool,
) -> tuple[float, float, float, float]:
    fit_offset, fit_slope, fit_amplitude, fit_center, fit_sigma = curve_fit(
        _gaussian_with_linear_baseline,
        x,
        y,
        p0=[baseline, 0.0, amplitude, center, sigma],
        bounds=(
            [-np.inf, -np.inf, 0.0 if is_peak else -np.inf, float(np.min(x)), 0.0],
            [np.inf, np.inf, np.inf if is_peak else 0.0, float(np.max(x)), np.inf],
        ),
        maxfev=20000,
    )[0]
    fit_sigma = abs(float(fit_sigma))
    fit_center = float(fit_center)
    fit_amplitude = float(fit_amplitude)
    fitted = _gaussian_with_linear_baseline(
        x,
        fit_offset,
        fit_slope,
        fit_amplitude,
        fit_center,
        fit_sigma,
    )
    residual_sum_squares = float(np.sum((y - fitted) ** 2))
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = (
        1 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else np.nan
    )
    fwhm = float(2 * np.sqrt(2 * np.log(2)) * fit_sigma)
    return fit_center, fwhm, fit_amplitude, r_squared


def _is_valid_gaussian_fit(
    x: np.ndarray,
    fit: tuple[float, float, float, float],
    *,
    max_fwhm_fraction: float = MAX_GAUSSIAN_FWHM_FRACTION_OF_SPAN,
) -> bool:
    center, fwhm, amplitude, r_squared = fit
    max_allowed_center = MAX_GAUSSIAN_CENTER_FRACTION_OF_SPAN * float(np.max(np.abs(x)))
    return bool(
        np.isfinite(center)
        and np.isfinite(fwhm)
        and np.isfinite(amplitude)
        and fwhm > 0
        and np.isfinite(r_squared)
        and r_squared >= MIN_GAUSSIAN_FWHM_R_SQUARED
        and abs(center) <= max_allowed_center
        and fwhm <= max_fwhm_fraction * np.ptp(x)
    )


def _gaussian_fit_quality(fit: tuple[float, float, float, float]) -> tuple[float, float]:
    _, fwhm, amplitude, r_squared = fit
    return float(r_squared), float(abs(amplitude) / fwhm)


def _select_min_fwhm_fit(
    positive: tuple[float, float, float, float],
    negative: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    candidates = [
        fit
        for fit in (positive, negative)
        if np.isfinite(fit[1]) and fit[1] > 0
    ]
    if not candidates:
        return np.nan, np.nan, np.nan, np.nan
    return min(candidates, key=lambda fit: fit[1])


def _gaussian(x, offset, amplitude, center, sigma):
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _gaussian_with_linear_baseline(x, offset, slope, amplitude, center, sigma):
    scale = np.ptp(x)
    baseline_x = (x - np.mean(x)) / scale if scale > 0 else x * 0
    return offset + slope * baseline_x + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _gaussian_superposition_with_linear_baseline(
    x,
    offset,
    slope,
    positive_amplitude,
    positive_center,
    positive_sigma,
    negative_amplitude,
    negative_center,
    negative_sigma,
):
    scale = np.ptp(x)
    baseline_x = (x - np.mean(x)) / scale if scale > 0 else x * 0
    return (
        offset
        + slope * baseline_x
        + positive_amplitude
        * np.exp(-0.5 * ((x - positive_center) / positive_sigma) ** 2)
        + negative_amplitude
        * np.exp(-0.5 * ((x - negative_center) / negative_sigma) ** 2)
    )
