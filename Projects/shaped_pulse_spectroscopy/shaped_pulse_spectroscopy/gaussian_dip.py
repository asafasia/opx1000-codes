"""Bounded echo-dip fitting shared by sweep diagnostics and optimizers."""

import numpy as np
import xarray as xr

SIMPLE_GAUSSIAN_MAXFEV = 800


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
    offsets = np.full(shape, np.nan)
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
            offsets[qubit_index, amp_index] = float(params[0])

    return dataset.assign(
        gaussian_negative_center_hz=(("qubit", "amp_prefactor"), centers),
        gaussian_negative_offset=(("qubit", "amp_prefactor"), offsets),
        gaussian_negative_fwhm_hz=(("qubit", "amp_prefactor"), widths),
        gaussian_negative_fit_r_squared=(("qubit", "amp_prefactor"), scores),
        gaussian_negative_fit_contrast=(("qubit", "amp_prefactor"), contrasts),
        gaussian_negative_center_error_hz=(
            ("qubit", "amp_prefactor"),
            center_errors,
        ),
    )

