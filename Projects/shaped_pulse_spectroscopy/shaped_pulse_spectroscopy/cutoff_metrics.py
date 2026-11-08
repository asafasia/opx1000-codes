"""Figure-4-style local dip metrics, using Hz and probability differences.

Background: 0.4 <= |detuning| <= 0.5 MHz. Search: +/-0.25 MHz.
No Gaussian assumption or smoothing; a valid width is a local measurement,
not a guarantee of a unique resonance. No dependency on the paper repository.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def measure_dip(detuning_hz, population, *, min_contrast=0.001, min_intervals=6):
    """Measure depth below a linear background and nearest half-depth width.

Rejected widths are NaN; retain candidate width and contrast for inspection.
Contrast is independent of the shot count and is not a signal-to-noise ratio.
"""
    x, y = np.asarray(detuning_hz, float), np.asarray(population, float)
    if (x.ndim != 1 or x.shape != y.shape or x.size < 9
            or not np.isfinite(x).all() or not np.all(np.diff(x) > 0)
            or x[0] > -500_000 or x[-1] < 500_000):
        raise ValueError("Require increasing matching arrays spanning +/-500 kHz.")
    if not np.isfinite(min_contrast) or min_contrast <= 0 or min_intervals < 1:
        raise ValueError("Require positive contrast threshold and interval count.")
    result = dict(accepted=False, status="invalid_population", center_hz=np.nan,
                  fwhm_hz=np.nan, candidate_fwhm_hz=np.nan, contrast=np.nan,
                  left_half_hz=np.nan, right_half_hz=np.nan,
                  background_at_center=np.nan, background_slope_per_hz=np.nan,
                  shoulder_contrast=np.nan, fwhm_grid_intervals=np.nan,
                  prominent_minima=0)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        return result
    inside = (x > -500_000) & (x < 500_000)
    xx = np.r_[-500_000., x[inside], 500_000.]
    yy = np.interp(xx, x, y)
    edges = np.abs(xx) >= 400_000
    central = np.flatnonzero(np.abs(xx) <= 250_000)
    if np.sum(edges & (xx < 0)) < 2 or np.sum(edges & (xx > 0)) < 2 or central.size < 3:
        raise ValueError("Need at least two background samples per side and three central samples.")
    slope, intercept = np.polyfit(xx[edges] / 1e6, yy[edges], 1)
    depth = slope * xx / 1e6 + intercept - yy
    peak = central[np.argmax(depth[central])]
    contrast = float(depth[peak])
    result.update(center_hz=float(xx[peak]), contrast=contrast,
                  background_at_center=float(yy[peak] + contrast),
                  background_slope_per_hz=float(slope / 1e6),
                  shoulder_contrast=float(np.mean(np.interp([-450_000, 450_000], xx, yy)) - yy[peak]),
                  prominent_minima=int(len(find_peaks(-yy, prominence=0.005)[0])))
    if contrast < min_contrast:
        result["status"] = "low_contrast"
        return result
    if peak in (central[0], central[-1]):
        result["status"] = "center_at_search_boundary"
        return result
    half = contrast / 2
    left = np.flatnonzero(depth[:peak] <= half)
    right = np.flatnonzero(depth[peak + 1:] <= half) + peak + 1
    if not left.size or not right.size:
        result["status"] = "missing_half_depth_crossing"
        return result
    li, ri = left[-1], right[0]
    xl = xx[li] + (half-depth[li])/(depth[li+1]-depth[li]) * (xx[li+1]-xx[li])
    xr = xx[ri-1] + (half-depth[ri-1])/(depth[ri]-depth[ri-1]) * (xx[ri]-xx[ri-1])
    width = float(xr-xl)
    intervals = width / np.max(np.diff(xx))
    result.update(candidate_fwhm_hz=width, left_half_hz=float(xl), right_half_hz=float(xr),
                  fwhm_grid_intervals=float(intervals))
    if result["shoulder_contrast"] <= 0:
        result["status"] = "nonpositive_shoulder_contrast"
    elif intervals < min_intervals:
        result["status"] = "undersampled_dip"
    else:
        result.update(accepted=True, status="accepted", fwhm_hz=width)
    return result
