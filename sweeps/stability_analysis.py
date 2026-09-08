"""Descriptive stability diagnostics for repeated scalar measurements.

Allan deviation uses adjacent, overlapping block means (NIST SP 1065).
For T2 estimates this describes the sampled estimator, including acquisition
dead time; it is not fractional oscillator-frequency stability.
"""

from __future__ import annotations

import math
import numpy as np


def point_diagnostics(rows: list[dict], row: dict, window: int, jump_sigma: float) -> dict:
    """Use only preceding, consecutive accepted points for a causal jump flag."""
    history = []
    for previous in reversed(rows[-window:]):
        if not previous["accepted"]:
            break
        history.append(previous)
    history.reverse()
    result = dict(delta_t2_us=None, jump_score=None, jump_candidate=False,
                  rolling_mean_us=None, rolling_std_us=None, rolling_variance_us2=None)
    if not row["accepted"]:
        return result
    values = [r["t2_us"] for r in history] + [row["t2_us"]]
    values = values[-window:]
    result["rolling_mean_us"] = float(np.mean(values))
    if len(values) > 1:
        result["rolling_variance_us2"] = float(np.var(values, ddof=1))
        result["rolling_std_us"] = math.sqrt(result["rolling_variance_us2"])
    if history:
        result["delta_t2_us"] = row["t2_us"] - history[-1]["t2_us"]
    if len(history) >= 5:
        differences = np.diff([r["t2_us"] for r in history])
        center = float(np.median(differences))
        robust_scale = 1.4826 * float(np.median(np.abs(differences - center)))
        fit_scale = math.hypot(row["t2_error_us"], history[-1]["t2_error_us"])
        scale = max(robust_scale, fit_scale, np.finfo(float).eps * abs(row["t2_us"]))
        result["jump_score"] = abs(result["delta_t2_us"] - center) / scale
        result["jump_candidate"] = result["jump_score"] > jump_sigma
    return result


def overlapping_allan(times: np.ndarray, values: np.ndarray, tolerance: float = 0.05) -> dict:
    """Never interpolate gaps or report tau for irregularly spaced samples."""
    result = {"tau_seconds": [], "deviation_us": [], "pair_count": [], "reason": ""}
    if len(values) < 4:
        result["reason"] = "At least four accepted, regularly sampled points are required."
        return result
    if not np.all(np.isfinite(values)):
        result["reason"] = "Rejected/missing points: Allan deviation withheld; no interpolation."
        return result
    steps = np.diff(times)
    spacing = float(np.median(steps))
    if spacing <= 0 or np.any(np.abs(steps - spacing) > tolerance * spacing):
        result["reason"] = "Irregular midpoint spacing (>5%): Allan deviation withheld."
        return result
    cumulative = np.r_[0.0, np.cumsum(values)]
    m = 1
    while len(values) - 2 * m + 1 >= 3:
        means = (cumulative[m:] - cumulative[:-m]) / m
        difference = means[m:] - means[:-m]
        result["tau_seconds"].append(m * spacing)
        result["deviation_us"].append(float(np.sqrt(np.mean(difference**2) / 2)))
        result["pair_count"].append(len(difference))
        m *= 2
    result["reason"] = "Absolute T2-estimate deviation; no detrending or dead-time correction."
    return result


def summarize(rows: list[dict]) -> dict:
    accepted = [r for r in rows if r["accepted"]]
    values = np.asarray([r["t2_us"] for r in accepted], dtype=float)
    times = np.asarray([r["elapsed_seconds"] for r in accepted], dtype=float)
    summary = dict(attempts=len(rows), accepted=len(accepted), rejected=len(rows)-len(accepted),
                   mean_t2_us=None, median_t2_us=None, std_t2_us=None,
                   variance_t2_us2=None, coefficient_of_variation=None,
                   robust_sigma_us=None, peak_to_peak_us=None, drift_us_per_hour=None,
                   jump_candidates=sum(bool(r["jump_candidate"]) for r in rows))
    if values.size:
        summary.update(mean_t2_us=float(np.mean(values)), median_t2_us=float(np.median(values)),
                       robust_sigma_us=float(1.4826*np.median(np.abs(values-np.median(values)))),
                       peak_to_peak_us=float(np.ptp(values)))
    if values.size > 1:
        summary["variance_t2_us2"] = float(np.var(values, ddof=1))
        summary["std_t2_us"] = float(np.std(values, ddof=1))
        summary["coefficient_of_variation"] = summary["std_t2_us"] / summary["mean_t2_us"]
        if np.ptp(times) > 0:
            summary["drift_us_per_hour"] = float(np.polyfit((times-times[0])/3600, values, 1)[0])
    summary["allan"] = overlapping_allan(
        np.asarray([r["elapsed_seconds"] for r in rows], dtype=float),
        np.asarray([r["t2_us"] if r["accepted"] else np.nan for r in rows], dtype=float),
    )
    return summary
