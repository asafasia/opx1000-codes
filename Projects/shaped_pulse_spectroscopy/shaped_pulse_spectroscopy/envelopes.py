"""Pure envelope functions shared by waveform generation and simulation."""

from __future__ import annotations

import numpy as np


def validate_edge_smoothing(mode: str, power: int, pulse_shape: str) -> None:
    if mode not in {"none", "polynomial"}:
        raise ValueError("edge_smoothing must be 'none' or 'polynomial'.")
    if not np.isfinite(power) or power < 1 or int(power) != power:
        raise ValueError("edge_smoothing_power must be a positive integer.")
    if mode == "polynomial" and pulse_shape != "root_lorentzian":
        raise ValueError("Polynomial edge smoothing requires pulse_shape='root_lorentzian'.")


def polynomial_root_envelope(time, *, duration: float, cutoff: float, power: int = 5):
    """Return g and dg/dt for the paper's floor-subtracted polynomial pulse.

    g = (f-c)/(1-c) * [1-(2t/L)^2]^p inside the pulse, zero outside.
    The rationalized expression avoids cancellation near c=1, where the
    continuous limit is [1-(2t/L)^2]^(p+1). Time and duration share units;
    the derivative is per that unit. No area or sampled-peak normalization.
    """
    validate_edge_smoothing("polynomial", power, "root_lorentzian")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive.")
    if not 0 < cutoff <= 1:
        raise ValueError("cutoff must satisfy 0 < cutoff <= 1.")
    t = np.asarray(time, dtype=float)
    half = duration / 2.0
    x = np.clip(t / half, -1.0, 1.0)
    y = np.maximum(0.0, 1.0 - x * x)
    c = cutoff
    root = np.sqrt(c * c + (1.0 - c) * (1.0 + c) * x * x)
    floor = c * (1.0 + c) * y / (root * (1.0 + root))
    floor_derivative = -c * (1.0 + c) * x / (half * root**3)
    window = y**power
    window_derivative = -2.0 * power * x / half * y ** (power - 1)
    inside = np.abs(t) < half
    return (
        np.where(inside, floor * window, 0.0),
        np.where(inside, floor_derivative * window + floor * window_derivative, 0.0),
    )
