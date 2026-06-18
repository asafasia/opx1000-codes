from __future__ import annotations

from typing import Any

import numpy as np


def pi_pulse_rabi_frequency_hz(pi_pulse: Any) -> float:
    """Return the Rabi frequency of a square pi pulse."""
    length_ns = float(pi_pulse.length)
    if length_ns <= 0:
        raise ValueError("Square pi pulse length must be positive.")
    return 1 / (2 * length_ns * 1e-9)


def amplitude_to_rabi_frequency_hz(
    general_amp: Any,
    pi_amp: float,
    pi_length_ns: float,
) -> Any:
    """Convert pulse amplitude to Rabi frequency using a square pi calibration."""
    if pi_amp == 0:
        raise ValueError("Square pi pulse amplitude must be non-zero.")
    if pi_length_ns <= 0:
        raise ValueError("Square pi pulse length must be positive.")
    pi_amp_hz = 1 / (2 * pi_length_ns * 1e-9)
    return (general_amp / pi_amp) * pi_amp_hz


def rabi_frequency_hz_to_amplitude(
    rabi_frequency_hz: Any,
    pi_amp: float,
    pi_length_ns: float,
) -> Any:
    """Convert Rabi frequency back to pulse amplitude."""
    if pi_length_ns <= 0:
        raise ValueError("Square pi pulse length must be positive.")
    pi_amp_hz = 1 / (2 * pi_length_ns * 1e-9)
    return np.asarray(rabi_frequency_hz) / pi_amp_hz * pi_amp


def qubit_amplitude_to_rabi_frequency_hz(general_amp: Any, qubit: Any) -> Any:
    """Convert amplitude to Rabi frequency using qubit.xy.operations['x180']."""
    pi_pulse = qubit.xy.operations["x180"]
    return amplitude_to_rabi_frequency_hz(
        general_amp,
        float(pi_pulse.amplitude),
        float(pi_pulse.length),
    )
