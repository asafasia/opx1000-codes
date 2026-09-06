"""Vectorized three-level simulation for fixed-amplitude spectroscopy slices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QutritSliceResult:
    """Final qutrit populations on a Rabi-by-detuning grid."""

    ground: np.ndarray
    excited: np.ndarray
    second_excited: np.ndarray


def _rhs(
    state: np.ndarray,
    *,
    detuning: np.ndarray,
    drive: np.ndarray,
    anharmonicity: float,
    inv_t1: float,
    inv_t_phi: float,
) -> np.ndarray:
    """Three-level Lindblad derivative in the rotating frame."""
    p0, p1, p2, rho01, rho02, rho12 = state
    coupling01 = 0.5 * drive
    coupling12 = drive / np.sqrt(2.0)
    energy1 = -detuning
    energy2 = -2.0 * detuning + anharmonicity

    derivative = np.empty_like(state)
    derivative[0] = 2.0 * np.imag(coupling01 * np.conj(rho01)) + inv_t1 * p1
    derivative[1] = (
        2.0 * np.imag(np.conj(coupling01) * rho01)
        - 2.0 * np.imag(np.conj(coupling12) * rho12)
        - inv_t1 * p1
        + 2.0 * inv_t1 * p2
    )
    derivative[2] = 2.0 * np.imag(np.conj(coupling12) * rho12) - 2.0 * inv_t1 * p2
    derivative[3] = (
        -1j * coupling01 * (p1 - p0)
        + 1j * energy1 * rho01
        + 1j * np.conj(coupling12) * rho02
        + np.sqrt(2.0) * inv_t1 * rho12
        - (0.5 * inv_t1 + inv_t_phi) * rho01
    )
    derivative[4] = (
        -1j * coupling01 * rho12
        + 1j * coupling12 * rho01
        + 1j * energy2 * rho02
        - (inv_t1 + 4.0 * inv_t_phi) * rho02
    )
    derivative[5] = (
        -1j * np.conj(coupling01) * rho02
        - 1j * coupling12 * (p2 - p1)
        + 1j * (energy2 - energy1) * rho12
        - (1.5 * inv_t1 + inv_t_phi) * rho12
    )
    return derivative


def _integrate_half(
    state: np.ndarray,
    *,
    time_start_us: float,
    time_stop_us: float,
    num_steps: int,
    detuning: np.ndarray,
    rabi: np.ndarray,
    anharmonicity: float,
    inv_t1: float,
    inv_t_phi: float,
    sigma_us: float,
    drive_sign: float,
    pulse_shape: str,
    echo: bool,
    drag_beta: float,
    echo_transition_time_us: float,
) -> np.ndarray:
    """Integrate one pulse half with vectorized RK4."""
    step = (time_stop_us - time_start_us) / num_steps

    def derivative(current_state: np.ndarray, time_us: float) -> np.ndarray:
        if pulse_shape == "root_lorentzian":
            envelope = 1.0 / np.sqrt(1.0 + (time_us / sigma_us) ** 2)
            envelope_derivative = (
                -time_us * envelope / (sigma_us**2 + time_us**2)
            )
        elif pulse_shape == "constant":
            envelope = 1.0
            envelope_derivative = 0.0
        else:  # Guarded by simulate_qutrit_slices; keeps this helper defensive.
            raise ValueError(f"Unsupported pulse_shape: {pulse_shape!r}")

        if drag_beta == 0.0:
            # Preserve the established beta=0 simulation exactly.
            scale = drive_sign * envelope
        else:
            if not echo:
                sign = 1.0
                sign_derivative = 0.0
            elif time_us <= -0.5 * echo_transition_time_us:
                sign = 1.0
                sign_derivative = 0.0
            elif time_us >= 0.5 * echo_transition_time_us:
                sign = -1.0
                sign_derivative = 0.0
            else:
                phase = np.pi * time_us / echo_transition_time_us
                sign = -np.sin(phase)
                sign_derivative = -(np.pi / echo_transition_time_us) * np.cos(
                    phase
                )
            signed_i = envelope * sign
            signed_i_derivative = envelope_derivative * sign + envelope * sign_derivative
            drag_q = (
                -drag_beta
                * signed_i_derivative
                / abs(anharmonicity)
            )
            scale = signed_i + 1j * drag_q
        return _rhs(
            current_state,
            detuning=detuning,
            drive=rabi * scale,
            anharmonicity=anharmonicity,
            inv_t1=inv_t1,
            inv_t_phi=inv_t_phi,
        )

    time_us = time_start_us
    for _ in range(num_steps):
        k1 = derivative(state, time_us)
        k2 = derivative(state + 0.5 * step * k1, time_us + 0.5 * step)
        k3 = derivative(state + 0.5 * step * k2, time_us + 0.5 * step)
        k4 = derivative(state + step * k3, time_us + step)
        state += (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time_us += step
    return state


def simulate_qutrit_slices(
    *,
    duration_us: float,
    detuning_mhz: np.ndarray,
    rabi_mhz: np.ndarray,
    t1_us: float,
    t2_star_us: float,
    anharmonicity_mhz: float,
    num_steps_per_half: int,
    cutoff: float,
    echo: bool,
    pulse_shape: str = "root_lorentzian",
    drag_beta: float = 0.0,
    echo_transition_time_ns: float = 0.0,
) -> QutritSliceResult:
    """Simulate root-Lorentzian or constant slices with the qutrit model."""
    if duration_us <= 0:
        raise ValueError("duration_us must be positive.")
    if t1_us <= 0 or t2_star_us <= 0:
        raise ValueError("T1 and T2* must be positive.")
    if t2_star_us >= 2 * t1_us:
        raise ValueError("T2* must be less than 2*T1 for positive pure dephasing.")
    if num_steps_per_half < 1:
        raise ValueError("num_steps_per_half must be positive.")
    if not 0 < cutoff < 1:
        raise ValueError("cutoff must lie strictly between zero and one.")
    if pulse_shape not in {"root_lorentzian", "constant"}:
        raise ValueError("pulse_shape must be 'root_lorentzian' or 'constant'.")
    if anharmonicity_mhz == 0:
        raise ValueError("anharmonicity_mhz must be nonzero.")
    if not np.isfinite(drag_beta):
        raise ValueError("drag_beta must be finite.")
    if not np.isfinite(echo_transition_time_ns) or echo_transition_time_ns < 0:
        raise ValueError("echo_transition_time_ns must be finite and nonnegative.")
    if echo_transition_time_ns >= duration_us * 1e3:
        raise ValueError("echo_transition_time_ns must be shorter than the pulse.")
    if echo and drag_beta != 0.0 and echo_transition_time_ns <= 0:
        raise ValueError(
            "Nonzero DRAG beta on an echo requires a positive transition time."
        )

    detuning, rabi = np.meshgrid(
        2.0 * np.pi * np.asarray(detuning_mhz, dtype=float),
        2.0 * np.pi * np.asarray(rabi_mhz, dtype=float),
    )
    state = np.zeros((6, *detuning.shape), dtype=np.complex128)
    state[0] = 1.0
    half_duration_us = duration_us / 2.0
    sigma_us = half_duration_us / np.sqrt(cutoff**-2 - 1.0)
    inv_t1 = 1.0 / t1_us
    inv_t_phi = 1.0 / t2_star_us - 0.5 * inv_t1
    common = {
        "num_steps": num_steps_per_half,
        "detuning": detuning,
        "rabi": rabi,
        "anharmonicity": 2.0 * np.pi * anharmonicity_mhz,
        "inv_t1": inv_t1,
        "inv_t_phi": inv_t_phi,
        "sigma_us": sigma_us,
        "pulse_shape": pulse_shape,
        "echo": echo,
        "drag_beta": drag_beta,
        "echo_transition_time_us": echo_transition_time_ns / 1e3,
    }
    state = _integrate_half(
        state,
        time_start_us=-half_duration_us,
        time_stop_us=0.0,
        drive_sign=1.0,
        **common,
    )
    state = _integrate_half(
        state,
        time_start_us=0.0,
        time_stop_us=half_duration_us,
        drive_sign=-1.0 if echo else 1.0,
        **common,
    )

    populations = np.real(state[:3])
    trace_error = float(np.max(np.abs(populations.sum(axis=0) - 1.0)))
    if not np.all(np.isfinite(populations)):
        raise RuntimeError("Qutrit simulation produced nonfinite populations.")
    if trace_error > 2e-6:
        raise RuntimeError(f"Qutrit population trace error is {trace_error:.3g}.")
    if float(populations.min()) < -2e-6 or float(populations.max()) > 1.0 + 2e-6:
        raise RuntimeError("Qutrit simulation left the physical probability interval.")
    populations = np.clip(populations, 0.0, 1.0)
    populations /= populations.sum(axis=0, keepdims=True)
    return QutritSliceResult(*populations)
