"""Create standard single-qubit gates from the calibrated x180 pulse."""

from math import pi
from typing import Any

from quam.components.pulses import SquarePulse


DERIVED_GATE_SPECS = {
    "y180": (1.0, pi / 2),
    "x90": (0.5, 0.0),
    "-x90": (-0.5, 0.0),
    "y90": (0.5, pi / 2),
    "-y90": (-0.5, pi / 2),
}


def add_derived_single_qubit_gates(qubit: Any) -> None:
    """Add RB gates using the calibrated x180 amplitude and duration."""
    x180 = qubit.xy.operations["x180"]
    if not isinstance(x180, SquarePulse):
        raise TypeError(
            f"{qubit.name} x180 must be a SquarePulse to derive the standard RB gates"
        )

    for gate_name, (amplitude_factor, axis_angle) in DERIVED_GATE_SPECS.items():
        qubit.xy.operations[gate_name] = SquarePulse(
            length=x180.length,
            amplitude=x180.amplitude * amplitude_factor,
            axis_angle=axis_angle,
        )
