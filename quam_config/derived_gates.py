"""Create standard single-qubit gates from the calibrated x180 pulse."""

from dataclasses import replace
from math import pi
from typing import Any


DERIVED_GATE_SPECS = {
    "y180": (1.0, pi / 2),
    "x90": (0.5, 0.0),
    "-x90": (-0.5, 0.0),
    "y90": (0.5, pi / 2),
    "-y90": (-0.5, pi / 2),
}


def add_derived_single_qubit_gates(qubit: Any) -> None:
    """Add RB gates while preserving the calibrated x180 pulse shape."""
    x180 = qubit.xy.operations["x180"]

    for gate_name, (amplitude_factor, axis_angle) in DERIVED_GATE_SPECS.items():
        qubit.xy.operations[gate_name] = replace(
            x180,
            amplitude=x180.amplitude * amplitude_factor,
            axis_angle=axis_angle,
        )
