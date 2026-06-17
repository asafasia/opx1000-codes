from typing import Literal, Protocol, runtime_checkable

import numpy as np
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


Transition = Literal["ge", "ef"]
GeOperation = Literal["x180", "x180_drag", "x180_cosine", "x90", "-x90", "y90", "-y90"]


class BasePowerRabiParameters(RunnableParameters):
    """Parameters shared by GE and EF power Rabi."""

    transition: Transition = "ge"
    """Transition to calibrate. GE uses the selected operation; EF always uses EF_x180."""
    num_shots: int = 300
    """Number of averages to perform. Default is 300    ."""
    min_amp_factor: float = 0.0
    """Minimum amplitude factor for the operation. Default is 0."""
    max_amp_factor: float = 1
    """Maximum amplitude factor for the operation. Default is 1.99."""
    amp_factor_step: float = 0.005
    """Step size for the amplitude factor. Default is 0.005."""


class NodeSpecificParameters(BasePowerRabiParameters):
    """Power Rabi parameters. Error amplification applies only to the GE transition."""

    operation: GeOperation = "x180"
    """GE operation to calibrate. Ignored when transition="ef"."""
    pi_repetitions: int = 3
    """Number of times to repeat each pulse-count point. Default is 3."""
    max_number_pulses_per_sweep: int = 1
    """Maximum number of Rabi pulses per sweep (error amplification). Default is 1."""
    update_x90: bool = True
    """Flag to update the x90 pulse amplitude after calibrating x180. Default is True."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for combined GE/EF power Rabi."""


class EfParameters(
    Parameters,
):
    """Backward-compatible EF parameter class."""

    transition: Transition = "ef"


@runtime_checkable
class HasErrorAmplification(Protocol):
    """Structural typing for objects supporting error amplification controls."""

    max_number_pulses_per_sweep: int
    pi_repetitions: int
    operation: str


def get_number_of_pulses(node_parameter: BasePowerRabiParameters):
    """Return array of number of pulses for error amplification.

    For EF, pi_repetitions controls how many EF_x180 pulses are applied after
    preparing |e>.
    """
    if not isinstance(node_parameter, HasErrorAmplification):
        return np.array([1], dtype=int)

    if node_parameter.pi_repetitions < 1:
        raise ValueError("pi_repetitions must be a positive integer.")

    if getattr(node_parameter, "transition", "ge") == "ef":
        return np.array([node_parameter.pi_repetitions], dtype=int)

    if node_parameter.max_number_pulses_per_sweep > 1:
        if node_parameter.operation.endswith("x180") or node_parameter.operation.startswith("x180_"):
            N_pulses = np.arange(1, node_parameter.max_number_pulses_per_sweep, 2).astype(int)
        elif node_parameter.operation in ["x90", "-x90", "y90", "-y90"]:
            N_pulses = np.arange(2, node_parameter.max_number_pulses_per_sweep, 4).astype(int)
        else:
            raise ValueError(f"Unrecognized operation {node_parameter.operation}.")
    else:
        N_pulses = np.linspace(
            1,
            node_parameter.max_number_pulses_per_sweep,
            node_parameter.max_number_pulses_per_sweep,
        ).astype(int)[::2]
    return N_pulses * node_parameter.pi_repetitions
