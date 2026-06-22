from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    IdleTimeNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform."""
    n_xy8_values: list[int] = [1, 2, 4, 8]
    """Number of XY8 cycles to sweep. Each cycle contains 8 pi pulses."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
