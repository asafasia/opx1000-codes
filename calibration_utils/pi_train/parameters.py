from typing import Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    operation: Literal["x180", "x90"] = "x180"
    """Gate repeated in the train: x180 (pi) or x90 (pi/2)."""
    use_state_discrimination: bool = True
    """Measure excited-state population instead of raw I/Q. Default is True."""
    num_shots: int = 500
    """Number of averages for every pulse-count point."""
    max_number_of_pulses: int = 20
    """Maximum number of consecutive selected gates."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameters for the pi-train experiment."""
