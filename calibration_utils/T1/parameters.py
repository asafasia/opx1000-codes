from typing import Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, IdleTimeNodeParameters
from calibration_utils.parameters import QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1500
    """Number of averages to perform. Default is 1000."""

    initial_state: Literal["g", "e"] = "e"
    """State prepared before the idle-time sweep: ground (``"g"``) or excited (``"e"``)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
