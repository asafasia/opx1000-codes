from typing import Literal
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 20000
    """Number of runs to perform. Default is 2000."""
    operation: Literal["readout", "readout_QND"] = "readout"
    """Type of operation to perform. Default is "readout"."""
    qubit_operation: Literal["saturation", "x180_const"] = "x180_const"
    """Qubit operation used to prepare the second IQ blob."""
    qubit_amplitude_factor: float = 1.0
    """Amplitude factor applied to the selected qubit operation."""
    pi_repetitions: int = 3
    """Number of x180_const pulses used to prepare the excited-state blob."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
