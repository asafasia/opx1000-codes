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
    qubit_amplitude_factor: float = 1
    """Amplitude factor applied to the selected qubit operation."""
    pi_repetitions: int = 1
    """Number of x180_const pulses used to prepare the excited-state blob. Default is 1."""
    xy_to_readout_delay_in_ns: int = 100
    """Delay between the end of the prepared-state XY pulse and readout. Default is 100 ns."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
