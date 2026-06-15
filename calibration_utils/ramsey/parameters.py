from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters

from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages to perform. Default is 100."""
    frequency_detuning_in_mhz: float = 1.0
    """Frequency detuning in MHz. Default is 10.0 MHz."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
