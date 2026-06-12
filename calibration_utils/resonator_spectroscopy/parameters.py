from typing import Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 500
    """Number of individual shots to acquire at each frequency."""
    frequency_span_in_mhz: float = 30.
    """Span of frequencies to sweep in MHz. Default is 130 MHz."""
    frequency_step_in_mhz: float = 0.3
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    qubit_operation: Literal["saturation", "x180_const"] = "saturation"
    """Qubit-drive pulse used before or during the driven-state resonator scan."""
    saturation_amplitude_factor: float = 1
    """Amplitude scale applied to the selected qubit operation."""
    saturation_lead_time_in_ns: int = 10_000
    """Lead time before readout when qubit_operation is saturation."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
