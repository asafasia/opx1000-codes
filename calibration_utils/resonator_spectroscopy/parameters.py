from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 300
    """Number of averages to perform. Defaut is 100."""
    frequency_span_in_mhz: float = 100.
    """Span of frequencies to sweep in MHz. Default is 130 MHz."""
    frequency_step_in_mhz: float = 1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    saturation_amplitude_factor: float = 1
    """Amplitude scale applied to the qubit saturation operation."""
    saturation_lead_time_in_ns: int = 10_000
    """Saturation time before readout; the saturation pulse remains active during readout."""
class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
