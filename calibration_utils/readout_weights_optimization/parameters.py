from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages for the sliced readout traces."""
    operation: str = "readout"
    """Readout operation to optimize."""
    division_length_clock_cycles: int = 10
    """Sliced-demod integration chunk in QUA clock cycles. One cycle is 4 ns."""
    use_current_integration_weights: bool = False
    """Use the profile's current readout integration weights. If false, measure with a flat kernel and zero angle."""
    xy_to_readout_delay_in_ns: int = 100
    """Delay between the x180 pulse and the excited-state readout."""
    reset_type: str = "active"
    """Reset mode used before each prepared state."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
