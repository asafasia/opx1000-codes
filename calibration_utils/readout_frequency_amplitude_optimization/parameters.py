from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
)
from calibration_utils.parameters import QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of shots per frequency/amplitude point."""
    frequency_span_in_mhz: float = 20
    """Span of readout frequencies to sweep in MHz."""
    frequency_step_in_mhz: float = 1
    """Readout frequency step in MHz."""
    start_amp: float = 0.2
    """Start readout amplitude prefactor."""
    end_amp: float = 2.0
    """End readout amplitude prefactor."""
    num_amps: int = 10
    """Number of readout amplitude prefactors."""
    reset_type: str = "thermal"
    """Qubit reset mode before each prepared state."""
    qubit_operation: str = "x180"
    """Qubit operation used to prepare the excited-state cloud."""
    qubit_amplitude_factor: float = 1.0
    """Amplitude factor applied to the selected qubit operation."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
