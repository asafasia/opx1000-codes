from typing import Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Frequency-versus-amplitude Rabi-chevron parameters."""

    num_shots: int = 30
    """Number of averages."""
    operation: Literal["x180", "x180_drag", "x180_cosine", "x90", "-x90", "y90", "-y90"] = "x180"
    """Fixed-duration qubit operation whose amplitude is swept."""
    min_amp_factor: float = 0.0
    """Minimum operation-amplitude prefactor."""
    max_amp_factor: float = 1.5
    """Exclusive maximum operation-amplitude prefactor."""
    amp_factor_step: float = 0.03
    """Operation-amplitude prefactor step."""
    frequency_span_in_mhz: float = 300
    """Total qubit-frequency span in MHz."""
    frequency_step_in_mhz: float = 2
    """Qubit-frequency step in MHz."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
