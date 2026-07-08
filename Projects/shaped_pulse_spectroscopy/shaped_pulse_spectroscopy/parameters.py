from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Frequency-versus-amplitude sweep for a Lorentzian qubit pulse."""

    num_shots: int = 100
    """Number of averages."""
    operation: str = "lorentzian"
    """Temporary qubit operation name installed by this node."""
    pulse_shape: str = "lorentzian"
    """Pulse shape to install: 'lorentzian', 'root_lorentzian', or 'gaussian'."""
    lorentzian_length_in_ns: int = 40
    """User-chosen Lorentzian pulse length in ns."""
    waveform_template_length_in_ns: int | None = None
    """Optional shorter arbitrary-waveform template length stretched to pulse length."""
    lorentzian_tau_in_ns: float = 8.0
    """Lorentzian half-width parameter tau in ns."""
    lorentzian_peak_amplitude: float = 0.1
    """Peak amplitude A of the unscaled Lorentzian waveform in V."""
    cutoff: float = 0.2
    """Shared edge/peak amplitude ratio for cutoff-based pulse shapes."""
    echo: bool = False
    """Apply a 180-degree phase jump at the waveform midpoint."""
    min_amp_factor: float = 0.0
    """Minimum Lorentzian-amplitude prefactor."""
    max_amp_factor: float = 1.0
    """Exclusive maximum Lorentzian-amplitude prefactor."""
    amp_factor_step: float = 0.03
    """Lorentzian-amplitude prefactor step."""
    fixed_amp_factor: float | None = None
    """Single Lorentzian-amplitude prefactor for fixed-amplitude spectroscopy."""
    fixed_rabi_frequency_mhz: float | None = None
    """Single target Rabi frequency in MHz, converted through the square x180 pulse."""
    frequency_span_in_mhz: float = 850
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
