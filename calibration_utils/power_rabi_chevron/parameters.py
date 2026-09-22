from typing import Literal
from pydantic import Field, field_validator

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.parameters import QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Frequency-versus-amplitude Rabi-chevron parameters."""

    num_shots: int = 100
    """Number of averages."""
    operation: Literal["saturation", "x180", "x180_drag", "x180_cosine", "x90", "-x90", "y90", "-y90"] = "x180"
    """Fixed-duration qubit operation whose amplitude is swept."""
    min_amp_factor: float = 0.0
    """Minimum operation-amplitude prefactor."""
    max_amp_factor: float = 2
    """Exclusive maximum operation-amplitude prefactor."""
    amp_factor_step: float = 0.03
    """Operation-amplitude prefactor step."""
    frequency_span_in_mhz: float = 850
    """Total qubit-frequency span in MHz."""
    frequency_step_in_mhz: float = 2
    """Qubit-frequency step in MHz."""


    fit_stark_shift: bool = True
    """Extract spectral peaks and test their amplitude dependence."""
    stark_reference_pi_operation: str = "x180"
    """Calibrated pi operation used to convert amplitude to Rabi frequency."""
    stark_peak_window_mhz: float | None = Field(default=50.0, gt=0)
    """Gaussian-center search half-width; background and width use the full scan. None allows any center."""
    stark_min_peak_snr: float = Field(default=5.0, gt=0)
    """Minimum fitted Gaussian height divided by its standard error."""
    stark_peak_fit_points: int = Field(default=5, ge=3)
    """Minimum finite samples and separation for competing peaks; fit uses the full scan (at least 8 finite samples)."""
    stark_min_amp_factor: float | None = None
    stark_max_amp_factor: float | None = None
    stark_max_rabi_to_anharmonicity: float | None = Field(default=0.2, gt=0)
    """Restrict the fit to weak drive; None disables this cut."""
    stark_min_valid_points: int = Field(default=6, ge=4)

    @field_validator("stark_peak_fit_points")
    @classmethod
    def odd_peak_window(cls, value):
        if value % 2 != 1:
            raise ValueError("stark_peak_fit_points must be odd")
        return value


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
