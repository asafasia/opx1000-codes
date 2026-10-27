from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from pydantic import Field
from qualibration_libs.parameters import (
    CommonNodeParameters,
)
from calibration_utils.parameters import QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 15
    """Span of frequencies to sweep in MHz. Default is 15 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    max_power_dbm: float | None = Field(default=None, allow_inf_nan=False)
    """Maximum RF output power; None uses the resonator's current full-scale power without changing it."""
    min_power_dbm: float = Field(default=-50, allow_inf_nan=False)
    """Minimum power level in dBm. Default is -50 dBm."""
    num_power_points: int = Field(default=100, ge=2)
    """Number of points of the readout power axis. Default is 100."""
    max_amp: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    """Normalized pulse amplitude limit. The default 1.0 reaches the current full-scale power."""
    derivative_crossing_threshold_in_hz_per_dbm: int = -50_000
    """Threshold for derivative crossing in Hz/dBm. Default is -50000 Hz/dBm."""
    derivative_smoothing_window_num_points: int = 10
    """Size of the window in number of points corresponding to the rolling average (number of points). Default is 10."""
    moving_average_filter_window_num_points: int = 10
    """Size of the moving average filter window (number of points). Default is 5."""
    buffer_from_crossing_threshold_in_dbm: int = 1
    """Buffer from the crossing threshold in dBm - the optimal readout power will be set to be this number in Db below
    the threshold. Default is 1 dBm."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
