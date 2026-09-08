"""Parameters for spectroscopy with a host-controlled external DC source."""

from typing import Literal

from pydantic import Field

from .parameters import Parameters as FluxParameters


class Parameters(FluxParameters):
    num_shots: int = Field(default=50, gt=0)
    frequency_span_in_mhz: float = Field(default=100.0, gt=0, allow_inf_nan=False)
    frequency_step_in_mhz: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    flux_offset_span_in_v: float = Field(default=0.002, gt=0, allow_inf_nan=False)
    num_flux_points: int = Field(default=11, ge=2)
    flux_bias_center_in_v: float | None = Field(default=None, allow_inf_nan=False)
    """Absolute source voltage at sweep center; None uses the qubit's dc_bias_v."""
    bias_settle_time_s: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    pause_timeout_s: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    """Maximum wait for each paused sweep or its result streams."""
    reset_type: Literal["thermal"] = "thermal"
    use_state_discrimination: Literal[False] = False
    input_line_impedance_in_ohm: int = Field(default=50, gt=0)
    line_attenuation_in_db: int = 0
