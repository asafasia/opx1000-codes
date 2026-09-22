from typing import Literal, Optional
from pydantic import Field, model_validator
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.parameters import QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    time_of_flight_in_ns: Optional[int] = 28
    """Time of flight in nanoseconds. Default is 28 ns."""
    readout_amplitude_in_dBm: Optional[float] = 0
    """Readout amplitude in dBm. Default is -12 dBm."""
    readout_length_in_ns: int = Field(default=2000, ge=16, multiple_of=4)
    """Readout length in nanoseconds. Default is 2000 ns."""
    readout_pulse_shape: Literal["flat_top_gaussian", "square"] = "flat_top_gaussian"
    """Gaussian edges by default; use square for time-of-flight calibration."""
    readout_edge_length_in_ns: int = Field(default=100, ge=4, multiple_of=4)
    """Duration of each Gaussian edge; sigma is edge length / 5."""

    @model_validator(mode="after")
    def validate_readout_edges(self):
        if (
            self.readout_pulse_shape == "flat_top_gaussian"
            and 2 * self.readout_edge_length_in_ns >= self.readout_length_in_ns
        ):
            raise ValueError("Gaussian edges must leave a positive flat-top duration.")
        return self



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
