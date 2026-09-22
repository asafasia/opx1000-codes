"""Compatibility parameters for the G/E/F IQ calibration entry point."""
from pydantic import Field
from typing import Literal
from calibration_utils.iq_blobs.parameters import Parameters as IqParameters

class Parameters(IqParameters):
    readout_states: list[Literal["g", "e", "f"]] = Field(default_factory=lambda: ["g", "e", "f"])
    operation: Literal["readout", "readout_QND", "readout_GEF"] = "readout_GEF"
