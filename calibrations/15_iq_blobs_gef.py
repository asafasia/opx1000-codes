"""G/E/F IQ calibration using the shared mode-aware IQ-blob implementation."""
import sys
from pathlib import Path
from importlib import import_module

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibration_utils.iq_blobs_ef.parameters import Parameters

IqBlobs = import_module("calibrations.07_iq_blobs").IqBlobs

class IqBlobsGef(IqBlobs):
    def __init__(self, parameters: Parameters, machine=None, **kwargs):
        super().__init__(parameters=parameters, machine=machine, **kwargs)
        self.name = "15_iq_blobs_gef"


if __name__ == "__main__":
    from quam_config import create_machine
    from calibrations import CalibrationOptions
    IqBlobsGef(parameters=Parameters(), machine=create_machine(qubit="q9"),
               options=CalibrationOptions(apply_profile_update=False)).run()
