import xarray as xr

from qualibrate import QualibrationNode
from utils.experiment_readout import convert_IQ_to_V


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert raw I/Q values to volts when state discrimination is disabled."""
    if not node.parameters.use_state_discrimination:
        return convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds
