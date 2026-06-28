import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from utils.rabi_amplitude import amplitude_to_rabi_frequency_hz


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add physical frequency and pulse-amplitude coordinates."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    full_freq = np.array(
        [ds.detuning + qubit.xy.RF_frequency for qubit in node.namespace["qubits"]]
    )
    full_amp = np.array(
        [
            ds.amp_prefactor * qubit.xy.operations[node.parameters.operation].amplitude
            for qubit in node.namespace["qubits"]
        ]
    )
    rabi_frequency_hz = np.array(
        [
            amplitude_to_rabi_frequency_hz(
                full_amp[index],
                float(qubit.xy.operations["x180"].amplitude),
                float(qubit.xy.operations["x180"].length),
            )
            for index, qubit in enumerate(node.namespace["qubits"])
        ]
    )
    ds = ds.assign_coords(
        full_freq=(["qubit", "detuning"], full_freq),
        full_amp=(["qubit", "amp_prefactor"], full_amp),
        rabi_frequency_hz=(["qubit", "amp_prefactor"], rabi_frequency_hz),
    )
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}
    ds.rabi_frequency_hz.attrs = {"long_name": "Rabi frequency", "units": "Hz"}
    return ds
