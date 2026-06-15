from __future__ import annotations

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


def lorentzian_envelope(
    length_ns: int,
    tau_ns: float,
    peak_amplitude: float,
) -> list[float]:
    """Return a centered Lorentzian envelope A / (1 + (t / tau)^2)."""
    if length_ns < 4:
        raise ValueError("lorentzian_length_in_ns must be at least 4 ns.")
    if tau_ns <= 0:
        raise ValueError("lorentzian_tau_in_ns must be positive.")

    times = np.arange(length_ns, dtype=float) - (length_ns - 1) / 2
    envelope = peak_amplitude / (1 + (times / tau_ns) ** 2)
    return envelope.tolist()


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add physical frequency and Lorentzian peak-amplitude coordinates."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    full_freq = np.array(
        [ds.detuning + qubit.xy.RF_frequency for qubit in node.namespace["qubits"]]
    )
    full_amp = np.array(
        [
            ds.amp_prefactor * node.parameters.lorentzian_peak_amplitude
            for _ in node.namespace["qubits"]
        ]
    )
    ds = ds.assign_coords(
        full_freq=(["qubit", "detuning"], full_freq),
        full_amp=(["qubit", "amp_prefactor"], full_amp),
    )
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    ds.full_amp.attrs = {"long_name": "Lorentzian peak amplitude", "units": "V"}
    return ds
