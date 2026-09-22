"""Chevron coordinates, including a waveform-area calibrated Rabi frequency."""

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from utils.experiment_readout import convert_IQ_to_V


def _axis_waveform(pulse):
    """Return samples in the pulse's rotation frame (scalar constants included)."""
    waveform = np.atleast_1d(np.asarray(pulse.calculate_waveform(), dtype=complex))
    return waveform * np.exp(-1j * float(getattr(pulse, "axis_angle", 0.0) or 0.0))


def pulse_rabi_calibration(qubit, operation, reference_operation="x180"):
    """Infer Hz per amplitude from the sampled in-phase area of a calibrated pi.

    Assumes a resonant calibrated reference and linear drive gain at the qubit.
    Using length alone is incorrect for cosine and Gaussian references.
    Unavailable calibration leaves the empirical amplitude fit usable.
    """
    try:
        reference = qubit.xy.operations[reference_operation]
        pulse = qubit.xy.operations[operation]
        if float(getattr(reference, "detuning", 0.0)) != 0:
            raise ValueError("reference pi pulse has nonzero waveform detuning")
        ref_waveform = _axis_waveform(reference)
        waveform = _axis_waveform(pulse)
        area = abs(np.sum(ref_waveform.real) * float(reference.length) * 1e-9 / ref_waveform.size)
        if not np.isfinite(area) or area <= 0 or not np.all(np.isfinite(waveform)):
            raise ValueError("invalid reference area or operation waveform")
        amplitude = abs(float(pulse.amplitude))
        if amplitude == 0:
            raise ValueError("operation amplitude is zero")
        peak_factor = np.max(np.abs(waveform.real)) / amplitude
        gain = peak_factor / (2 * area)
        flat = np.allclose(waveform.real, waveform.real[0], rtol=1e-6, atol=1e-12)
        no_quadrature = np.allclose(waveform.imag, 0, atol=1e-10)
        plain_square = flat and no_quadrature and float(getattr(pulse, "detuning", 0.0)) == 0
        status = "calibrated pi waveform area; assumes linear drive gain"
        if not np.allclose(ref_waveform.imag, 0, atol=1e-10):
            status += "; approximate for a reference with nonzero quadrature"
        return gain, area, bool(plain_square), status
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return np.nan, np.nan, False, f"unavailable: {error}"


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add drive coordinates without changing machine or profile state."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    qubits = list(node.namespace["qubits"])
    operation = node.parameters.operation
    reference = getattr(node.parameters, "stark_reference_pi_operation", "x180")
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in qubits])
    full_amp = np.array([ds.amp_prefactor * q.xy.operations[operation].amplitude for q in qubits])
    calibration = [pulse_rabi_calibration(q, operation, reference) for q in qubits]
    rabi_frequency_hz = np.array([full_amp[i] * values[0] for i, values in enumerate(calibration)])
    # Device profiles store the transmon anharmonicity as a positive magnitude.
    anharmonicity = np.array([-abs(float(getattr(q, "anharmonicity", np.nan))) for q in qubits])
    ds = ds.assign_coords(
        full_freq=(["qubit", "detuning"], full_freq),
        full_amp=(["qubit", "amp_prefactor"], full_amp),
        rabi_frequency_hz=(["qubit", "amp_prefactor"], rabi_frequency_hz),
        signed_anharmonicity_hz=("qubit", anharmonicity),
        reference_pi_area_amplitude_s=("qubit", [c[1] for c in calibration]),
        stark_theory_applicable=("qubit", [c[2] and np.isfinite(a) and a < 0 for c, a in zip(calibration, anharmonicity)]),
        rabi_calibration_status=("qubit", [c[3] for c in calibration]),
    )
    from quam.components.channels import MWChannel

    units = "a.u." if any(isinstance(q.xy, MWChannel) for q in qubits) else "V"
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": units}
    ds.rabi_frequency_hz.attrs = {"long_name": "Rabi frequency", "units": "Hz", "reference_operation": reference, "calibration": "pi pulse sampled in-phase area"}
    ds.signed_anharmonicity_hz.attrs = {"long_name": "configured signed transmon anharmonicity", "units": "Hz", "source": "negative of configured anharmonicity magnitude; calibration must be verified"}
    return ds
