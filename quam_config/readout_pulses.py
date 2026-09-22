"""Readout pulse envelopes shared by profiles and calibration experiments."""

import numpy as np
from quam.core import quam_dataclass
from quam.components.pulses import SquareReadoutPulse
from qualang_tools.config.waveform_tools import flattop_gaussian_waveform


@quam_dataclass
class FlatTopGaussianReadoutPulse(SquareReadoutPulse):
    """Gaussian edges of fixed duration around a flat readout plateau.

    ``length`` includes both edges. Gaussian sigma is ``edge_length_ns / 5``.
    Readout metadata and integration weights use the standard QuAM interface.
    """

    edge_length_ns: int = 100

    @property
    def flat_length(self):
        return self.length - 2 * self.edge_length_ns

    def waveform_function(self):
        if (
            not isinstance(self.edge_length_ns, int)
            or isinstance(self.edge_length_ns, bool)
            or self.edge_length_ns <= 0
            or self.edge_length_ns % 4
            or self.length < 16
            or self.length % 4
            or self.flat_length <= 0
        ):
            raise ValueError("Gaussian edges must be positive multiples of 4 ns and leave a positive flat-top duration.")
        if not np.isfinite(self.amplitude) or abs(self.amplitude) > 0.7:
            raise ValueError("Flat-top Gaussian readout amplitude must be finite and at most 0.7.")
        waveform = np.asarray(flattop_gaussian_waveform(
            amplitude=self.amplitude,
            flat_length=self.flat_length,
            rise_fall_length=self.edge_length_ns,
            sampling_rate=self._get_sampling_rate(),
        ))
        if self.axis_angle is not None:
            waveform = waveform * np.exp(1j * self.axis_angle)
        return waveform


def _readout_fields(pulse):
    fields = pulse.to_dict(follow_references=True)
    if pulse.get_raw_value("integration_weights") == "#./default_integration_weights":
        fields["integration_weights"] = "#./default_integration_weights"
    for key in ("__class__", "flat_length", "edge_length_ns"):
        fields.pop(key, None)
    return fields


def with_gaussian_edges(pulse, edge_length_ns: int):
    """Copy a readout pulse, preserving readout settings and replacing its edges."""
    if edge_length_ns <= 0 or 2 * edge_length_ns >= pulse.length:
        raise ValueError("Gaussian edges must leave a positive flat-top duration.")
    return FlatTopGaussianReadoutPulse(
        **_readout_fields(pulse), edge_length_ns=edge_length_ns
    )


def with_square_envelope(pulse):
    """Copy a readout pulse with a rectangular envelope."""
    return SquareReadoutPulse(**_readout_fields(pulse))
