"""Check the default DRAG envelope and derivative quadrature without hardware."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from profiles import ProfileError
from profiles.loader import _validate_pulse
from quam_config.populate_quam_lf_mw_fems import _create_pulse


def build(**overrides):
    definition = dict(target="qubit", type="drag", amplitude=0.2, length_ns=40,
                      beta=1.4, detuning_hz=0, axis_angle_rad=0)
    definition.update(overrides)
    _validate_pulse("test", definition)
    return _create_pulse("x180_drag", definition,
                         SimpleNamespace(anharmonicity=237e6), {}, "test", Path("."))


@pytest.mark.parametrize("beta", [0, 0.08, 1.4, -0.5])
def test_default_cosine_and_analytic_derivative(beta):
    pulse = build(beta=beta)
    waveform = pulse.waveform_function()
    t = np.arange(pulse.length)
    phase = 2 * np.pi * t / (pulse.length - 1)
    envelope = 0.5 * pulse.amplitude * (1 - np.cos(phase))
    derivative_per_second = (
        0.5 * pulse.amplitude * 2 * np.pi * 1e9 / (pulse.length - 1) * np.sin(phase)
    )
    np.testing.assert_allclose(waveform.real, envelope, atol=1e-15)
    np.testing.assert_allclose(
        waveform.imag, beta * derivative_per_second / (2 * np.pi * 237e6), atol=1e-15
    )


@pytest.mark.parametrize("width", [{}, {"sigma_ns": 4}, {"sigma_fraction": 1 / 3}])
def test_legacy_width_does_not_change_default_cosine(width):
    np.testing.assert_allclose(build(**width).waveform_function(),
                               build(envelope="cosine").waveform_function())


def test_detuning_and_axis_rotate_both_quadratures():
    detuning = 2e6
    angle = 0.37
    # Adjust beta so the derivative coefficient is held fixed under detuning.
    shifted = build(detuning_hz=detuning, axis_angle_rad=angle,
                    beta=1.4 * (237e6 - detuning) / 237e6)
    expected = build().waveform_function() * np.exp(
        1j * (angle + 2 * np.pi * detuning * np.arange(40) * 1e-9)
    )
    np.testing.assert_allclose(shifted.waveform_function(), expected, atol=1e-15)


@pytest.mark.parametrize("envelope", ["invalid", None, [], {}])
def test_invalid_envelope_rejected(envelope):
    with pytest.raises(ProfileError, match="unsupported envelope"):
        build(envelope=envelope)
