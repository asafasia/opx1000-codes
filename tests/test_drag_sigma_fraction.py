"""Profile width validation and generated DRAG waveforms, without hardware."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from profiles import ProfileError
from profiles.loader import _validate_pulse
from quam_config.populate_quam_lf_mw_fems import _create_pulse


def pulse_definition(**width):
    return dict(target="qubit", type="drag", envelope="gaussian", amplitude=0.2, length_ns=40,
                beta=0.08, detuning_hz=0, axis_angle_rad=0, **width)


def build(definition):
    return _create_pulse("x180_drag", definition,
                         SimpleNamespace(anharmonicity=237e6), {}, "test", Path("."))


@pytest.mark.parametrize("length", [40, 80])
def test_fraction_matches_absolute_width_waveform_at_each_length(length):
    relative = pulse_definition(sigma_fraction=1 / 3)
    relative["length_ns"] = length
    absolute = pulse_definition(sigma_ns=length / 3)
    absolute["length_ns"] = length
    _validate_pulse("test", relative)
    _validate_pulse("legacy", absolute)
    actual = build(relative)
    assert actual.sigma == pytest.approx(length / 3)
    np.testing.assert_allclose(actual.waveform_function(), build(absolute).waveform_function())


def test_fraction_scales_width_when_length_changes():
    definition = pulse_definition(sigma_fraction=1 / 3)
    short = build(definition)
    longer = deepcopy(definition)
    longer["length_ns"] = 80
    assert build(longer).sigma == pytest.approx(2 * short.sigma)


@pytest.mark.parametrize("field", ["sigma_fraction", "sigma_ns"])
@pytest.mark.parametrize("value", [0, -0.1, True, None, "1/3", float("nan"), float("inf")])
def test_invalid_width_is_rejected(field, value):
    with pytest.raises(ProfileError, match="finite positive"):
        _validate_pulse("test", pulse_definition(**{field: value}))


@pytest.mark.parametrize("width", [{}, {"sigma_fraction": 1 / 3, "sigma_ns": 4}])
def test_missing_or_ambiguous_width_is_rejected(width):
    with pytest.raises(ProfileError, match="exactly one"):
        _validate_pulse("test", pulse_definition(**width))
