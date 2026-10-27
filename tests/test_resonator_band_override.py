from copy import deepcopy

import pytest

from tests.profile_fixtures import multi_qubit_profile
from profiles import ProfileError, load_profile
from profiles.loader import _select_qubit, validate_profile


@pytest.fixture
def profile():
    profile = deepcopy(load_profile("iqm_chip"))
    connection = profile["connectivity"]["connections"]["q11"]
    connection["resonator_band"] = 2
    connection["lo_frequencies_hz"]["resonator_output"] = 6360800000
    qubit = profile["qubits"]["qubits"]["q11"]
    qubit["frequencies_hz"]["resonator"] = 6160800000
    qubit["readout_gef"]["frequency_hz"] = 6160800000
    return profile


def test_band_override_is_applied_only_to_selected_qubit(profile):
    original = deepcopy(profile)
    validate_profile(profile)
    selected = _select_qubit(profile, "q11")
    fem = selected["connectivity"]["controllers"]["con1"]["fems"]["7"]
    connection = selected["connectivity"]["connections"]["q11"]
    output = fem["outputs"][str(connection["resonator_output"]["port"])]
    input_port = fem["inputs"][str(connection["resonator_input"]["port"])]
    assert output["band"] == 2
    assert input_port["band"] == 2
    assert output["lo_frequency_hz"] == 6360800000
    other = _select_qubit(profile, "q10")
    other_fem = other["connectivity"]["controllers"]["con1"]["fems"]["7"]
    assert other_fem["outputs"]["1"]["band"] == 3
    assert other_fem["inputs"]["1"]["band"] == 3
    assert profile == original


@pytest.mark.parametrize("band", [0, 4, True, 2.0, "2", None, []])
def test_invalid_band_override_is_rejected(profile, band):
    profile["connectivity"]["connections"]["q11"]["resonator_band"] = band
    with pytest.raises(ProfileError, match="unsupported resonator_band"):
        validate_profile(profile)


def test_band_override_still_checks_lo_range(profile):
    profile["connectivity"]["connections"]["q11"]["resonator_band"] = 3
    with pytest.raises(ProfileError, match="outside band 3"):
        validate_profile(profile)


def test_band_override_still_checks_gef_frequency(profile):
    profile["qubits"]["qubits"]["q11"]["readout_gef"]["frequency_hz"] = 7674300000
    with pytest.raises(ProfileError, match="readout_gef.frequency_hz"):
        validate_profile(profile)


def test_multi_qubit_profile_rejects_band_override():
    profile = deepcopy(multi_qubit_profile().load())
    profile["connectivity"]["connections"]["q11"]["resonator_band"] = 2
    with pytest.raises(ProfileError, match="requires a single-qubit profile"):
        validate_profile(profile)
