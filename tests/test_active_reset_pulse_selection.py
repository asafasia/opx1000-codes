"""Reset must use its profile pulse even when experiments replace x180."""
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from unittest.mock import patch

import pytest
from qm.qua import program

from profiles import ProfileError, load_profile, validate_profile
from profiles.loader import ACTIVE_RESET_OPERATION
from quam_config import create_machine
from quam_config.populate_quam_lf_mw_fems import apply_profile
from utils.readout_macro import active_reset_configured


@pytest.mark.parametrize('selection', ['', None, [], 'missing_pulse', 'readout', 'saturation'])
def test_rejects_invalid_reset_selection(selection):
    profile = load_profile('iqm_chip', qubit='q1')
    profile['manifest']['active_reset_pulse'] = selection
    with pytest.raises(ProfileError, match='active_reset_pulse'):
        validate_profile(profile)


def test_rejects_reserved_reset_operation_collision():
    profile = load_profile('iqm_chip', qubit='q1')
    profile['qubits']['qubits']['q1']['operations'][ACTIVE_RESET_OPERATION] = 'x180_drag'
    with pytest.raises(ProfileError, match='reserved'):
        validate_profile(profile)


@pytest.fixture
def machine():
    return create_machine(profile_name='iqm_chip', qubit='q1')


def test_reset_waveform_is_independent_of_experiment_pulses(machine):
    q = machine.qubits['q1']
    profile = load_profile('iqm_chip', qubit='q1')
    expected = profile['pulses']['pulses']['q1']['x180_const']
    reset = q.xy.operations[q.active_reset_operation]
    assert q.active_reset_operation == ACTIVE_RESET_OPERATION
    assert reset.length == expected['length_ns']
    assert reset.amplitude == expected['amplitude']
    assert reset is not q.xy.operations['x180']
    # Mimic the runtime family changes in RB and the previous one-off scripts.
    q.xy.operations['x180'] = replace(q.xy.operations['x180_drag'])
    q.xy.operations['x180'].amplitude = .2
    config = machine.generate_config()
    pulse_name = config['elements'][q.xy.name]['operations'][ACTIVE_RESET_OPERATION]
    assert config['pulses'][pulse_name]['length'] == expected['length_ns']
    assert reset.amplitude == expected['amplitude']
    assert reset.length == expected['length_ns']


@pytest.mark.parametrize('family', ['drag', 'cosine', 'square'])
def test_rb_family_selection_preserves_reset_pulse(machine, family):
    q = machine.qubits['q1']
    reset = q.xy.operations[ACTIVE_RESET_OPERATION]
    original = (reset.amplitude, reset.length)
    rb = import_module('calibrations.11a_single_qubit_randomized_benchmarking')
    rb.install_rb_gate_family([q], family)
    assert q.xy.operations[ACTIVE_RESET_OPERATION] is reset
    assert (reset.amplitude, reset.length) == original
    assert reset is not q.xy.operations['x180']


@pytest.mark.parametrize('num_states', [2, 3])
def test_shared_macro_plays_selected_reset_for_ge_and_gef(machine, num_states):
    q = machine.qubits['q1']
    q.xy.operations['x180'] = replace(q.xy.operations['x180_drag'])
    q.resonator.readout_gef['gef_centers'] = [[-.001,0],[.001,0],[0,.002]]
    original_play = type(q.xy).play
    with patch.object(type(q.xy), 'play', autospec=True, side_effect=original_play) as play:
        with program():
            active_reset_configured(q, num_states=num_states, max_attempts=3)
    played = [call.args[1] for call in play.call_args_list]
    assert ACTIVE_RESET_OPERATION in played
    assert 'x180' not in played
    assert set(played) == ({ACTIVE_RESET_OPERATION, 'EF_x180'} if num_states == 3
                           else {ACTIVE_RESET_OPERATION})


def test_profiles_without_selector_retain_legacy_reset(machine):
    profile = deepcopy(load_profile('iqm_chip', qubit='q1'))
    profile['manifest'].pop('active_reset_pulse')
    validate_profile(profile)
    apply_profile(machine, profile)
    q = machine.qubits['q1']
    assert q.active_reset_operation == 'x180'
    assert ACTIVE_RESET_OPERATION not in q.xy.operations


def test_diagnostic_override_is_explicit_and_validated(machine):
    q = machine.qubits['q1']
    with pytest.raises(ValueError, match='GE reset operation'):
        active_reset_configured(q, ge_operation='missing')
    original_play = type(q.xy).play
    with patch.object(type(q.xy), 'play', autospec=True, side_effect=original_play) as play:
        with program():
            active_reset_configured(q, ge_operation='x180_drag')
    assert {call.args[1] for call in play.call_args_list} == {'x180_drag'}
