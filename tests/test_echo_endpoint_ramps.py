from types import SimpleNamespace
import numpy as np
import pytest
from Projects.shaped_pulse_spectroscopy.shaped_pulse_spectroscopy import lorentzian

def params(**changes):
    return SimpleNamespace(**(dict(pulse_shape='root_lorentzian',
        lorentzian_length_in_ns=50000, waveform_template_length_in_ns=50000,
        lorentzian_peak_amplitude=1., cutoff=.001, edge_smoothing='none',
        endpoint_ramp_ns=1., echo=True, echo_transition_time_ns=16.) | changes))

def test_direct_one_ns_ramp_changes_only_edge_samples_and_preserves_echo():
    direct = np.asarray(lorentzian.build_waveform(params()))
    untapered = np.asarray(lorentzian.build_waveform(params(endpoint_ramp_ns=0)))
    assert direct[0] == direct[-1] == 0
    np.testing.assert_array_equal(direct[1:-1], untapered[1:-1])
    np.testing.assert_allclose(direct, -direct[::-1], atol=1e-13)
    q = lorentzian.drag_quadrature_waveform(direct, drag_beta=.5,
        anharmonicity_mhz=237., pulse_length_ns=50000)
    assert np.isfinite(q).all()
    assert q[0] != 0  # derivative includes the endpoint ramp

def test_resolved_quintic_ramp_and_physical_time_scaling():
    p = params(lorentzian_length_in_ns=200, waveform_template_length_in_ns=100,
               endpoint_ramp_ns=20, echo=False)
    waveform = np.asarray(lorentzian.build_waveform(p))
    base = np.asarray(lorentzian.build_waveform(params(**(vars(p) | {'endpoint_ramp_ns':0}))))
    x = np.clip(np.minimum(np.linspace(0,200,100),np.linspace(200,0,100))/20,0,1)
    np.testing.assert_allclose(waveform, base*(6*x**5-15*x**4+10*x**3), atol=1e-15)

@pytest.mark.parametrize('ramp', [-1, np.nan, np.inf, 25001])
def test_invalid_endpoint_ramps_rejected(ramp):
    with pytest.raises(ValueError):
        lorentzian.build_waveform(params(endpoint_ramp_ns=ramp))

def test_nonuniform_amplitude_list_overrides_generated_grid():
    requested = [0, .1, .15, .2, .4, .6]
    np.testing.assert_array_equal(lorentzian.amplitude_prefactors(
        SimpleNamespace(amp_factor_values=requested, amp_factor_points=100)), requested)

@pytest.mark.parametrize('values', [[], [0,0], [np.nan], [[0,1]]])
def test_invalid_amplitude_lists(values):
    with pytest.raises(ValueError):
        lorentzian.amplitude_prefactors(SimpleNamespace(amp_factor_values=values))
