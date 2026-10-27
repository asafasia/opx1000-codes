from types import SimpleNamespace
import numpy as np
import pytest
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms

from Projects.shaped_pulse_spectroscopy.scripts import compare_echo_inversion_times as compare
from shaped_pulse_spectroscopy.lorentzian import drag_quadrature_waveform, _pulse_metadata
from utils.rabi_amplitude import pi_pulse_effective_length_ns, amplitude_to_rabi_frequency_hz


def test_echo_positive_beta_matches_library_gaussian_quadrature():
    i,q=drag_gaussian_pulse_waveforms(amplitude=.2,length=100,sigma=10,alpha=.5,
        anharmonicity=200e6,detuning=0)
    actual=np.array(drag_quadrature_waveform(i,drag_beta=.5,anharmonicity_mhz=200,pulse_length_ns=100))
    # Finite difference versus analytic derivative; same sign, scale and time units.
    assert np.linalg.norm(actual-np.array(q))/np.linalg.norm(q)<.004
    assert np.dot(actual,q)>0


def test_cosine_and_square_pi_references_agree_without_changing_voltage():
    cosine=np.array(.3095272550117577*(1-np.cos(2*np.pi*np.arange(40)/39))/2)
    p=SimpleNamespace(length=40,amplitude=.3095272550117577,waveform_function=lambda:cosine)
    assert pi_pulse_effective_length_ns(p)==pytest.approx(19.5)
    physical=amplitude_to_rabi_frequency_hz(.7,p.amplitude,pi_pulse_effective_length_ns(p))
    square=amplitude_to_rabi_frequency_hz(.7,.15076984441776575,40)
    assert physical==pytest.approx(square,rel=.001)
    assert p.amplitude==.3095272550117577


def test_square_and_rotated_reference_areas():
    p=SimpleNamespace(length=40,amplitude=.15,waveform_function=lambda:.15j)
    assert pi_pulse_effective_length_ns(p)==pytest.approx(40)
    p.waveform_function=lambda:np.zeros(40)
    with pytest.raises(ValueError,match='nonzero'):pi_pulse_effective_length_ns(p)


@pytest.mark.parametrize('transition',[4,8,16,50])
def test_comparison_changes_only_transition_and_uses_requested_beta(transition):
    p=compare.parameters(compare.opt.Settings(beta=.5),transition)
    assert p.drag_beta==.5 and p.echo_transition_time_ns==transition
    assert not p.ac_stark_correction and p.stark_kappa_mhz_inv==0
    assert p.lorentzian_length_in_ns==p.waveform_template_length_in_ns==20000
    assert p.lorentzian_peak_amplitude==.7 and p.cutoff==.001
    assert (p.amp_factor_points,p.frequency_points,p.num_shots)==(25,121,400)
    assert _pulse_metadata(p)['drag_beta_convention']==compare.CONVENTION
