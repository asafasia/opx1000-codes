"""Numerical checks of the paper's edge correction and its wiring."""
from types import SimpleNamespace

import numpy as np
import pytest

from Projects.shaped_pulse_spectroscopy.shaped_pulse_spectroscopy.envelopes import (
    polynomial_root_envelope,
)
from Projects.shaped_pulse_spectroscopy.shaped_pulse_spectroscopy import lorentzian
from Projects.shaped_pulse_spectroscopy.simulation.qutrit_slices import simulate_qutrit_slices


def parameters(**overrides):
    values = dict(pulse_shape="root_lorentzian", lorentzian_length_in_ns=1000,
                  lorentzian_peak_amplitude=0.2, cutoff=0.1, echo=True,
                  echo_transition_time_ns=16.0, drag_beta=0.0,
                  edge_smoothing="polynomial", edge_smoothing_power=5)
    return SimpleNamespace(**(values | overrides))


def test_envelope_matches_paper_equation_and_analytic_derivative():
    t = np.linspace(-14.9, 14.9, 301)
    c, length = 0.001, 30.0
    tau = length / 2 / np.sqrt(c**-2 - 1)
    expected = ((1 / np.sqrt(1 + (t / tau)**2) - c) / (1-c)
                * (1 - (2*t/length)**2)**5)
    g, derivative = polynomial_root_envelope(t, duration=length, cutoff=c)
    np.testing.assert_allclose(g, expected, rtol=1e-11, atol=1e-15)
    h = 1e-7
    forward, _ = polynomial_root_envelope(t+h, duration=length, cutoff=c)
    backward, _ = polynomial_root_envelope(t-h, duration=length, cutoff=c)
    np.testing.assert_allclose(derivative, (forward-backward)/(2*h), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("cutoff", [0.001, 0.2, 1-1e-12, 1.0])
def test_zero_edges_even_envelope_and_center_normalization(cutoff):
    t = np.array([-16., -15., -10., 0., 10., 15., 16.])
    g, dg = polynomial_root_envelope(t, duration=30., cutoff=cutoff)
    assert g[3] == pytest.approx(1.)
    assert dg[3] == 0
    np.testing.assert_allclose(g, g[::-1])
    np.testing.assert_allclose(dg, -dg[::-1])
    np.testing.assert_array_equal(g[[0, 1, 5, 6]], 0.)
    np.testing.assert_array_equal(dg[[0, 1, 5, 6]], 0.)
    if cutoff == 1.0:
        np.testing.assert_allclose(g, np.maximum(0, 1-(t/15)**2)**6)


def test_polynomial_echo_preserves_cancellation_and_drag_is_even():
    p = parameters()
    i = np.asarray(lorentzian.build_waveform(p))
    np.testing.assert_allclose(i, -i[::-1], atol=1e-15)
    assert abs(i.sum()) < 1e-12
    assert i[0] == i[-1] == 0
    assert np.max(abs(i)) < p.lorentzian_peak_amplitude  # no signed-peak rematching
    q = lorentzian.drag_quadrature_waveform(i, drag_beta=-0.5,
            anharmonicity_mhz=216, pulse_length_ns=p.lorentzian_length_in_ns)
    np.testing.assert_allclose(q, q[::-1], atol=1e-14)
    p.drag_beta = -0.5
    np.testing.assert_array_equal(lorentzian.build_waveform(p), i)
    meta = lorentzian._pulse_metadata(p)
    assert meta['edge_smoothing'] == 'polynomial'
    assert meta['edge_smoothing_power'] == 5


def test_disabled_option_and_stretched_template():
    p = parameters(edge_smoothing="none", echo=False)
    np.testing.assert_array_equal(lorentzian.build_waveform(p),
        lorentzian.root_lorentzian_envelope(1000, 0.1, 0.2))
    p = parameters(lorentzian_length_in_ns=2000, waveform_template_length_in_ns=1000)
    i = lorentzian.build_waveform(p)
    assert len(i) == 1000 and i[0] == i[-1] == 0
    p.echo_transition_time_ns = 4
    with pytest.raises(ValueError, match="at least 8 ns"):
        lorentzian.build_waveform(p)


@pytest.mark.parametrize("overrides", [dict(edge_smoothing="bad"),
    dict(edge_smoothing_power=0), dict(edge_smoothing_power=1.5),
    dict(edge_smoothing_power=np.nan), dict(pulse_shape="gaussian")])
def test_invalid_options_fail_before_playback(overrides):
    with pytest.raises(ValueError):
        lorentzian.build_waveform(parameters(**overrides))


def test_polynomial_simulation_is_physical_and_converges_with_and_without_drag():
    common = dict(duration_us=0.1, detuning_mhz=np.array([-1., 0., 1.]),
                  rabi_mhz=np.array([3.]), t1_us=48., t2_star_us=20.,
                  anharmonicity_mhz=-216., cutoff=0.2, echo=True,
                  echo_transition_time_ns=16., edge_smoothing="polynomial")
    for beta in (0., -0.5):
        coarse = simulate_qutrit_slices(**common, drag_beta=beta, num_steps_per_half=200)
        fine = simulate_qutrit_slices(**common, drag_beta=beta, num_steps_per_half=400)
        np.testing.assert_allclose(coarse.excited, fine.excited, atol=1e-7)
        np.testing.assert_allclose(fine.ground+fine.excited+fine.second_excited, 1.)
        plain = simulate_qutrit_slices(**(common | {'edge_smoothing': 'none'}),
                                      drag_beta=beta, num_steps_per_half=400)
        assert np.max(abs(fine.excited - plain.excited)) > 1e-6


def test_plan_passes_edge_options_to_calibration_parameters():
    from Projects.shaped_pulse_spectroscopy.experiments.drag_beta_kappa_calibration import (
        WaveformSpec, beta_only_plan, parameters_for_point,
    )
    plan = beta_only_plan(target_qubit="q1",
                         waveform=WaveformSpec(edge_smoothing="polynomial", edge_smoothing_power=3))
    p = parameters_for_point(plan, plan.points[0])
    assert p.edge_smoothing == "polynomial"
    assert p.edge_smoothing_power == 3


def test_report_passes_smoothing_to_simulator_and_invalidates_cache(tmp_path, monkeypatch):
    from Projects.shaped_pulse_spectroscopy.scripts import simulate_drag_beta_report as report
    calls = []
    def simulate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(ground=np.ones((1, 1)), excited=np.zeros((1, 1)),
                               second_excited=np.zeros((1, 1)))
    monkeypatch.setattr(report, "simulate_qutrit_slices", simulate)
    payload = dict(beta=0., duration_us=1., detuning_mhz=[0.], rabi_mhz=[1.],
                   t1_us=48., t2_star_us=20., anharmonicity_mhz=-216.,
                   num_steps_per_half=1000, cutoff=0.1, transition_time_ns=16.,
                   edge_smoothing="polynomial", edge_smoothing_power=5,
                   output_path=str(tmp_path / "simulation.npz"), force=False)
    report.simulate_one(payload)
    assert calls[0]['echo_transition_time_ns'] == 16.
    assert calls[0]['edge_smoothing'] == 'polynomial'
    report.simulate_one(payload)
    assert len(calls) == 1
    report.simulate_one(payload | {'edge_smoothing_power': 3})
    assert len(calls) == 2
