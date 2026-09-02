from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPOSITORY_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = REPOSITORY_ROOT / "Projects" / "shaped_pulse_spectroscopy"
for path in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.drag_beta_kappa_calibration import (
    coarse_plan,
    joint_plan,
    plan_sha256,
    select_best,
)
from shaped_pulse_spectroscopy import lorentzian


def test_beta_zero_is_exactly_backward_compatible() -> None:
    envelope = lorentzian.root_lorentzian_envelope(101, 0.0025, 0.2)
    legacy = lorentzian.apply_echo_phase_jump(envelope)
    parameters = SimpleNamespace(
        pulse_shape="root_lorentzian",
        lorentzian_length_in_ns=101,
        lorentzian_peak_amplitude=0.2,
        cutoff=0.0025,
        echo=True,
        drag_beta=0.0,
        echo_transition_time_ns=16.0,
    )

    np.testing.assert_array_equal(lorentzian.build_waveform(parameters), legacy)
    old_i, old_q = lorentzian.ac_stark_corrected_iq_waveforms(
        legacy,
        amplitude_factor=0.8,
        pi_amplitude=0.1,
        pi_length_ns=40,
        pulse_length_ns=101,
        kappa_mhz_inv=0.00225,
    )
    new_i, new_q = lorentzian.ac_stark_corrected_iq_waveforms(
        legacy,
        amplitude_factor=0.8,
        pi_amplitude=0.1,
        pi_length_ns=40,
        pulse_length_ns=101,
        kappa_mhz_inv=0.00225,
        drag_beta=0.0,
        anharmonicity_mhz=217.0,
    )
    np.testing.assert_array_equal(new_i, old_i)
    np.testing.assert_array_equal(new_q, old_q)


def test_drag_derivative_has_correct_sign_and_physical_time_normalization() -> None:
    signed_i = np.linspace(-0.2, 0.2, 5)
    pulse_length_ns = 4_000
    beta = 0.75
    alpha_mhz = 200.0
    dt_us = pulse_length_ns / len(signed_i) / 1000.0
    expected = -beta * 0.1 / dt_us / (2 * np.pi * alpha_mhz)

    waveform_q = lorentzian.drag_quadrature_waveform(
        signed_i,
        drag_beta=beta,
        anharmonicity_mhz=alpha_mhz,
        pulse_length_ns=pulse_length_ns,
    )

    np.testing.assert_allclose(waveform_q, expected, rtol=1e-14, atol=1e-14)
    assert np.all(np.asarray(waveform_q) < 0)


def test_drag_uses_complete_smooth_signed_echo_and_has_finite_endpoints() -> None:
    envelope = np.ones(200)
    signed_i = lorentzian.apply_echo_phase_jump(
        envelope,
        transition_time_ns=16.0,
        pulse_length_ns=200,
    )
    waveform_q = np.asarray(
        lorentzian.drag_quadrature_waveform(
            signed_i,
            drag_beta=1.0,
            anharmonicity_mhz=200.0,
            pulse_length_ns=200,
        )
    )

    assert np.all(np.isfinite(waveform_q))
    assert np.isfinite(waveform_q[0]) and np.isfinite(waveform_q[-1])
    assert np.max(np.abs(waveform_q[90:110])) > 0
    # A differentiated 1 ns sign jump would produce about 0.8 V here; the
    # 16 ns raised-cosine transition remains far below that artificial spike.
    assert np.max(np.abs(waveform_q)) < 0.2


def test_accumulated_phase_rotates_i_plus_iq_after_drag() -> None:
    signed_i = np.asarray([0.05, 0.10, 0.15, 0.20, 0.15, 0.10])
    kwargs = dict(
        amplitude_factor=0.6,
        pi_amplitude=0.1,
        pi_length_ns=40.0,
        pulse_length_ns=600,
        kappa_mhz_inv=0.00225,
        drag_beta=0.8,
        anharmonicity_mhz=217.0,
    )
    waveform_i, waveform_q = lorentzian.ac_stark_corrected_iq_waveforms(
        signed_i, **kwargs
    )

    drag_q = np.asarray(
        lorentzian.drag_quadrature_waveform(
            signed_i,
            drag_beta=kwargs["drag_beta"],
            anharmonicity_mhz=kwargs["anharmonicity_mhz"],
            pulse_length_ns=kwargs["pulse_length_ns"],
        )
    )
    rabi_mhz = (
        np.asarray(
            lorentzian.amplitude_to_rabi_frequency_hz(
                kwargs["amplitude_factor"] * signed_i,
                kwargs["pi_amplitude"],
                kwargs["pi_length_ns"],
            )
        )
        / 1e6
    )
    phase = np.zeros_like(signed_i)
    phase[1:] = (
        2
        * np.pi
        * kwargs["kappa_mhz_inv"]
        * 0.1
        * np.cumsum(0.5 * (rabi_mhz[:-1] ** 2 + rabi_mhz[1:] ** 2))
    )
    expected = (signed_i + 1j * drag_q) * np.exp(-1j * phase)

    np.testing.assert_allclose(
        np.asarray(waveform_i) + 1j * np.asarray(waveform_q), expected
    )


def test_twenty_microsecond_drag_waveform_has_no_clipping() -> None:
    envelope = lorentzian.root_lorentzian_envelope(20_000, 0.0025, 0.2)
    signed_i = lorentzian.apply_echo_phase_jump(
        envelope,
        transition_time_ns=16.0,
        pulse_length_ns=20_000,
    )
    waveform_q = lorentzian.drag_quadrature_waveform(
        signed_i,
        drag_beta=1.5,
        anharmonicity_mhz=217.106667,
        pulse_length_ns=20_000,
    )
    headroom = lorentzian.complex_waveform_headroom(signed_i, waveform_q)

    assert headroom["max_abs_complex_waveform_v"] < 0.7
    assert headroom["complex_waveform_headroom_v"] > 0


def test_metadata_records_drag_and_stark_conventions() -> None:
    parameters = SimpleNamespace(
        operation="lorentzian",
        pulse_shape="root_lorentzian",
        lorentzian_length_in_ns=20_000,
        waveform_template_length_in_ns=20_000,
        cutoff=0.0025,
        lorentzian_peak_amplitude=0.2,
        echo=True,
        ac_stark_correction=True,
        stark_kappa_mhz_inv=0.00225,
        drag_beta=0.8,
        echo_transition_time_ns=16.0,
    )
    namespace = {
        "lorentzian_waveform_metrics": {
            "q7": {
                "anharmonicity_hz": 217_106_667.0,
                "max_abs_complex_waveform_v": 0.21,
                "complex_waveform_headroom_v": 0.79,
                "project_ceiling_headroom_v": 0.49,
                "max_abs_i_v": 0.2,
                "max_abs_q_v": 0.05,
            }
        }
    }

    metadata = lorentzian._pulse_metadata(parameters, namespace)

    assert metadata["drag_beta"] == 0.8
    assert metadata["stark_kappa_mhz_inv"] == 0.00225
    assert metadata["lorentzian_length_in_ns"] == 20_000
    assert metadata["cutoff"] == 0.0025
    assert metadata["applied_echo_transition_time_ns"] == 16.0
    assert "217106667" in metadata["drag_anharmonicity_hz_by_qubit"]
    assert "physical time" in metadata["drag_derivative_convention"]
    assert metadata["complex_envelope_sign"] == "I+iQ"


def test_two_stage_plan_is_exact_hashable_and_joint_grid_centers_on_best_beta() -> None:
    coarse = coarse_plan(target_qubit="q7", existing_kappa_mhz_inv=0.00225)
    joint = joint_plan(coarse, best_fixed_kappa_beta=0.5)

    assert coarse.waveform.duration_ns == 20_000
    assert coarse.waveform.cutoff == 0.0025
    assert len(coarse.points) == 7
    assert len(plan_sha256(coarse)) == 64
    assert len(joint.points) == 9
    assert {point.drag_beta for point in joint.points} == {0.25, 0.5, 0.75}


def test_selection_minimizes_leakage_only_after_center_and_contrast_gates() -> None:
    records = [
        {
            "label": "low_leak_bad_center",
            "center_rms_hz": 80_000.0,
            "spectroscopy_contrast": 0.2,
            "max_p_f_at_center": 0.001,
        },
        {
            "label": "good_a",
            "center_rms_hz": 30_000.0,
            "spectroscopy_contrast": 0.2,
            "max_p_f_at_center": 0.02,
        },
        {
            "label": "good_b",
            "center_rms_hz": 40_000.0,
            "spectroscopy_contrast": 0.1,
            "max_p_f_at_center": 0.01,
        },
    ]

    selected = select_best(
        records,
        acceptable_center_rms_hz=50_000.0,
        minimum_contrast=0.02,
    )

    assert selected["label"] == "good_b"
