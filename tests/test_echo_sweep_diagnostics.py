from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from Projects.shaped_pulse_spectroscopy.experiments import detuning_amplitude_sweep as experiment
from shaped_pulse_spectroscopy.sweep_diagnostics import add_sweep_diagnostics, plot_sweep_diagnostics
from calibration_io.calibration_saver import CalibrationSaver
from shaped_pulse_spectroscopy import lorentzian


def example():
    frequency = np.linspace(-500e3, 500e3, 200)
    amps = np.linspace(0.02, 0.3, 9)
    centers = np.linspace(-25e3, 35e3, len(amps))
    sigma = np.linspace(40e3, 65e3, len(amps))
    contrast = np.linspace(0.2, 0.4, len(amps))
    values = 0.6 - contrast * np.exp(-0.5 * ((frequency[:, None] - centers) / sigma) ** 2)
    values += np.random.default_rng(17).normal(0, 0.0003, values.shape)
    values[:, 0] = 0.5  # No resonance at this drive point.
    # Last feature extends beyond the fitted window; never report its width as accepted.
    values[:, -1] = 0.6 - 0.3 * np.exp(-0.5 * ((frequency - 260e3) / 70e3) ** 2)
    ds = xr.Dataset({"state": (("qubit", "detuning", "amp_prefactor"), values[None])},
                    coords={"qubit": ["q1"], "detuning": frequency, "amp_prefactor": amps,
                            "full_amp": (("qubit", "amp_prefactor"), amps[None]),
                            "peak_complex_amplitude_v": (("qubit", "amp_prefactor"), .8*amps[None]),
                            "t2_star_fwhm_limit_hz": ("qubit", [40e3])})
    pulse = SimpleNamespace(amplitude=0.1, length=40,
                            waveform_function=lambda: 0.05 * (1 - np.cos(np.linspace(0, 2*np.pi, 40))))
    qubit = SimpleNamespace(name="q1", xy=SimpleNamespace(operations={"x180": pulse}))
    return ds, qubit, centers, sigma, contrast


def test_fit_recovers_center_width_and_contrast_with_quality_flags():
    ds, qubit, centers, sigma, contrast = example()
    fitted = add_sweep_diagnostics(ds.transpose("amp_prefactor", "qubit", "detuning"), [qubit])
    good = fitted.sweep_fit_accepted.sel(qubit="q1").values
    assert not good[0] and not good[-1] and good[1:-1].all()
    np.testing.assert_allclose(fitted.sweep_center_hz.values[0, 1:-1], centers[1:-1], atol=150)
    np.testing.assert_allclose(fitted.sweep_fwhm_hz.values[0, 1:-1], 2.35482*sigma[1:-1], rtol=.015)
    np.testing.assert_allclose(fitted.sweep_fit_contrast.values[0, 1:-1], contrast[1:-1], atol=.003)
    np.testing.assert_allclose(fitted.sweep_rabi_frequency_mhz.values[0],
                               .8 * ds.amp_prefactor.values / .1 / (2*19.5e-9) / 1e6)
    assert "sweep_center_hz" not in ds
    assert "gaussian_negative_center_hz" not in fitted


def test_figures_use_rabi_and_correct_t2_references():
    ds, qubit, *_ = example()
    fitted = add_sweep_diagnostics(ds, [qubit])
    figures = plot_sweep_diagnostics(fitted)
    try:
        assert len(figures) == 5
        resonance = figures["echo_lorentzian_resonance_q1"].axes[0]
        fwhm = figures["echo_lorentzian_fwhm_q1"].axes[0]
        assert resonance.get_xlabel() == "Peak Rabi frequency [MHz]"
        assert [line.get_ydata()[0] for line in resonance.lines[1:]] == [20, -20]
        assert fwhm.lines[-1].get_ydata()[0] == 40
        assert np.isnan(resonance.lines[0].get_ydata()[0])
        assert any("rejected" in text.get_text().lower()
                   for ax in figures["echo_lorentzian_detuning_slices_q1"].axes for text in ax.texts)
        assert len(figures["echo_lorentzian_detuning_slices_q1"].axes) == 8
    finally:
        for fig in figures.values():
            plt.close(fig)


@pytest.mark.parametrize("plot_enabled,diagnostics_enabled", [(False, True), (True, False), (True, True)])
def test_experiment_registers_savable_analysis_and_figures(monkeypatch, tmp_path, plot_enabled, diagnostics_enabled):
    ds, qubit, *_ = example()
    node = SimpleNamespace(results={"ds_raw": ds}, namespace={"qubits": [qubit]},
                           options=SimpleNamespace(plot_data=plot_enabled),
                           parameters=SimpleNamespace(echo=True, use_state_discrimination=True,
                                                      plot_sweep_diagnostics=diagnostics_enabled))
    monkeypatch.setattr(experiment, "process_raw_dataset", lambda data, node: data)
    experiment.EchoLorentzian.analyse(node)
    enabled = plot_enabled and diagnostics_enabled
    assert ("analysis" in node.results) == enabled
    if not enabled:
        assert "sweep_fit_accepted" not in node.results["ds_raw"]
        return
    saver = CalibrationSaver(output_root=tmp_path)
    saved = saver.save_analysis_result(tmp_path, node.results["analysis"])
    assert "sweep_fit_contrast" in saved.read_text()
    def map_plotter(*args, **kwargs):
        assert kwargs["cmap"] == "viridis"
        return plt.figure()
    monkeypatch.setattr(experiment, "plot_raw_data", map_plotter)
    monkeypatch.setattr(plt, "show", lambda: None)
    experiment.EchoLorentzian.plot_data(node)
    assert len(node.results["figures"]) == 6
    for fig in node.results["figures"].values():
        plt.close(fig)


def test_too_short_scan_and_missing_t2_are_supported():
    ds, qubit, *_ = example()
    assert plot_sweep_diagnostics(add_sweep_diagnostics(ds.isel(detuning=slice(0, 5)), [qubit])) == {}
    fitted = add_sweep_diagnostics(ds.drop_vars("t2_star_fwhm_limit_hz"), [qubit])
    figures = plot_sweep_diagnostics(fitted)
    assert len(figures["echo_lorentzian_fwhm_q1"].axes[0].lines) == 1
    for fig in figures.values():
        plt.close(fig)


def test_actual_complex_peak_uses_coincident_iq_samples_and_absolute_sweep_factor():
    ds, qubit, *_ = example()
    ds = ds.drop_vars("peak_complex_amplitude_v").isel(amp_prefactor=[0, 1, 2])
    ds = ds.assign_coords(amp_prefactor=[-0.5, 0, 1])
    node = SimpleNamespace(parameters=SimpleNamespace(drag_beta=2), namespace={
        "qubits": [qubit], "lorentzian_waveform": [.3, .4, -.3],
        "lorentzian_drag_waveforms_q": {"q1": [.4, 0, -.4]},
    })
    corrected = lorentzian.add_peak_complex_amplitude(ds, node)
    # max |I+iQ| is .5, neither max |I|=.4 nor hypot(max|I|,max|Q|).
    np.testing.assert_allclose(corrected.peak_complex_amplitude_v, [[.25, 0, .5]])
    selected = corrected.sel(qubit="q1")
    mapped = lorentzian._with_plot_coords(selected, qubit)
    sliced = lorentzian._with_amplitude_plot_coords(selected, qubit)
    expected = np.array([.25, 0, .5]) / .1 / (2 * 19.5e-9) / 1e6
    np.testing.assert_allclose(mapped.rabi_frequency_MHz, expected)
    np.testing.assert_allclose(sliced.rabi_frequency_MHz, expected)
    # Saved peaks remain usable after the installed operation/settings disappear.
    assert lorentzian.add_peak_complex_amplitude(corrected, SimpleNamespace()) is corrected


def test_smoothed_peak_is_saved_before_analysis(tmp_path):
    ds, qubit, *_ = example()
    ds = ds.drop_vars("peak_complex_amplitude_v")
    parameters = SimpleNamespace(
        pulse_shape="root_lorentzian", lorentzian_length_in_ns=20000,
        lorentzian_peak_amplitude=1, cutoff=.005, echo=True,
        echo_transition_time_ns=32, edge_smoothing="polynomial",
        edge_smoothing_power=2, drag_beta=0,
    )
    captured = {}
    def save(name, dataset, **kwargs):
        captured["dataset"] = dataset
        return tmp_path
    node = SimpleNamespace(
        name="echo_lorentzian", parameters=parameters, namespace={"qubits": [qubit]},
        results={"ds_raw": ds}, saver=SimpleNamespace(save_xarray=save),
        active_profile_name=lambda: "saved", run_timing_metadata=lambda: {}, log=lambda message: None,
    )
    experiment.EchoLorentzian.save_raw_results(node)
    peak = np.max(np.abs(lorentzian.build_waveform(parameters)))
    assert 0 < peak < .99  # Smoothing lowers the peak relative to the nominal 1 V.
    np.testing.assert_allclose(captured["dataset"].peak_complex_amplitude_v,
                               peak * abs(ds.amp_prefactor.values)[None])
    assert "sweep_fit_accepted" not in captured["dataset"]


def test_rabi_axis_refuses_nominal_only_data():
    ds, qubit, *_ = example()
    with pytest.raises(ValueError, match="Actual pulse peak is missing"):
        lorentzian.peak_rabi_frequency_mhz(ds.drop_vars("peak_complex_amplitude_v"), qubit)
