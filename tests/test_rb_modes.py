"""Offline checks for RB mode acquisition and known population models."""
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from calibration_utils.single_qubit_randomized_benchmarking.parameters import Parameters
from calibration_utils.single_qubit_randomized_benchmarking.analysis import fit_raw_data
from calibration_utils.single_qubit_randomized_benchmarking.plotting import plot_individual_data_with_fit
from calibration_utils.single_qubit_randomized_benchmarking.modes import interleaved_metrics
from calibrations import CalibrationOptions
from quam_config.create_machine_from_profile import create_machine_from_profile


def make_node(mode, **kwargs):
    return SimpleNamespace(parameters=Parameters(mode=mode, fidelity_bootstrap_samples=6,
        fidelity_bootstrap_seed=4, **kwargs), outcomes={}, log=lambda _: None)


def leakage_data(flat=False):
    x = np.arange(1, 802, 20)
    # Markov population process with leakage .002 and seepage .008 per step.
    pc = np.full_like(x, .99, dtype=float) if flat else .8 + .19 * .99 ** x
    pg = pc / 2 + .48 * .98 ** x
    ds = xr.Dataset(coords={"qubit": ["q1"], "depths": x, "nb_of_sequences": np.arange(5)})
    for s, y in zip("gef", [pg, pc - pg, 1 - pc]):
        ds[f"population_{s}"] = (("qubit", "depths", "nb_of_sequences"), np.broadcast_to(y[None, :, None], (1, len(x), 5)).copy())
    ds["state"] = ds.population_e
    return ds


def interleaved_data(pc=.98):
    x = np.arange(1, 402, 10)
    curves = np.array([.49 + .48 * .99 ** x, .47 + .46 * pc ** x])
    ds = xr.Dataset({"population_g": (("qubit", "rb_variant", "nb_of_sequences", "depths"),
        np.broadcast_to(curves[None, :, None, :], (1, 2, 5, len(x))).copy())},
        coords={"qubit": ["q1"], "rb_variant": ["reference", "interleaved"], "nb_of_sequences": np.arange(5), "depths": x})
    ds["state"] = 1 - ds.population_g
    return ds


def test_leakage_rates_and_plot():
    ds = leakage_data()
    fit, results = fit_raw_data(ds, make_node("leakage", readout_states=["g", "e", "f"]))
    r = results["q1"]
    assert r.success, r.message
    assert r.leakage_per_clifford == pytest.approx(.002, abs=1e-7)
    assert r.seepage_per_clifford == pytest.approx(.008, abs=1e-7)
    assert np.isnan(r.error_per_gate)
    assert r.bootstrap_successes == 6
    np.testing.assert_allclose(fit.computational_population, 1 - ds.population_f)
    fig, ax = plt.subplots()
    plot_individual_data_with_fit(ax, ds, {"qubit": "q1"}, fit.sel(qubit="q1"))
    assert "Leakage fit" in ax.get_legend_handles_labels()[1]
    plt.close(fig)


def test_interleaved_ratio_not_standard_epg():
    ds = interleaved_data()
    fit, results = fit_raw_data(ds, make_node("interleaved"))
    r = results["q1"]
    assert r.success, r.message
    assert r.error_per_gate == pytest.approx((1 - .98 / .99) / 2, abs=1e-7)
    assert r.systematic_error_bound >= 0
    assert r.error_bound_low <= r.error_per_gate <= r.error_bound_high
    assert r.bootstrap_successes == 6
    fig, ax = plt.subplots()
    plot_individual_data_with_fit(ax, ds, {"qubit": "q1"}, fit.sel(qubit="q1"))
    assert "Reference" in ax.get_legend_handles_labels()[1]
    plt.close(fig)


def test_negative_irb_estimate_is_not_clipped_or_accepted():
    _, results = fit_raw_data(interleaved_data(.995), make_node("interleaved"))
    assert results["q1"].error_per_gate < 0
    assert not results["q1"].success


def test_missing_reference_rejected():
    with pytest.raises(ValueError, match="reference"):
        fit_raw_data(interleaved_data().sel(rb_variant="interleaved", drop=True), make_node("interleaved"))


def test_flat_leakage_is_unidentifiable():
    fit, results = fit_raw_data(leakage_data(flat=True), make_node("leakage"))
    assert not results["q1"].success
    assert np.isnan(results["q1"].seepage_per_clifford)
    assert "population_f" in fit


def test_missing_leakage_populations_rejected():
    with pytest.raises(ValueError, match="population_g/e/f"):
        fit_raw_data(leakage_data().drop_vars("population_f"), make_node("leakage"))


@pytest.mark.parametrize("mode", ["standard", "interleaved", "leakage"])
def test_qua_builds_offline_with_gef(mode):
    module = import_module("calibrations.11a_single_qubit_randomized_benchmarking")
    machine = create_machine_from_profile("single_qubit", qubit="q1", save=False)
    machine.qubits["q1"].resonator.readout_gef["gef_centers"] = [[-.001, 0], [.001, 0], [0, .002]]
    parameters = Parameters(mode=mode, readout_states=["g", "e", "f"], qubits=["q1"],
        reset_type="thermal", num_shots=2, num_random_sequences=3, max_circuit_depth=7, delta_clifford=3)
    node = module.SingleQubitRandomizedBenchmarking(parameters, machine=machine,
        options=CalibrationOptions(save_raw_data=False, save_figures=False, plot_data=False,
            update_state=False, propose_profile_update=False, apply_profile_update=False))
    with patch.object(machine, "connect", side_effect=AssertionError("Hardware access forbidden")):
        assert node.create_qua_program() is not None
        assert ("rb_variant" in node.namespace["sweep_axes"]) == (mode == "interleaved")
        np.testing.assert_array_equal(node.namespace["sweep_axes"]["depths"], [1, 3, 6, 7])
        machine.generate_config()


def test_leakage_requires_gef_before_program_construction():
    module = import_module("calibrations.11a_single_qubit_randomized_benchmarking")
    node = module.SingleQubitRandomizedBenchmarking(Parameters(mode="leakage"), machine=object())
    with pytest.raises(ValueError, match="Leakage RB requires"):
        node.create_qua_program()


def test_irb_bound_perfect_reference():
    error, bound = interleaved_metrics(1., .98)
    assert error == pytest.approx(.01)
    assert bound == 0


def test_analog_irb_offsets_are_not_constrained_to_probabilities():
    ds = interleaved_data()
    ds["I"] = 4.0 - 2.0 * ds.population_g
    fit, results = fit_raw_data(ds.drop_vars(["population_g", "state"]),
                                make_node("interleaved", use_state_discrimination=False))
    assert results["q1"].success
    assert results["q1"].error_per_gate == pytest.approx((1 - .98 / .99) / 2, abs=1e-7)
    assert float(abs(fit.fit_residual).max()) < 1e-6


def test_metadata_mode_mismatch_rejected():
    ds = interleaved_data()
    ds.attrs["rb_mode"] = "interleaved"
    with pytest.raises(ValueError, match="does not match"):
        fit_raw_data(ds, make_node("standard"))


def test_legacy_interleaved_uses_paired_acquisition():
    module = import_module("calibrations.11b_single_qubit_randomized_benchmarking_interleaved")
    machine = create_machine_from_profile("single_qubit", qubit="q1", save=False)
    node = module.SingleQubitRandomizedBenchmarkingInterleaved(
        module.Parameters(qubits=["q1"], reset_type="thermal", max_circuit_depth=7,
                          delta_clifford=3, num_random_sequences=2, num_shots=2), machine=machine)
    with patch.object(machine, "connect", side_effect=AssertionError("Hardware access forbidden")):
        assert node.create_qua_program() is not None
    assert list(node.namespace["sweep_axes"]["rb_variant"].values) == ["reference", "interleaved"]


def test_noisy_leakage_recovers_rates_and_nonzero_uncertainties():
    ds = leakage_data()
    rng = np.random.default_rng(27)
    for seq in ds.nb_of_sequences.values:
        pg = ds.population_g.sel(qubit="q1", nb_of_sequences=seq).values
        pe = ds.population_e.sel(qubit="q1", nb_of_sequences=seq).values
        counts = np.array([rng.multinomial(20000, [g, e, 1-g-e]) for g, e in zip(pg, pe)]) / 20000
        for i, state in enumerate("gef"):
            ds[f"population_{state}"].loc[{"qubit": "q1", "nb_of_sequences": seq}] = counts[:, i]
    ds["state"] = ds.population_e
    fit, results = fit_raw_data(ds, make_node("leakage"))
    r = results["q1"]
    assert r.success, r.message
    assert r.leakage_per_clifford == pytest.approx(.002, rel=.08)
    assert r.seepage_per_clifford == pytest.approx(.008, rel=.08)
    assert r.leakage_per_clifford_std > 0
    assert r.seepage_per_clifford_std > 0


@pytest.mark.parametrize("mode", ["standard", "interleaved", "leakage"])
def test_profile_proposal_is_mode_specific_and_never_applies_by_default(mode):
    from unittest.mock import Mock
    module = import_module("calibrations.11a_single_qubit_randomized_benchmarking")
    updater = Mock()
    node = module.SingleQubitRandomizedBenchmarking(Parameters(mode=mode),
        machine=object(), profile_name="single_qubit", profile_updater=updater,
        options=CalibrationOptions(apply_profile_update=False))
    node.namespace["qubits"] = [SimpleNamespace(name="q1")]
    node.outcomes = {"q1": "successful"}
    node.results["fit_results"] = {"q1": {"error_per_gate": .01}}
    assert node._propose_profile_update_from_options() == (mode == "standard")
    assert updater.stage.call_count == (1 if mode == "standard" else 0)
    updater.confirm_and_apply.assert_not_called()
