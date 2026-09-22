"""Offline regression checks for separate ground/excited T1 storage."""

from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from quam_builder.architecture.superconducting.qubit import FixedFrequencyTransmon

from calibration_utils.analysis_base import AnalysisResult
from calibration_utils.T1.analysis import FIT_VALUES, fit_raw_data, log_fitted_results
from calibration_utils.T1.plotting import plot_individual_data_with_fit
from parameter_scans.runner import extract_fit_records
from profiles import load_profile
from profiles.loader import ProfileError, validate_profile
from quam_config.populate_quam_lf_mw_fems import _apply_transmon_times


t1_module = import_module("calibrations.05_T1")


@pytest.mark.parametrize("initial_state,label,key", [("g", "T1_ge", "t1_ge"), ("e", "T1", "t1")])
def test_result_serialization_scan_labels_and_plot(initial_state, label, key):
    times = np.linspace(16, 250_000, 100)
    amplitude = -0.1 if initial_state == "g" else 0.9
    ds = xr.Dataset(
        {"state": (("qubit", "idle_time"), [0.1 + amplitude * np.exp(-times / 30_000)])},
        coords={"qubit": ["q1"], "idle_time": times},
    )
    node = SimpleNamespace(
        parameters=SimpleNamespace(initial_state=initial_state, use_state_discrimination=True),
        log=lambda message: None,
    )
    # Use a known fit for the unchanged library path; exercise the real g fit.
    values = np.zeros((1, len(FIT_VALUES)))
    values[0, :3] = [amplitude, 0.1, -1 / 30_000]
    known_fit = xr.DataArray(values, dims=["qubit", "fit_vals"],
                             coords={"qubit": ["q1"], "fit_vals": FIT_VALUES})
    with patch("calibration_utils.T1.analysis.fit_decay_exp", return_value=known_fit):
        fitted, results = fit_raw_data(ds, node)
    payload = AnalysisResult(ds_fit=fitted, fit_results=results).to_dict()
    assert set(payload["fit_results"]["q1"]) == {key, f"{key}_error", "success"}
    assert payload["fit_results"]["q1"][key] == pytest.approx(30_000)
    assert fitted.tau.attrs["long_name"] == label
    if initial_state == "g":
        assert float(fitted.T1_ge.sel(qubit="q1")) == pytest.approx(30_000)

    node.results = {"fit_results": results}
    records = extract_fit_records(node, timestamp="2026-09-16", cycle=1,
                                  experiment_name="05_T1", script=Path("calibrations/05_T1.py"),
                                  duration_s=1.0)
    assert [record["parameter"] for record in records] == [label, f"{label} error"]
    assert all(record["unit"] == "ns" for record in records)
    messages = []
    log_fitted_results(fitted, messages.append)
    assert messages[0].startswith(f"{label} for qubit q1")
    fig, ax = plt.subplots()
    try:
        plot_individual_data_with_fit(ax, ds, {"qubit": "q1"}, fitted.sel(qubit="q1"))
        assert ax.texts[0].get_text().startswith(f"{label} =")
    finally:
        plt.close(fig)


@pytest.mark.parametrize("initial_state", ["g", "e"])
def test_state_and_profile_updates_keep_transitions_separate(initial_state):
    qubit = FixedFrequencyTransmon(id="q1", T1=40e-6, extras={"T1_ge": 50e-6})
    failed = FixedFrequencyTransmon(id="q2", T1=60e-6, extras={"T1_ge": 70e-6})
    node = SimpleNamespace(
        name="05_T1",
        parameters=SimpleNamespace(initial_state=initial_state),
        namespace={"qubits": [qubit, failed]},
        outcomes={"q1": "successful", "q2": "failed"},
        results={"ds_fit": xr.Dataset(coords={"qubit": ["q1", "q2"],
                                             "tau": ("qubit", [30_000.0, 80_000.0])})},
        record_state_updates=nullcontext,
    )
    t1_module.T1.update_state(node)
    assert qubit.T1 == pytest.approx(40e-6 if initial_state == "g" else 30e-6)
    assert qubit.extras["T1_ge"] == pytest.approx(30e-6 if initial_state == "g" else 50e-6)
    assert failed.T1 == 60e-6
    assert failed.extras["T1_ge"] == 70e-6
    assert qubit.to_dict()["extras"]["T1_ge"] == qubit.extras["T1_ge"]
    with patch.object(t1_module, "ProfileUpdater") as updater:
        t1_module.T1.propose_profile_update(node)
    metric = "t1_ge_ns" if initial_state == "g" else "t1_ns"
    assert updater.return_value.stage.call_args.args[1] == {
        f"metrics.json.qubits.q1.coherence.{metric}": 30_000.0
    }


def test_profile_reload_retains_t1_ge_without_changing_thermalization():
    qubit = FixedFrequencyTransmon(id="q1")
    transmon = {"thermalization_time_ns": 200_000}
    _apply_transmon_times(qubit, transmon, {"t1_ns": 40_000, "t1_ge_ns": 30_000})
    assert qubit.T1 == pytest.approx(40e-6)
    assert qubit.extras["T1_ge"] == pytest.approx(30e-6)
    assert qubit.thermalization_time_factor == 5


def test_optional_profile_t1_ge_metric_is_validated():
    profile = load_profile("main")
    coherence = profile["metrics"]["qubits"]["q1"]["coherence"]
    coherence["t1_ge_ns"] = 30_000
    validate_profile(profile)
    coherence["t1_ge_ns"] = "invalid"
    with pytest.raises(ProfileError, match="t1_ge_ns"):
        validate_profile(profile)
