from importlib import import_module
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

MODULE = import_module("calibrations.08c_readout_frequency_amplitude_optimization")


@pytest.mark.parametrize("states, section, field, operation", [
    (["g", "e"], "readout", "frequencies_hz.resonator", "readout"),
    (["g", "e", "f"], "readout_gef", "readout_gef.frequency_hz", "readout_GEF"),
])
def test_proposal_targets_selected_frequency_and_pulse(states, section, field, operation):
    node = SimpleNamespace(
        parameters=SimpleNamespace(readout_states=states, readout_operation=operation, reset_type="thermal"),
        namespace={"qubits": [SimpleNamespace(name="q1", resonator=SimpleNamespace(readout_pulse_names={operation: "selected_pulse"}))]},
        results={"fit_results": {"q1": dict(success=True, optimal_frequency=6.7e9, optimal_amplitude=.12, readout_fidelity=95)}},
        log=lambda message: None,
    )
    updates = MODULE.ReadoutFrequencyAmplitudeOptimization.profile_updates(node)
    assert updates[f"qubits.json.qubits.q1.{field}"] == 6.7e9
    assert updates["pulses.json.pulses.q1.selected_pulse.amplitude"] == .12
    assert updates[f"qubits.json.qubits.q1.{section}.gef_centers"] is None
    assert updates[f"qubits.json.qubits.q1.{section}.confusion_matrix"] is None
    if section == "readout":
        assert updates["metrics.json.qubits.q1.readout.fidelity_percent.thermal"] == 95
    for bad_amplitude in (float("nan"), float("inf"), .71, -.71):
        node.results["fit_results"]["q1"]["optimal_amplitude"] = bad_amplitude
        assert MODULE.ReadoutFrequencyAmplitudeOptimization.profile_updates(node) == {}
    node.results["fit_results"]["q1"].update(success=False, optimal_amplitude=.12)
    assert MODULE.ReadoutFrequencyAmplitudeOptimization.profile_updates(node) == {}


@pytest.mark.parametrize("valid", [True, False])
def test_plot_reports_findings_and_handles_missing_fidelity(valid):
    from calibration_utils.readout_frequency_amplitude_optimization.plotting import plot_optimization_maps
    dims = ("qubit", "detuning", "amp_prefactor")
    ds = xr.Dataset(
        {
            "readout_fidelity": (dims, [[[60., 95.], [80., 90.]]] if valid else np.full((1, 2, 2), np.nan)),
            "state_difference": (dims, np.full((1, 2, 2), .002)),
            "separation_to_width": (dims, np.full((1, 2, 2), 3.)),
            "success": ("qubit", [valid]),
        },
        coords={"qubit": ["q1"], "detuning": [-1e6, 0.], "amp_prefactor": [.5, 1.],
                "full_freq": (("qubit", "detuning"), [[6.699e9, 6.7e9]]),
                "readout_amplitude": (("qubit", "amp_prefactor"), [[.05, .1]])},
    )
    fig = plot_optimization_maps(ds, [SimpleNamespace(name="q1")], ds)["frequency_amplitude_maps_q1"]
    fig.canvas.draw()
    title = fig._suptitle.get_text()
    if valid:
        assert "6.699000 GHz" in title
        assert "100.00 mV" in title
        assert "95.0%" in title
        assert len(fig.axes[0].lines) == 1
    else:
        assert "No finite fidelity" in title
        assert not fig.axes[0].lines
    plt.close(fig)
