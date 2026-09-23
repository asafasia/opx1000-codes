from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from calibration_utils.iq_blobs.analysis import fit_raw_data
from calibration_utils.iq_blobs.parameters import Parameters
from calibration_utils.iq_blobs.plotting import plot_individual_iq_blobs


@pytest.mark.parametrize("states", ["ge", "gef"])
@pytest.mark.parametrize("method", ["mean", "median"])
def test_outlier_centers_rotation_confusion_and_markers(states, method):
    clouds = {}
    for index, state in enumerate(states):
        clouds[f"I{state}"] = (("qubit", "n_runs"), [np.array([0, .1, -.1, 0, 12]) * .001 + index * .003])
        clouds[f"Q{state}"] = (("qubit", "n_runs"), [np.array([0, -.1, .1, 0, -8]) * .001 * (index + 1)])
    ds = xr.Dataset(clouds, coords={"qubit": ["q1"], "n_runs": np.arange(5)})
    node = SimpleNamespace(parameters=Parameters(center_method=method), namespace={"qubits": [SimpleNamespace(name="q1")]})
    fit, results = fit_raw_data(ds, node)
    reducer = getattr(np, method)
    expected = np.array([[reducer(ds[f"I{s}"].values), reducer(ds[f"Q{s}"].values)] for s in states])
    np.testing.assert_allclose(fit.state_center_matrix.sel(qubit="q1"), expected)
    assert results["q1"].center_method == method
    assert fit.attrs["center_method"] == method
    delta = expected[1] - expected[0]
    assert float(fit.iw_angle.item()) == pytest.approx(np.arctan2(-delta[1], delta[0]))
    assert results["q1"].center_separation == pytest.approx(np.linalg.norm(delta))
    for j, state in enumerate(states):
        points = np.column_stack([ds[f"I{state}"].values.ravel(), ds[f"Q{state}"].values.ravel()])
        classified = np.argmin(np.linalg.norm(points[:, None] - expected[None], axis=2), axis=1)
        np.testing.assert_allclose(results["q1"].confusion_matrix[j], np.bincount(classified, minlength=len(states)) / 5)
    fig, ax = plt.subplots()
    plot_individual_iq_blobs(ax, ds, {"qubit": "q1"}, fit.sel(qubit="q1"))
    markers = {line.get_label(): line for line in ax.lines}
    expected_display = expected.copy()
    if states == "gef":
        angle = np.arctan2(-delta[1], delta[0])
        c, s = np.cos(angle), np.sin(angle)
        expected_display = expected @ np.array([[c, s], [-s, c]])
    for j, label in enumerate(["Ground", "Prepared", "F"][:len(states)]):
        np.testing.assert_allclose(np.array(markers[f"{label} center"].get_data()).ravel(), expected_display[j] * 1000, atol=1e-12)
    plt.close(fig)
    # Estimator changes must not discard or alter acquired shots.
    assert ds.sizes["n_runs"] == 5
    assert "center_method" not in ds.attrs


def test_center_method_validation_and_legacy_default():
    assert Parameters().center_method == "mean"
    with pytest.raises(ValueError):
        Parameters(center_method="mode")
