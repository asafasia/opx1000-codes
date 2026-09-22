import numpy as np
import xarray as xr
from calibration_utils.iq_blobs.plotting import _robust_iq_limits


def test_extreme_outliers_do_not_expand_the_cloud_view():
    rng = np.random.default_rng(4)
    values = {}
    for state, center in zip("gef", [(-1, 0), (1, 0), (0, 1)]):
        cloud = rng.normal(center, 0.1, (1000, 2)) / 1e3
        for k, quadrature in enumerate("IQ"):
            values[quadrature + state] = ("shot", cloud[:, k])
    ds = xr.Dataset(values)
    before = np.asarray(_robust_iq_limits(ds))
    extreme = ds.copy(deep=True)
    extreme.Ig.values[0] = 100
    extreme.Qf.values[-1] = -100
    after = np.asarray(_robust_iq_limits(extreme))
    np.testing.assert_allclose(after, before, atol=0.02)
    assert np.max(np.abs(after)) < 2
    assert extreme.Ig.values[0] == 100


def test_degenerate_and_nonfinite_clouds_have_finite_limits():
    ds = xr.Dataset({"Ig": ("shot", [0., 0., 0., np.nan, 1000.]),
                     "Qg": ("shot", [0., 0., 0., np.inf, 0.])})
    limits = np.asarray(_robust_iq_limits(ds))
    assert np.isfinite(limits).all()
    assert np.all(limits[:, 1] > limits[:, 0])
    empty = xr.Dataset({"Ig": ("shot", [np.nan]), "Qg": ("shot", [np.nan])})
    assert _robust_iq_limits(empty) is None
