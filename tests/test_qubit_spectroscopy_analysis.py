from types import SimpleNamespace

import xarray as xr

from calibration_utils.qubit_spectroscopy.analysis import _extract_relevant_fit_parameters


def test_large_suggested_saturation_amplitude_does_not_fail_valid_peak():
    qubit = SimpleNamespace(
        name="q1",
        xy=SimpleNamespace(
            RF_frequency=4.5e9,
            operations={
                "saturation": SimpleNamespace(amplitude=0.2),
                "x180": SimpleNamespace(length=40),
            },
        ),
        resonator=SimpleNamespace(
            operations={"readout": SimpleNamespace(integration_weights_angle=0.0)}
        ),
    )
    node = SimpleNamespace(
        name="03a_qubit_spectroscopy",
        namespace={"qubits": [qubit]},
        parameters=SimpleNamespace(
            frequency_span_in_mhz=500,
            frequency_step_in_mhz=0.5,
            operation_amplitude_factor=1,
            target_peak_width=3e6,
            use_state_discrimination=False,
        ),
    )
    fit = xr.Dataset(
        {
            "position": ("qubit", [0.0]),
            "width": ("qubit", [1.0]),
            "iw_angle": ("qubit", [0.0]),
        },
        coords={"qubit": ["q1"]},
    )

    fit_data, fit_results = _extract_relevant_fit_parameters(fit, node)

    assert bool(fit_data.sel(qubit="q1").success.values)
    assert fit_results["q1"].success
