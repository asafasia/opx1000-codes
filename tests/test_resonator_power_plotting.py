import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibration_utils.resonator_spectroscopy_vs_amplitude.plotting import (
    _add_voltage_axis,
    _dbm_to_peak_voltage,
    _peak_voltage_to_dbm,
)


def test_output_voltage_conversion_uses_peak_voltage_into_50_ohms():
    powers = np.array([-50.0, -30.0, -10.0, 0.0, 10.0])
    np.testing.assert_allclose(
        _dbm_to_peak_voltage(powers), [0.001, 0.01, 0.1, np.sqrt(0.1), 1.0]
    )
    np.testing.assert_allclose(_peak_voltage_to_dbm(_dbm_to_peak_voltage(powers)), powers)


def test_voltage_ticks_align_with_power_ticks_after_zoom():
    fig, ax = plt.subplots()
    try:
        ax.set_ylim(-50, -5)
        secondary = _add_voltage_axis(ax)
        for limits in [(-50, -5), (-35, -15)]:
            ax.set_ylim(*limits)
            fig.canvas.draw()
            powers = ax.get_yticks()
            voltages = secondary.get_yticks()
            np.testing.assert_allclose(voltages, _dbm_to_peak_voltage(powers))
            primary_pixels = ax.transData.transform(np.column_stack((np.zeros_like(powers), powers)))
            secondary_pixels = secondary.transData.transform(np.column_stack((np.zeros_like(voltages), voltages)))
            np.testing.assert_allclose(primary_pixels[:, 1], secondary_pixels[:, 1])
    finally:
        plt.close(fig)
