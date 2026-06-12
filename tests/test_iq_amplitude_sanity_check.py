"""Offline checks for the two-amplitude IQ sanity-check analysis."""

import numpy as np

from ising_machine.iq_amplitude_sanity_check import iq_summary, plot_iq_results


def test_iq_summary_ratio() -> None:
    results = {
        "high_i": np.asarray([2.0, 2.0]),
        "high_q": np.asarray([0.0, 0.0]),
        "low_i": np.asarray([1.0, 1.0]),
        "low_q": np.asarray([0.0, 0.0]),
    }
    summary = iq_summary(results)
    assert summary["high_magnitude"] == 2.0
    assert summary["low_magnitude"] == 1.0
    assert summary["magnitude_ratio"] == 0.5


def test_iq_plot() -> None:
    results = {
        "high_i": np.asarray([1.9, 2.1]),
        "high_q": np.asarray([-0.1, 0.1]),
        "low_i": np.asarray([0.9, 1.1]),
        "low_q": np.asarray([-0.1, 0.1]),
    }
    figure = plot_iq_results(results)
    assert len(figure.axes) == 1


if __name__ == "__main__":
    test_iq_summary_ratio()
    test_iq_plot()
    print("IQ amplitude sanity-check tests passed.")
