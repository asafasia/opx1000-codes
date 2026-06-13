"""Robust QUA simulation and plotting shared by calibration experiments."""

import logging

import matplotlib.pyplot as plt
from qm import SimulationConfig
from qm.exceptions import QMSimulationError

from utils.plotting_settings import FIGURE_SIZE

logger = logging.getLogger(__name__)


def plot_waveform_report_safely(waveform_report, samples=None):
    """Plot a waveform report without allowing renderer compatibility bugs to crash simulation."""
    try:
        waveform_report.create_plot(samples=samples, plot=True, save_path=None)
        return plt.gcf()
    except Exception as error:
        logger.warning(
            "QOP produced a waveform report, but qm-qua could not render it: %s",
            error,
        )
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "Simulation completed, but qm-qua could not render the waveform report.\n"
            f"{type(error).__name__}: {error}",
            ha="center",
            va="center",
            wrap=True,
        )
        figure.tight_layout()
        return figure


def simulate_and_plot(qmm, config, program, node_parameters):
    """Simulate a QUA program and fall back to its waveform report if samples fail."""
    simulation_config = SimulationConfig(
        duration=node_parameters.simulation_duration_ns // 4
    )
    job = qmm.simulate(config, program, simulation_config)

    # QOP v2 can return the simulated job before its samples are ready.
    if hasattr(job, "wait_until"):
        job.wait_until("Done", timeout=node_parameters.timeout)

    waveform_report = None
    if node_parameters.use_waveform_report:
        waveform_report = job.get_simulated_waveform_report()

    try:
        samples = job.get_simulated_samples()
    except QMSimulationError:
        logger.warning(
            "QOP simulated the program but qm-qua could not pull analog samples; "
            "showing the waveform report instead."
        )
        if waveform_report is None:
            waveform_report = job.get_simulated_waveform_report()
        figure = plot_waveform_report_safely(waveform_report, samples=None)
        return None, figure, waveform_report

    figure, axes = plt.subplots(
        nrows=len(samples.keys()),
        sharex=True,
        squeeze=False,
        figsize=FIGURE_SIZE,
    )
    for axis, controller in zip(axes.flat, samples.keys()):
        plt.sca(axis)
        samples[controller].plot()
        axis.set_title(controller)
    figure.tight_layout()

    if waveform_report is not None:
        plot_waveform_report_safely(waveform_report, samples)

    return samples, figure, waveform_report
