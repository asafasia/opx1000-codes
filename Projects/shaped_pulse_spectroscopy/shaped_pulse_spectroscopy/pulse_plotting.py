"""Linear and log views of the actual baseband pulse used by spectroscopy."""

import matplotlib.pyplot as plt
import numpy as np

from .lorentzian import pulse_iq_samples
from utils.rabi_amplitude import amplitude_to_rabi_frequency_hz, pi_pulse_effective_length_ns


def plot_pulse_waveforms(parameters, namespace, amplitude_factors) -> dict:
    """Plot the largest swept pulse: signed I/Q on linear y, magnitudes on log y.

    The plots use physical playback time, including waveform stretching, and
    sampled pi-area calibration. Zero samples are masked on logarithmic axes.
    Chirped runs show the baseband envelope before carrier frequency modulation.
    """
    factors = np.asarray(amplitude_factors, dtype=float)
    if factors.size == 0 or not np.isfinite(factors).all():
        raise ValueError("Pulse plots require finite, nonempty amplitude factors.")
    factor = float(factors.flat[np.argmax(np.abs(factors))])
    duration = float(parameters.lorentzian_length_in_ns)
    transition = float(getattr(parameters, "echo_transition_time_ns", 0)) if parameters.echo else 0
    figures = {}
    for qubit in namespace["qubits"]:
        i, q = pulse_iq_samples(parameters, namespace, qubit)
        reference = qubit.xy.operations["x180"]
        scale = amplitude_to_rabi_frequency_hz(
            factor, abs(float(reference.amplitude)), pi_pulse_effective_length_ns(reference)
        ) / 1e6
        i, q = i * scale, q * scale
        magnitude = np.abs(i + 1j*q)
        times_ns = (np.arange(i.size) + 0.5) * duration / i.size
        centered_ns = times_ns - duration / 2
        zoom_half_width = min(duration / 2, max(4 * transition, 100))
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for column, (times, xlabel, limits) in enumerate((
            (times_ns / 1000, "Time [µs]", (0, duration / 1000)),
            (centered_ns, "Time from pulse midpoint [ns]", (-zoom_half_width, zoom_half_width)),
        )):
            visible = (times >= limits[0]) & (times <= limits[1])
            for row in (0, 1):
                ax = axes[row, column]
                for values, label, color in ((i, "I", "#2878a0"), (q, "Q", "#cc6b27"),
                                              (magnitude, "|I + iQ|", "#278448")):
                    plotted = values if row == 0 else np.ma.masked_less_equal(np.abs(values), 0)
                    ax.plot(times[visible], plotted[visible], label=label if row == 0 or label.startswith("|") else f"|{label}|",
                            color=color, lw=1.2)
                if row == 1:
                    ax.set_yscale("log")
                else:
                    ax.axhline(0, color="0.5", lw=0.6)
                if transition > 0:
                    bounds = (duration/2-transition/2, duration/2+transition/2)
                    left, right = (np.array(bounds)/1000 if column == 0 else np.array(bounds)-duration/2)
                    ax.axvspan(left, right, color="0.5", alpha=0.12)
                ax.set(xlabel=xlabel, xlim=limits,
                       ylabel="Rabi-equivalent amplitude [MHz]" if row == 0 else "Magnitude [MHz]",
                       title=("Full pulse" if column == 0 else "Midpoint detail") + (" · linear y" if row == 0 else " · logarithmic y"))
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
        fig.suptitle(
            f"{qubit.name}: baseband pulse | β={parameters.drag_beta:g} | "
            f"transition={transition:g} ns | peak={magnitude.max():.2f} MHz | "
            f"amplitude factor={factor:g}"
        )
        figures[f"echo_lorentzian_pulse_{qubit.name}"] = fig
    return figures
