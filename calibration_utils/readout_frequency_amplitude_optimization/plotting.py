from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_optimization_maps(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    figures = {}
    for q in qubits:
        selected = fits.sel(qubit=q.name)
        frequency_ghz = np.asarray(selected.full_freq) / 1e9
        amplitude_mv = np.asarray(selected.readout_amplitude) * 1e3

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        diff = 1e3 * selected.state_difference.transpose("amp_prefactor", "detuning")
        fidelity = selected.readout_fidelity.transpose("amp_prefactor", "detuning")

        diff_plot = axes[0].pcolormesh(
            frequency_ghz,
            amplitude_mv,
            diff,
            shading="auto",
            cmap="viridis",
        )
        fig.colorbar(diff_plot, ax=axes[0], label="|e - g| [mV]")
        axes[0].set_title("State Difference")
        axes[0].set_xlabel("Readout RF frequency [GHz]")
        axes[0].set_ylabel("Readout amplitude [mV]")

        fidelity_plot = axes[1].pcolormesh(
            frequency_ghz,
            amplitude_mv,
            fidelity,
            shading="auto",
            cmap="magma",
            vmin=50,
            vmax=100,
        )
        fig.colorbar(fidelity_plot, ax=axes[1], label="Fidelity [%]")
        axes[1].set_title("Readout Fidelity")
        axes[1].set_xlabel("Readout RF frequency [GHz]")
        axes[1].set_ylabel("Readout amplitude [mV]")

        if np.isfinite(selected.readout_fidelity).any():
            values = selected.readout_fidelity.transpose("detuning", "amp_prefactor").values
            di, ai = np.unravel_index(np.nanargmax(values), values.shape)
            point = selected.isel(detuning=di, amp_prefactor=ai)
            best_freq = float(point.full_freq) / 1e9
            best_amplitude = float(point.readout_amplitude) * 1e3
            for ax in axes:
                ax.plot(best_freq, best_amplitude, "wo", markersize=7, markeredgecolor="black")
            status = "SUCCESS" if bool(selected.success) else "FAILED"
            findings = (
                f"{status} | Best RF: {best_freq:.6f} GHz | Detuning: {float(point.detuning) / 1e6:+.2f} MHz\n"
                f"Amplitude: {best_amplitude:.2f} mV (prefactor {float(point.amp_prefactor):.3f}) | "
                f"Fidelity: {float(point.readout_fidelity):.1f}%\n"
                f"State difference: {float(point.state_difference) * 1e3:.3f} mV | "
                f"Separation/width: {float(point.separation_to_width):.2f}"
            )
        else:
            findings = "FAILED | No finite fidelity values; no optimum found."

        fig.set_size_inches(12, 5.5)
        fig.suptitle(f"{q.name} Readout Frequency-Amplitude Optimization\n{findings}", fontsize=11)
        figures[f"frequency_amplitude_maps_{q.name}"] = fig
    return figures
