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

        best = selected.readout_fidelity.argmax(dim=("detuning", "amp_prefactor"))
        best_detuning = selected.detuning.isel(detuning=best["detuning"])
        best_amp = selected.amp_prefactor.isel(amp_prefactor=best["amp_prefactor"])
        best_freq = selected.full_freq.sel(detuning=best_detuning) / 1e9
        best_amplitude = selected.readout_amplitude.sel(amp_prefactor=best_amp) * 1e3
        for ax in axes:
            ax.plot(best_freq, best_amplitude, "wo", markersize=5, markeredgecolor="black")

        fig.suptitle(f"{q.name} Readout Frequency-Amplitude Optimization")
        figures[f"frequency_amplitude_maps_{q.name}"] = fig
    return figures
