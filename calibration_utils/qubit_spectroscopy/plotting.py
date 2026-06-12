from typing import List
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def _add_detuning_axis(ax, current_frequency_ghz: float):
    """Add a top x-axis showing detuning from the configured resonance."""
    detuning_axis = ax.secondary_xaxis(
        "top",
        functions=(
            lambda frequency_ghz: (frequency_ghz - current_frequency_ghz) * 1e3,
            lambda detuning_mhz: current_frequency_ghz + detuning_mhz / 1e3,
        ),
    )
    detuning_axis.set_xlabel("Detuning from current resonance [MHz]")
    return detuning_axis


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plot the raw I and Q qubit-spectroscopy responses on separate subplots.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - Each qubit occupies two rows containing separate I and Q subplots.
    - The fitted qubit frequency is marked on both subplots.
    """
    fig, axes = plt.subplots(
        2 * len(qubits),
        1,
        figsize=(10, max(7, 7 * len(qubits))),
        squeeze=False,
    )

    for qubit_index, qubit in enumerate(qubits):
        selected = ds.sel(qubit=qubit.name).assign_coords(
            full_freq_GHz=ds.full_freq.sel(qubit=qubit.name) / u.GHz
        )
        fit = fits.sel(qubit=qubit.name)
        fitted_frequency_ghz = float(fit.res_freq.values) / u.GHz
        current_frequency_ghz = float(qubit.xy.RF_frequency) / u.GHz

        qubit_axes = axes[2 * qubit_index : 2 * qubit_index + 2, 0]
        for ax, quadrature, color in zip(qubit_axes, ("I", "Q"), ("tab:blue", "tab:orange")):
            (selected[quadrature] / u.mV).plot(ax=ax, x="full_freq_GHz", color=color)
            ax.axvline(
                current_frequency_ghz,
                color="black",
                linestyle=":",
                label=f"Current resonance: {current_frequency_ghz:.6f} GHz",
            )
            ax.axvline(
                fitted_frequency_ghz,
                color="tab:red",
                linestyle="--",
                label=f"New resonance: {fitted_frequency_ghz:.6f} GHz",
            )
            ax.set_title(f"{qubit.name}: {quadrature}")
            ax.set_xlabel("RF frequency [GHz]")
            ax.set_ylabel(f"{quadrature} [mV]")
            ax.legend()
            ax.grid(alpha=0.25)
            _add_detuning_axis(ax, current_frequency_ghz)

    fig.suptitle("Qubit spectroscopy: I and Q quadratures")
    fig.tight_layout()
    return fig
