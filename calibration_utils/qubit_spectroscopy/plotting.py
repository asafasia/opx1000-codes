from typing import List
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


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

        qubit_axes = axes[2 * qubit_index : 2 * qubit_index + 2, 0]
        for ax, quadrature, color in zip(qubit_axes, ("I", "Q"), ("tab:blue", "tab:orange")):
            (selected[quadrature] / u.mV).plot(ax=ax, x="full_freq_GHz", color=color)
            ax.axvline(
                fitted_frequency_ghz,
                color="tab:red",
                linestyle="--",
                label=f"Fit center: {fitted_frequency_ghz:.6f} GHz",
            )
            ax.set_title(f"{qubit.name}: {quadrature}")
            ax.set_xlabel("RF frequency [GHz]")
            ax.set_ylabel(f"{quadrature} [mV]")
            ax.legend()
            ax.grid(alpha=0.25)

    fig.suptitle("Qubit spectroscopy: I and Q quadratures")
    fig.tight_layout()
    return fig
