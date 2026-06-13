from typing import List
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import FIGURE_SIZE

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


def _transition_frequency(qubit, transition: str) -> float:
    if transition == "ef":
        if qubit.f_12 is not None:
            return float(qubit.f_12)
        return float(qubit.f_01 - qubit.anharmonicity)
    return float(qubit.xy.RF_frequency)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset,
    use_state_discrimination: bool = False,
    transition: str = "ge",
):
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
    variables = ("state",) if use_state_discrimination else ("I", "Q")
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Qubit-spectroscopy plot expected {variables} for "
            f"use_state_discrimination={use_state_discrimination}, but dataset contains {list(ds.data_vars)}"
        )

    fig, axes = plt.subplots(
        len(variables) * len(qubits),
        1,
        figsize=FIGURE_SIZE,
        squeeze=False,
    )

    for qubit_index, qubit in enumerate(qubits):
        selected = ds.sel(qubit=qubit.name).assign_coords(
            full_freq_GHz=ds.full_freq.sel(qubit=qubit.name) / u.GHz
        )
        fit = fits.sel(qubit=qubit.name)
        fitted_frequency_ghz = float(fit.res_freq.values) / u.GHz
        current_frequency_ghz = _transition_frequency(qubit, transition) / u.GHz
        current_ge_frequency_ghz = float(qubit.f_01) / u.GHz if transition == "ef" else None
        sweep_limits = (
            float(selected.full_freq_GHz.min()),
            float(selected.full_freq_GHz.max()),
        )

        start = len(variables) * qubit_index
        qubit_axes = axes[start : start + len(variables), 0]
        for ax, variable, color in zip(qubit_axes, variables, ("tab:blue", "tab:orange")):
            scale = 1 if variable == "state" else 1 / u.mV
            label = "Measured state" if variable == "state" else f"{variable} [mV]"
            (selected[variable] * scale).plot(ax=ax, x="full_freq_GHz", color=color)
            ax.axvline(
                current_frequency_ghz,
                color="black",
                linestyle=":",
                label=f"Current {transition}: {current_frequency_ghz:.6f} GHz",
            )
            if current_ge_frequency_ghz is not None:
                ax.axvline(
                    current_ge_frequency_ghz,
                    color="tab:purple",
                    linestyle=":",
                    label=f"Current ge: {current_ge_frequency_ghz:.6f} GHz",
                )
            ax.axvline(
                fitted_frequency_ghz,
                color="tab:red",
                linestyle="--",
                label=f"New resonance: {fitted_frequency_ghz:.6f} GHz",
            )
            ax.set_xlim(*sweep_limits)
            ax.set_title(f"{qubit.name}: {label}")
            ax.set_xlabel("RF frequency [GHz]")
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(alpha=0.25)
            _add_detuning_axis(ax, current_frequency_ghz)

    fig.suptitle(
        "Qubit spectroscopy: measured state"
        if use_state_discrimination
        else "Qubit spectroscopy: I and Q quadratures"
    )
    fig.tight_layout()
    return fig
