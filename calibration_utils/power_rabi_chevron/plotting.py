from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from utils.plotting_settings import FIGURE_SIZE


u = unit(coerce_to_integer=True)


def plot_raw_data(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    use_state_discrimination: bool = False,
):
    """Plot the frequency-versus-amplitude Rabi chevron."""
    variables = ("state",) if use_state_discrimination else ("I", "Q")
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Power-Rabi-chevron plot expected {missing!r} for "
            f"use_state_discrimination={use_state_discrimination}, "
            f"but dataset contains {list(ds.data_vars)}"
        )

    variables = ("state",) if use_state_discrimination else ("I", "Q")
    figure, axes = plt.subplots(
        len(variables) * len(qubits),
        1,
        squeeze=False,
        figsize=FIGURE_SIZE,
    )

    for qubit_index, qubit in enumerate(qubits):
        for variable_index, variable in enumerate(variables):
            plot_individual_data_with(
                axes[len(variables) * qubit_index + variable_index, 0],
                ds,
                {"qubit": qubit.name},
                use_state_discrimination=use_state_discrimination,
                variable=variable,
            )

    figure.suptitle(
        "Power Rabi chevron: measured state"
        if use_state_discrimination
        else "Power Rabi chevron: I and Q quadratures"
    )
    figure.tight_layout()
    return figure


def plot_individual_data_with(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    use_state_discrimination: bool = False,
    variable: str | None = None,
):
    """Plot one power-Rabi chevron panel with the same axes style as 04a."""
    data = variable or ("state" if use_state_discrimination else "I")
    expected_variables = ("state",) if use_state_discrimination else ("I", "Q")
    if data not in expected_variables:
        raise ValueError(
            f"Power-Rabi-chevron variable {data!r} is incompatible with "
            f"use_state_discrimination={use_state_discrimination}"
        )
    if data not in ds:
        raise RuntimeError(
            f"Power-Rabi-chevron plot expected {data!r} for "
            f"use_state_discrimination={use_state_discrimination}, but dataset contains {list(ds.data_vars)}"
        )

    selected = ds.sel(qubit=qubit["qubit"]).assign_coords(
        full_freq_GHz=ds.sel(qubit=qubit["qubit"]).full_freq / u.GHz,
        full_amp_mV=ds.sel(qubit=qubit["qubit"]).full_amp / u.mV,
        rabi_frequency_MHz=ds.sel(qubit=qubit["qubit"]).rabi_frequency_hz / u.MHz,
    )
    scale = 1 if data == "state" else 1 / u.mV
    data_label = "Measured state" if data == "state" else f"{data} [mV]"

    plotted = (selected[data] * scale).plot(
        ax=ax,
        x="full_freq_GHz",
        y="rabi_frequency_MHz",
        add_colorbar=True,
        robust=True,
    )
    plotted.colorbar.set_label(data_label)
    ax.set_title(f"{qubit['qubit']}: {data_label}")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel("Rabi frequency [MHz]")
    _add_amplitude_yaxis(ax, ds, qubit["qubit"])

    ax2 = ax.twiny()
    (selected[data] * scale).assign_coords(
        detuning_MHz=selected.detuning / u.MHz
    ).plot(
        ax=ax2,
        x="detuning_MHz",
        y="rabi_frequency_MHz",
        add_colorbar=False,
        robust=True,
    )
    ax2.set_title("")
    ax2.set_xlabel("Detuning [MHz]")


def _add_amplitude_yaxis(ax: Axes, ds: xr.Dataset, qubit_name: str) -> None:
    full_amp = ds.sel(qubit=qubit_name).full_amp
    rabi_frequency_hz = ds.sel(qubit=qubit_name).rabi_frequency_hz
    finite = (
        np.isfinite(np.asarray(full_amp.values, dtype=float))
        & np.isfinite(np.asarray(rabi_frequency_hz.values, dtype=float))
        & (np.asarray(rabi_frequency_hz.values, dtype=float) != 0)
    )
    if not np.any(finite):
        return

    amp_per_hz = float(
        np.mean(
            np.asarray(full_amp.values, dtype=float)[finite]
            / np.asarray(rabi_frequency_hz.values, dtype=float)[finite]
        )
    )

    def rabi_mhz_to_amp_mv(rabi_mhz):
        return np.asarray(rabi_mhz) * u.MHz * amp_per_hz / u.mV

    def amp_mv_to_rabi_mhz(amp_mv):
        return np.asarray(amp_mv) * u.mV / amp_per_hz / u.MHz

    right_axis = ax.secondary_yaxis(
        "right",
        functions=(rabi_mhz_to_amp_mv, amp_mv_to_rabi_mhz),
    )
    right_axis.set_ylabel("Pulse amplitude [mV]")
