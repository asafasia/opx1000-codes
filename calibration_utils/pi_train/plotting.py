from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from utils.plotting_settings import FIGURE_SIZE


def plot_pi_train(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    use_state_discrimination: bool,
    operation: str = "x180",
) -> Figure:
    """Plot the response after each number of consecutive selected gates."""
    variables = ("state",) if use_state_discrimination else ("I", "Q")
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Pi-train plot expected {variables}, but dataset contains {list(ds.data_vars)}"
        )

    figure, axes = plt.subplots(
        len(qubits) * len(variables),
        1,
        figsize=FIGURE_SIZE,
        squeeze=False,
    )
    axis_index = 0
    for qubit in qubits:
        selected = ds.sel(qubit=qubit.name)
        for variable in variables:
            axis = axes[axis_index, 0]
            selected[variable].plot(
                ax=axis,
                x="number_of_pulses",
                marker="o",
                markersize=8,
                label="Measured",
            )
            if variable == "state":
                rotation_angle = np.pi if operation == "x180" else np.pi / 2
                ideal_population = np.sin(ds.number_of_pulses.values * rotation_angle / 2) ** 2
                axis.plot(
                    ds.number_of_pulses.values,
                    ideal_population,
                    "k--",
                    alpha=0.6,
                    label="Ideal",
                )
                axis.set_ylim(-0.05, 1.05)
                axis.legend()
            label = "Excited-state population" if variable == "state" else f"{variable} [V]"
            axis.set_title(f"{qubit.name}: {label}")
            axis.set_xlabel(f"Number of consecutive {operation} gates")
            axis.set_ylabel(label)
            axis.set_xticks(ds.number_of_pulses.values)
            axis.grid(alpha=0.3)
            axis_index += 1

    figure.suptitle(f"Gate train: {operation}")
    figure.tight_layout()
    return figure
