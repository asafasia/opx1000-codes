from typing import List

import matplotlib.pyplot as plt
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
            )
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
