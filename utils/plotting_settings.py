"""Shared settings and annotations for calibration plots."""

from datetime import datetime
from typing import Any, Callable, Iterable


FIGURE_SIZE = (13, 8)
CALIBRATION_TIMESTAMP_GID = "calibration_timestamp"


def plot_per_qubit(
    plotter: Callable[..., Any],
    dataset: Any,
    qubits: Iterable[Any],
    *args: Any,
    figure_name: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create one independent figure per qubit with a stable save name."""
    figures = {}
    for qubit in qubits:
        selected_dataset = _select_qubit(dataset, qubit.name)
        selected_args = tuple(_select_qubit(value, qubit.name) for value in args)
        selected_kwargs = {
            name: _select_qubit(value, qubit.name)
            for name, value in kwargs.items()
        }
        figures[f"{figure_name}_{qubit.name}"] = plotter(
            selected_dataset,
            [qubit],
            *selected_args,
            **selected_kwargs,
        )
    return figures


def _select_qubit(value: Any, qubit_name: str) -> Any:
    """Keep the qubit dimension while selecting one qubit from xarray-like data."""
    dims = getattr(value, "dims", ())
    if "qubit" not in dims:
        return value
    return value.sel(qubit=[qubit_name])


def qubit_grid_locations(qubits: Iterable[Any]) -> list[str]:
    """Use a compact location when plotting one qubit, otherwise preserve the device grid."""
    qubits = list(qubits)
    if len(qubits) == 1:
        return ["0,0"]
    return [qubit.grid_location for qubit in qubits]


def add_calibration_timestamp(figure, timestamp: datetime | str | None = None):
    """Add a duplicate-safe date/time footer to a Matplotlib figure."""
    for text in figure.texts:
        if text.get_gid() == CALIBRATION_TIMESTAMP_GID:
            return figure

    if timestamp is None:
        timestamp = datetime.now().astimezone()
    if isinstance(timestamp, datetime):
        timestamp = timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    text = figure.text(
        0.995,
        0.005,
        f"Calibration run: {timestamp}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    text.set_gid(CALIBRATION_TIMESTAMP_GID)
    return figure
