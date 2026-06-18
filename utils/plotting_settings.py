"""Shared settings and annotations for calibration plots."""

from datetime import datetime
from typing import Any, Callable, Iterable


FIGURE_SIZE = (13, 8)
CALIBRATION_TIMESTAMP_GID = "calibration_timestamp"
CALIBRATION_PARAMETERS_GID = "calibration_parameters"


class CalibrationPlot:
    """Small Matplotlib wrapper for shared calibration plot annotations."""

    def __init__(self, figure):
        self.figure = figure

    def add_parameters(
        self,
        lines: Iterable[Any],
        *,
        gid: str = CALIBRATION_PARAMETERS_GID,
    ):
        """Add a qubit-spectroscopy-style parameter box."""
        text_lines = [str(line) for line in lines if line not in (None, "")]
        if not text_lines:
            return None

        for text in self.figure.texts:
            if text.get_gid() == gid:
                text.set_text("\n".join(text_lines))
                return text

        parameters = self.figure.text(
            0.01,
            0.01,
            "\n".join(text_lines),
            ha="left",
            va="bottom",
            fontsize=8,
            family="monospace",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.9,
            },
        )
        parameters.set_gid(gid)
        return parameters

    def add_timestamp(self, timestamp: datetime | str | None = None):
        return add_calibration_timestamp(self.figure, timestamp)

    def tight_layout_for_parameters(
        self,
        parameter_rows: int,
        *,
        top: float = 0.95,
        max_bottom: float = 0.25,
    ):
        bottom_margin = min(max_bottom, 0.055 + 0.018 * parameter_rows)
        self.figure.tight_layout(rect=(0, bottom_margin, 1, top))
        return self.figure


def add_calibration_parameter_box(
    figure,
    lines: Iterable[Any],
    *,
    gid: str = CALIBRATION_PARAMETERS_GID,
):
    """Add the standard calibration parameter box to a Matplotlib figure."""
    return CalibrationPlot(figure).add_parameters(lines, gid=gid)


def format_readout_parameter_lines(
    qubits: Iterable[Any],
    *,
    operation: str = "readout",
) -> list[str]:
    """Format common per-qubit readout pulse parameters."""
    lines = []
    for qubit in qubits:
        operations = getattr(getattr(qubit, "resonator", None), "operations", {})
        readout = operations.get(operation) if hasattr(operations, "get") else None
        if readout is None:
            continue
        lines.append(
            f"{qubit.name}: readout length={_format_parameter_value(getattr(readout, 'length', None), 'ns')}, "
            f"readout amp={_format_parameter_value(getattr(readout, 'amplitude', None), 'V')}"
        )
    return lines


def _format_parameter_value(value: Any, units: str) -> str:
    if value is None:
        return "unknown"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return f"{value} {units}"
    return f"{numeric_value:g} {units}"


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
