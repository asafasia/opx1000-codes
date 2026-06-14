"""Shared settings and annotations for calibration plots."""

from datetime import datetime


FIGURE_SIZE = (13, 8)
CALIBRATION_TIMESTAMP_GID = "calibration_timestamp"


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
