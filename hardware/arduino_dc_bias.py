"""Host-side driver for the Arduino multi-channel DC-bias source."""

from __future__ import annotations

import math
import time
from typing import Protocol


DEFAULT_PORT = "COM7"
DEFAULT_BAUD_RATE = 115_200
DEFAULT_CHANNEL_COUNT = 8
MAX_ABS_VOLTAGE_V = 0.01


def _validate_controller_limits(
    channel_count: int, max_abs_voltage_v: float
) -> None:
    if channel_count <= 0:
        raise ValueError("channel_count must be positive.")
    if (
        not math.isfinite(max_abs_voltage_v)
        or max_abs_voltage_v <= 0
        or max_abs_voltage_v > MAX_ABS_VOLTAGE_V
    ):
        raise ValueError(
            "max_abs_voltage_v must be positive and no greater than "
            f"{MAX_ABS_VOLTAGE_V:g} V."
        )


class SerialConnection(Protocol):
    in_waiting: int

    def write(self, data: bytes) -> int: ...

    def readline(self) -> bytes: ...

    def close(self) -> None: ...


class DCBiasController:
    """Wrapper around the Arduino ``SET,<channel>,<voltage>`` protocol."""

    def __init__(
        self,
        connection: SerialConnection,
        *,
        channel_count: int = DEFAULT_CHANNEL_COUNT,
        max_abs_voltage_v: float = MAX_ABS_VOLTAGE_V,
        response_delay_s: float = 0.005,
    ) -> None:
        _validate_controller_limits(channel_count, max_abs_voltage_v)

        self.connection = connection
        self.channel_count = channel_count
        self.max_abs_voltage_v = max_abs_voltage_v
        self.response_delay_s = response_delay_s

    def set_voltage(
        self, channel: int, voltage: float, *, verbose: bool = True
    ) -> None:
        if not 0 <= channel < self.channel_count:
            raise ValueError(
                f"Channel must be between 0 and {self.channel_count - 1}; got {channel}."
            )
        if not math.isfinite(voltage):
            raise ValueError(f"Voltage must be finite; got {voltage}.")
        if abs(voltage) > self.max_abs_voltage_v:
            raise ValueError(
                f"Voltage magnitude must not exceed {self.max_abs_voltage_v:g} V; "
                f"got {voltage:g} V."
            )

        if verbose:
            print(f"DC: setting channel {channel} to {voltage:g} V")

        command = f"SET,{channel},{voltage:.12g}\r"
        self.connection.write(command.encode("ascii"))
        time.sleep(self.response_delay_s)
        self._print_responses()

    def zero_all(self) -> None:
        print("DC: setting all channels to zero")
        for channel in range(self.channel_count):
            self.set_voltage(channel, 0.0, verbose=False)

    def close(self) -> None:
        self.connection.close()

    def _print_responses(self) -> None:
        while self.connection.in_waiting:
            response = (
                self.connection.readline().decode("utf-8", errors="replace").strip()
            )
            if response:
                print(f"Arduino responded: {response}")


def open_controller(
    port: str = DEFAULT_PORT,
    baud_rate: int = DEFAULT_BAUD_RATE,
    *,
    channel_count: int = DEFAULT_CHANNEL_COUNT,
    max_abs_voltage_v: float = MAX_ABS_VOLTAGE_V,
    startup_delay_s: float = 2.0,
) -> DCBiasController:
    """Open the serial port and wait for the Arduino to initialize."""
    _validate_controller_limits(channel_count, max_abs_voltage_v)
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required. Install it with: python -m pip install pyserial"
        ) from exc

    connection = serial.Serial(port, baud_rate, timeout=1)
    time.sleep(startup_delay_s)
    return DCBiasController(
        connection,
        channel_count=channel_count,
        max_abs_voltage_v=max_abs_voltage_v,
    )
