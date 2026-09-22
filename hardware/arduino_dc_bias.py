"""Host-side driver for the Arduino multi-channel DC-bias source."""

from __future__ import annotations

import math
import re
import time
from typing import Protocol

DEFAULT_PORT = "COM7"
DEFAULT_BAUD_RATE = 115_200
DEFAULT_CHANNEL_COUNT = 8


def _validate_controller_limits(channel_count: int, max_abs_voltage_v: float) -> None:
    if channel_count <= 0:
        raise ValueError("channel_count must be positive.")
    if (
        isinstance(max_abs_voltage_v, bool)
        or not math.isfinite(max_abs_voltage_v)
        or max_abs_voltage_v <= 0
    ):
        raise ValueError("max_abs_voltage_v must be finite and positive.")


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
        max_abs_voltage_v: float,
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
            print(f"DC: requesting channel {channel} = {voltage:g} V", flush=True)

        command = f"SET,{channel},{voltage:.12g}\r"
        payload = command.encode("ascii")
        for attempt in range(2):
            written = self.connection.write(payload)
            if written != len(payload):
                raise IOError(
                    f"Incomplete DC-bias command: wrote {written}/{len(payload)} bytes."
                )
            time.sleep(self.response_delay_s)
            responses = self._print_responses()
            # This device answers NOP to the first command after opening the
            # port. Retry only this explicit response, once, at the same voltage.
            if responses == ["NOP"] and attempt == 0:
                print("DC: received NOP; retrying the same command once.", flush=True)
                continue
            for response in responses:
                match = re.fullmatch(r"DAC (\d+) UPDATED TO ([+-]?\d+(?:\.\d+)?)V", response)
                if match and int(match.group(1)) == channel:
                    return
            raise IOError(
                f"DC bias channel {channel} request {voltage:g} V was not acknowledged: "
                + ("; ".join(responses) or "no reply before serial timeout")
            )

    def zero_all(self) -> None:
        print("DC: setting all channels to zero")
        for channel in range(self.channel_count):
            self.set_voltage(channel, 0.0, verbose=False)

    def close(self) -> None:
        self.connection.close()

    def _print_responses(self) -> list[str]:
        # Wait using the finite serial timeout, even if no bytes have arrived
        # after the post-write delay. A reply is not measured voltage readback.
        response = self.connection.readline()
        if not response:
            print(
                "DC WARNING: no Arduino reply before the serial timeout; "
                "the requested voltage is unconfirmed.",
                flush=True,
            )
            return []
        responses = []
        while response:
            decoded = response.decode("utf-8", errors="replace").strip()
            if decoded:
                responses.append(decoded)
                print(f"Arduino responded: {decoded}", flush=True)
            if not self.connection.in_waiting:
                break
            response = self.connection.readline()
        return responses


def open_controller(
    port: str = DEFAULT_PORT,
    baud_rate: int = DEFAULT_BAUD_RATE,
    *,
    channel_count: int = DEFAULT_CHANNEL_COUNT,
    max_abs_voltage_v: float,
    startup_delay_s: float = 2.0,
) -> DCBiasController:
    """Open the source with the limit supplied by the selected profile."""
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
