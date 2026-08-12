"""Control an Arduino-based multi-channel DC-bias source over serial."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterable
from typing import Protocol


DEFAULT_PORT = "COM7"
DEFAULT_BAUD_RATE = 115_200
DEFAULT_CHANNEL_COUNT = 8


class SerialConnection(Protocol):
    in_waiting: int

    def write(self, data: bytes) -> int: ...

    def readline(self) -> bytes: ...

    def close(self) -> None: ...


class DCBiasController:
    """Small wrapper around the Arduino ``SET,<channel>,<voltage>`` protocol."""

    def __init__(
        self,
        connection: SerialConnection,
        *,
        channel_count: int = DEFAULT_CHANNEL_COUNT,
        response_delay_s: float = 0.005,
    ) -> None:
        self.connection = connection
        self.channel_count = channel_count
        self.response_delay_s = response_delay_s

    def set_voltage(self, channel: int, voltage: float, *, verbose: bool = True) -> None:
        if not 0 <= channel < self.channel_count:
            raise ValueError(
                f"Channel must be between 0 and {self.channel_count - 1}; got {channel}."
            )
        if not math.isfinite(voltage):
            raise ValueError(f"Voltage must be finite; got {voltage}.")

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
            response = self.connection.readline().decode("utf-8", errors="replace").strip()
            if response:
                print(f"Arduino responded: {response}")


def open_controller(
    port: str = DEFAULT_PORT,
    baud_rate: int = DEFAULT_BAUD_RATE,
    *,
    startup_delay_s: float = 2.0,
) -> DCBiasController:
    """Open the serial port and wait for the Arduino to initialize."""
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required. Install it with: python -m pip install pyserial"
        ) from exc

    connection = serial.Serial(port, baud_rate, timeout=1)
    time.sleep(startup_delay_s)
    return DCBiasController(connection)


def _parse_setting(value: str) -> tuple[int, float]:
    try:
        channel_text, voltage_text = value.split("=", maxsplit=1)
        return int(channel_text), float(voltage_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected CHANNEL=VOLTAGE, for example 7=0.06; got {value!r}."
        ) from exc


def _validate_settings(
    settings: Iterable[tuple[int, float]], channel_count: int
) -> list[tuple[int, float]]:
    validated = list(settings)
    for channel, voltage in validated:
        if not 0 <= channel < channel_count:
            raise ValueError(
                f"Channel must be between 0 and {channel_count - 1}; got {channel}."
            )
        if not math.isfinite(voltage):
            raise ValueError(f"Voltage must be finite; got {voltage}.")
    return validated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Set one or more Arduino DC-bias channels, hold them for a fixed time, "
            "then return every channel to zero."
        )
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port (default: COM7).")
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_BAUD_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNEL_COUNT)
    parser.add_argument(
        "--set",
        dest="settings",
        action="append",
        type=_parse_setting,
        metavar="CHANNEL=VOLTAGE",
        required=True,
        help="Channel voltage in volts; repeat this option for multiple channels.",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Time to hold the requested voltages before zeroing all channels (default: 2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned sequence without opening a serial port.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.channels <= 0:
        raise SystemExit("--channels must be positive.")
    if args.hold < 0 or not math.isfinite(args.hold):
        raise SystemExit("--hold must be a finite, non-negative number.")

    try:
        settings = _validate_settings(args.settings, args.channels)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        for channel, voltage in settings:
            print(f"DRY RUN: set channel {channel} to {voltage:g} V")
        print(f"DRY RUN: hold for {args.hold:g} s")
        print(f"DRY RUN: set channels 0-{args.channels - 1} to zero")
        return

    controller = open_controller(args.port, args.baud_rate)
    try:
        for channel, voltage in settings:
            controller.set_voltage(channel, voltage)
        time.sleep(args.hold)
    finally:
        # Always attempt to leave the DC outputs in a safe state.
        try:
            controller.zero_all()
        finally:
            controller.close()


if __name__ == "__main__":
    main()
