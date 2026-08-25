"""QuAM component for the host-controlled Arduino DC-bias source."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import field
from typing import ClassVar, Iterator

from quam.core import QuamComponent, quam_dataclass

from hardware.arduino_dc_bias import (
    DEFAULT_BAUD_RATE,
    DEFAULT_CHANNEL_COUNT,
    DEFAULT_PORT,
    MAX_ABS_VOLTAGE_V,
    DCBiasController,
    open_controller,
)


@quam_dataclass
class ArduinoDCBias(QuamComponent):
    """Serializable settings and host-side control for the DC-bias source.

    The live serial connection is intentionally excluded from QuAM state. This
    component has no QUA-config representation and must only be used from host
    code, outside QUA program construction.
    """

    output_channel: ClassVar[int] = 0

    port: str = DEFAULT_PORT
    baud_rate: int = DEFAULT_BAUD_RATE
    channel_count: int = DEFAULT_CHANNEL_COUNT
    max_abs_voltage_v: float = MAX_ABS_VOLTAGE_V
    qubit_biases_v: dict[str, float] = field(default_factory=dict)
    _controller: DCBiasController | None = field(
        default=None,
        init=False,
        repr=False,
        metadata={"skip_save": True},
    )

    @property
    def is_connected(self) -> bool:
        return self._controller is not None

    def connect(self) -> DCBiasController:
        if self._controller is None:
            self._controller = open_controller(
                self.port,
                self.baud_rate,
                channel_count=self.channel_count,
                max_abs_voltage_v=self.max_abs_voltage_v,
            )
        return self._controller

    def _validate_setting(self, channel: int, voltage_v: float) -> None:
        if not 0 <= channel < self.channel_count:
            raise ValueError(
                f"Channel must be between 0 and {self.channel_count - 1}; got {channel}."
            )
        if not math.isfinite(voltage_v):
            raise ValueError(f"Voltage must be finite; got {voltage_v}.")
        if abs(voltage_v) > self.max_abs_voltage_v:
            raise ValueError(
                f"Voltage magnitude must not exceed {self.max_abs_voltage_v:g} V; "
                f"got {voltage_v:g} V."
            )

    def set_voltage(self, channel: int, voltage_v: float) -> None:
        self._validate_setting(channel, voltage_v)
        self.connect().set_voltage(channel, voltage_v)

    def voltage_for_qubit(self, qubit_name: str) -> float:
        try:
            return self.qubit_biases_v[qubit_name]
        except KeyError as exc:
            raise KeyError(
                f"No DC-bias setting is configured for qubit {qubit_name!r}."
            ) from exc

    def set_for_qubit(self, qubit_name: str) -> None:
        self.set_voltage(self.output_channel, self.voltage_for_qubit(qubit_name))

    def zero_all(self) -> None:
        self.connect().zero_all()

    def disconnect(self) -> None:
        if self._controller is not None:
            try:
                self._controller.close()
            finally:
                self._controller = None

    @contextmanager
    def applied(self, channel: int, voltage_v: float) -> Iterator[None]:
        """Apply a bias and reliably return that channel to zero on exit."""
        self._validate_setting(channel, voltage_v)
        was_connected = self.is_connected
        controller = self.connect()
        try:
            controller.set_voltage(channel, voltage_v)
            yield
        finally:
            try:
                controller.set_voltage(channel, 0.0)
            finally:
                if not was_connected:
                    self.disconnect()

    @contextmanager
    def applied_for_qubit(self, qubit_name: str) -> Iterator[None]:
        """Apply the selected qubit's profile bias and zero it on exit."""
        voltage_v = self.voltage_for_qubit(qubit_name)
        with self.applied(self.output_channel, voltage_v):
            yield
