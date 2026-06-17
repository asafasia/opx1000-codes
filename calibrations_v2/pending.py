"""Migration placeholder for calibrations that still need a full v2 port."""

from __future__ import annotations

from typing import Any

from qualibrate import NodeParameters
from quam_config import Quam

from .base import BaseCalibration, CalibrationError


class PendingCalibration(BaseCalibration[NodeParameters, Quam]):
    """Base for same-named v2 modules that are not fully ported yet."""

    legacy_file: str = ""

    def __init__(
        self,
        *,
        name: str,
        parameters: NodeParameters | None = None,
        machine: Quam | None = None,
        description: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            parameters=parameters or NodeParameters(),
            machine=machine,
            description=description,
            **kwargs,
        )

    def create_qua_program(self) -> Any:
        raise CalibrationError(
            f"{self.name} has a v2 module, but its QUA body has not been ported yet. "
            f"Use calibrations/{self.legacy_file} until this file is fully migrated."
        )
