"""Class-based EF Power Rabi calibration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from calibration_utils.power_rabi import EfParameters, Parameters
from quam_config import Quam, create_machine

if __package__ in {None, ""}:
    from calibrations_v2.power_rabi import PowerRabi
else:
    from .power_rabi import PowerRabi


class PowerRabiEf(PowerRabi):
    """EF-specific Power Rabi using the shared v2 Power Rabi implementation."""

    def __init__(
        self,
        parameters: Parameters | None = None,
        machine: Quam | None = None,
        **kwargs: Any,
    ) -> None:
        parameters = parameters or EfParameters()
        parameters.transition = "ef"
        super().__init__(
            parameters=parameters,
            machine=machine,
            name="13_power_rabi_ef",
            **kwargs,
        )


if __name__ == "__main__":
    parameters = EfParameters()
    parameters.reset_type = "thermal"
    parameters.num_shots = 500
    parameters.pi_repetitions = 4

    power_rabi = PowerRabiEf(
        parameters=parameters,
        # options=options,
        machine=create_machine(qubit="q1"),
        auto_connect=True,
    )
    power_rabi.run()
