"""Class-based v2 migration for end_work."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from qualibrate import NodeParameters
from quam_config import Quam
from quam_config import create_machine

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration
else:
    from .base import BaseCalibration

description = """
        CLOSE ALL OTHER QMs.
"""



# %% {Close_all_quantum_machines}

class EndWork(BaseCalibration[NodeParameters, Quam]):
    """v2 class migration for ``calibrations/end_work.py``."""

    def __init__(
        self,
        parameters: NodeParameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="end_work",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    def create_qua_program(self):
        return None

    def run(self):
        self.run_calibration()
        return None
    def run_calibration(self):
        node = self
        """Closes all the opened quantum machines."""
        qmm = node.machine.connect()
        qmm.close_all_qms()


if __name__ == "__main__":
    parameters = NodeParameters()

    calibration = EndWork(
        parameters=parameters,
        machine=create_machine(),
    )
    calibration.run()
