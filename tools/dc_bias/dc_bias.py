"""Apply the profile-configured DC bias to one output for one minute."""

from __future__ import annotations

import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from quam_config import create_machine

# The voltage comes from profiles/single_qubit/qubits.json.
QUBIT = "q3"
HOLD_TIME_S = 60.0


def main() -> None:
    machine = create_machine(qubit=QUBIT)
    if machine.dc_bias is None:
        raise RuntimeError("The selected profile does not define connectivity.dc_bias.")

    with machine.dc_bias.applied_for_qubit(QUBIT):
        print(f"DC: holding bias for {HOLD_TIME_S:g} s")
        time.sleep(HOLD_TIME_S)


if __name__ == "__main__":
    main()
