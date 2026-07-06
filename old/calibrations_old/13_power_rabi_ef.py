"""Compatibility entry point for EF power Rabi.

The combined implementation lives in 04b_power_rabi.py. This wrapper preserves
the historical script name while selecting the EF transition.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


_previous_transition = os.environ.get("POWER_RABI_TRANSITION")
os.environ["POWER_RABI_TRANSITION"] = "ef"
try:
    _namespace = runpy.run_path(
        str(Path(__file__).with_name("04b_power_rabi.py")),
        run_name="__main__",
    )
finally:
    if _previous_transition is None:
        os.environ.pop("POWER_RABI_TRANSITION", None)
    else:
        os.environ["POWER_RABI_TRANSITION"] = _previous_transition

globals().update(_namespace)
