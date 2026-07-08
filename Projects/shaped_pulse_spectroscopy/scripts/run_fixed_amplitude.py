"""Run one shaped-pulse fixed-amplitude detuning spectroscopy experiment."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

runpy.run_module("experiments.fixed_amplitude_spectroscopy", run_name="__main__")
