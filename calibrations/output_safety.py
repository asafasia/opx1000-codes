"""Persistent operator latch for inhibiting real OPX outputs.

This is a software safety layer, not a replacement for a physical RF/DC
interlock. The state lives outside device profiles so engaging it never
rewrites calibrated hardware parameters.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INHIBIT_PATH = REPOSITORY_ROOT / ".lab_state" / "outputs_inhibited.json"
INHIBIT_PATH_ENV = "OPX_OUTPUT_INHIBIT_FILE"


class OutputsInhibitedError(RuntimeError):
    """Raised when code attempts real hardware execution while inhibited."""


def inhibit_path() -> Path:
    override = os.environ.get(INHIBIT_PATH_ENV)
    return Path(override).expanduser() if override else DEFAULT_INHIBIT_PATH


def output_inhibit_status() -> dict[str, Any] | None:
    path = inhibit_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutputsInhibitedError(
            f"Cannot verify OPX output safety state at {path}; refusing real execution."
        ) from exc
    if not isinstance(payload, dict) or payload.get("outputs_inhibited") is not True:
        raise OutputsInhibitedError(
            f"Invalid OPX output safety state at {path}; refusing real execution."
        )
    return payload


def outputs_are_inhibited() -> bool:
    return output_inhibit_status() is not None


def assert_outputs_allowed() -> None:
    status = output_inhibit_status()
    if status is None:
        return
    reason = status.get("reason") or "no reason recorded"
    engaged_at = status.get("engaged_at") or "unknown time"
    raise OutputsInhibitedError(
        "Real OPX execution is inhibited "
        f"(engaged {engaged_at}; reason: {reason}). "
        "After the refrigerator is cold and the hardware is verified safe, run "
        "`python -m calibrations.runner outputs enable --confirm-fridge-cold`."
    )


def engage_output_inhibit(reason: str) -> Path:
    path = inhibit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "outputs_inhibited": True,
        "reason": reason.strip() or "dilution refrigerator heating",
        "engaged_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)
    return path


def clear_output_inhibit() -> bool:
    path = inhibit_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True
