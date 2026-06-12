"""Stage calibration-derived profile changes and apply them after confirmation."""

import json
import os
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "calibration_updates"
DEFAULT_PROFILES_ROOT = REPOSITORY_ROOT / "profiles"
_VALID_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def current_profile_name() -> str:
    """Return the profile selected for calibration experiments."""
    return os.environ.get("QUAM_PROFILE", "main")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _validate_name(name: str, label: str) -> str:
    if not name or not _VALID_NAME.fullmatch(name):
        raise ValueError(f"{label} must contain only letters, numbers, '.', '_' or '-'")
    return name


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def _get_nested(document: Mapping[str, Any], keys: list[str]) -> Any:
    value: Any = document
    for key in keys:
        value = value[key]
    return value


def _set_nested(document: dict[str, Any], keys: list[str], value: Any) -> None:
    target = document
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def _split_update_path(path: str) -> tuple[str, list[str]]:
    filename_stem, separator, nested_path = path.partition(".json.")
    if not separator or not filename_stem or not nested_path:
        raise ValueError(f"Update path must look like 'file.json.field.path': {path!r}")
    return f"{filename_stem}.json", nested_path.split(".")


class ProfileUpdater:
    """Stage and optionally apply updates to a selected device profile."""

    def __init__(
        self,
        output_root: Path | str = DEFAULT_OUTPUT_ROOT,
        profiles_root: Path | str = DEFAULT_PROFILES_ROOT,
    ) -> None:
        self.output_root = Path(output_root)
        self.profiles_root = Path(profiles_root)

    def stage(
        self,
        experiment_name: str,
        updates: Mapping[str, Any],
        profile_name: str | None = None,
        now: datetime | None = None,
    ) -> Path:
        """Write a pending update proposal and return its directory."""
        experiment_name = _validate_name(experiment_name, "experiment_name")
        profile_name = _validate_name(profile_name or current_profile_name(), "profile_name")
        profile_directory = self.profiles_root / profile_name
        if not profile_directory.is_dir():
            raise FileNotFoundError(f"Profile directory does not exist: {profile_directory}")
        if not updates:
            raise ValueError("At least one profile update must be provided")

        documents: dict[str, dict[str, Any]] = {}
        changes = []
        for path, new_value in updates.items():
            filename, keys = _split_update_path(path)
            if filename not in documents:
                documents[filename] = _read_json(profile_directory / filename)
            old_value = _get_nested(documents[filename], keys)
            changes.append({"path": path, "old": old_value, "new": new_value})

        timestamp = now or datetime.now().astimezone()
        proposal_directory = (
            self.output_root
            / timestamp.strftime("%Y-%m-%d")
            / experiment_name
            / timestamp.strftime("%H-%M-%S-%f")
        )
        proposal_directory.mkdir(parents=True, exist_ok=False)
        proposal = {
            "experiment_name": experiment_name,
            "profile_name": profile_name,
            "timestamp": timestamp.isoformat(),
            "status": "pending",
            "changes": changes,
        }
        _write_json(proposal_directory / "proposal.json", proposal)
        return proposal_directory

    def apply(self, proposal_directory: Path | str) -> None:
        """Apply a staged proposal and save original profile files beside it."""
        proposal_directory = Path(proposal_directory)
        proposal_path = proposal_directory / "proposal.json"
        proposal = _read_json(proposal_path)
        if proposal["status"] != "pending":
            raise ValueError(f"Proposal status is {proposal['status']!r}, expected 'pending'")

        profile_directory = self.profiles_root / proposal["profile_name"]
        backup_directory = proposal_directory / "profile_before_update"
        backup_directory.mkdir()
        documents: dict[str, dict[str, Any]] = {}
        for change in proposal["changes"]:
            filename, keys = _split_update_path(change["path"])
            source = profile_directory / filename
            if filename not in documents:
                documents[filename] = _read_json(source)
                shutil.copy2(source, backup_directory / filename)
            _set_nested(documents[filename], keys, change["new"])

        for filename, document in documents.items():
            _write_json(profile_directory / filename, document)

        proposal["status"] = "applied"
        proposal["applied_at"] = datetime.now().astimezone().isoformat()
        _write_json(proposal_path, proposal)

    def confirm_and_apply(self, proposal_directory: Path | str) -> bool:
        """Print a proposal, ask for explicit confirmation, and apply on yes."""
        proposal_directory = Path(proposal_directory)
        proposal = _read_json(proposal_directory / "proposal.json")
        print(f"\nPending profile update for {proposal['profile_name']!r}:")
        for change in proposal["changes"]:
            print(f"  {change['path']}: {change['old']!r} -> {change['new']!r}")
        try:
            response = input("Apply these calibration updates? Type 'yes' to confirm: ")
        except EOFError:
            response = ""
        if response.strip().lower() != "yes":
            print(f"Profile not updated. Proposal kept at {proposal_directory}")
            return False
        self.apply(proposal_directory)
        print(f"Profile updated. Backup and proposal saved at {proposal_directory}")
        return True
