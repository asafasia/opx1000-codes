"""SQLite storage for calibration provenance, outcomes, and fit metrics."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = REPOSITORY_ROOT / "data" / "calibration_results.sqlite"


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    calibration_name TEXT NOT NULL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    profile_name TEXT,
    profile_hash TEXT,
    selected_qubit TEXT,
    git_commit TEXT,
    raw_data_path TEXT,
    figures_path TEXT,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    outcomes_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_calibration_time
    ON experiment_runs(calibration_name, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_qubit_time
    ON experiment_runs(selected_qubit, started_at DESC);

CREATE TABLE IF NOT EXISTS run_targets (
    run_id INTEGER NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
    target_type TEXT NOT NULL,
    target_name TEXT NOT NULL,
    PRIMARY KEY (run_id, target_type, target_name)
);

CREATE INDEX IF NOT EXISTS idx_targets_name ON run_targets(target_name);

CREATE TABLE IF NOT EXISTS calibration_metrics (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
    target_name TEXT,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    uncertainty REAL,
    unit TEXT,
    fit_quality REAL,
    accepted INTEGER NOT NULL DEFAULT 0 CHECK (accepted IN (0, 1)),
    UNIQUE(run_id, target_name, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_metrics_lookup
    ON calibration_metrics(target_name, metric_name, id DESC);

CREATE TABLE IF NOT EXISTS profile_updates (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
    field_path TEXT NOT NULL,
    previous_value_json TEXT,
    proposed_value_json TEXT NOT NULL,
    unit TEXT,
    state TEXT NOT NULL CHECK (state IN ('proposed', 'applied', 'rejected')),
    applied_at TEXT,
    UNIQUE(run_id, field_path)
);
"""


def _json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _profile_hash(profile_name: str | None) -> str | None:
    """Return a stable digest of the executable profile files used by a run."""
    if not profile_name:
        return None
    profile_directory = REPOSITORY_ROOT / "profiles" / profile_name
    if not profile_directory.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(item for item in profile_directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(profile_directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class CalibrationResultsDatabase:
    """A local SQLite registry. Each public write is atomic."""

    def __init__(self, path: Path | str = DEFAULT_DATABASE_PATH) -> None:
        self.path = Path(path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the database and its schema if needed."""
        with self._connection() as connection:
            connection.executescript(SCHEMA)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(experiment_runs)")}
            if "profile_hash" not in columns:
                connection.execute("ALTER TABLE experiment_runs ADD COLUMN profile_hash TEXT")
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (1, ?)",
                (datetime.now(timezone.utc).isoformat(),),
            )

    def record_run(
        self,
        *,
        calibration_name: str,
        status: str,
        mode: str,
        profile_name: str | None = None,
        selected_qubit: str | None = None,
        raw_data_path: Path | str | None = None,
        figures_path: Path | str | None = None,
        parameters: Mapping[str, Any] | None = None,
        outcomes: Mapping[str, str] | None = None,
        targets: Sequence[tuple[str, str]] = (),
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        notes: str | None = None,
    ) -> int:
        """Register a run and return its database id."""
        self.initialize()
        now = datetime.now(timezone.utc)
        started_at = started_at or now
        completed_at = completed_at or now
        normalized_targets = set(targets)
        if selected_qubit:
            normalized_targets.add(("qubit", selected_qubit))
        for name in (outcomes or {}):
            normalized_targets.add(("qubit", str(name)))

        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO experiment_runs(
                    started_at, completed_at, calibration_name, status, mode,
                    profile_name, profile_hash, selected_qubit, git_commit, raw_data_path,
                    figures_path, parameters_json, outcomes_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    started_at.astimezone(timezone.utc).isoformat(),
                    completed_at.astimezone(timezone.utc).isoformat(),
                    calibration_name, status, mode, profile_name, _profile_hash(profile_name), selected_qubit,
                    _git_commit(), str(raw_data_path) if raw_data_path else None,
                    str(figures_path) if figures_path else None,
                    _json(parameters or {}), _json(outcomes or {}), notes,
                ),
            )
            run_id = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO run_targets(run_id, target_type, target_name) VALUES (?, ?, ?)",
                [(run_id, kind, name) for kind, name in sorted(normalized_targets)],
            )
        return run_id

    def record_metric(
        self, run_id: int, *, metric_name: str, value: float,
        target_name: str | None = None, uncertainty: float | None = None,
        unit: str | None = None, fit_quality: float | None = None,
        accepted: bool = False,
    ) -> None:
        """Store one fitted/calibrated scalar; raw arrays belong in run files."""
        self.initialize()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO calibration_metrics(
                    run_id, target_name, metric_name, value, uncertainty, unit, fit_quality, accepted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, target_name, metric_name) DO UPDATE SET
                    value=excluded.value, uncertainty=excluded.uncertainty, unit=excluded.unit,
                    fit_quality=excluded.fit_quality, accepted=excluded.accepted""",
                (run_id, target_name, metric_name, value, uncertainty, unit, fit_quality, int(accepted)),
            )

    def record_profile_update(
        self, run_id: int, *, field_path: str, proposed_value: Any,
        previous_value: Any = None, unit: str | None = None, state: str = "proposed",
    ) -> None:
        if state not in {"proposed", "applied", "rejected"}:
            raise ValueError("state must be proposed, applied, or rejected")
        self.initialize()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO profile_updates(
                    run_id, field_path, previous_value_json, proposed_value_json, unit, state, applied_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, field_path) DO UPDATE SET
                    proposed_value_json=excluded.proposed_value_json, unit=excluded.unit,
                    state=excluded.state, applied_at=excluded.applied_at""",
                (run_id, field_path, _json(previous_value), _json(proposed_value), unit, state,
                 datetime.now(timezone.utc).isoformat() if state == "applied" else None),
            )

    def latest_metrics(self, target_name: str, metric_name: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return a metric history, newest first, for plotting or drift checks."""
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT r.id AS run_id, r.completed_at, r.calibration_name, m.value,
                          m.uncertainty, m.unit, m.fit_quality, m.accepted
                   FROM calibration_metrics AS m JOIN experiment_runs AS r ON r.id = m.run_id
                   WHERE m.target_name = ? AND m.metric_name = ?
                   ORDER BY r.completed_at DESC LIMIT ?""",
                (target_name, metric_name, limit),
            ).fetchall()
        return [dict(row) for row in rows]
