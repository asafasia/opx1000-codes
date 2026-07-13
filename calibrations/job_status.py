"""Read-only QOP job status helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


ACTIVE_JOB_STATUSES = ("Running", "Processing", "In queue")


@dataclass(frozen=True)
class JobStatus:
    """Small serializable view of a QOP job."""

    id: str
    status: str
    description: str = ""
    is_simulation: bool = False


@dataclass(frozen=True)
class QopStatus:
    """Read-only status report for open QMs and matching jobs."""

    open_qms: tuple[str, ...]
    jobs: tuple[JobStatus, ...]

    @property
    def has_active_jobs(self) -> bool:
        return any(job.status in ACTIVE_JOB_STATUSES for job in self.jobs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_qms": list(self.open_qms),
            "has_active_jobs": self.has_active_jobs,
            "jobs": [asdict(job) for job in self.jobs],
        }


def _job_field(job: Any, name: str, default: Any = "") -> Any:
    if isinstance(job, dict):
        return job.get(name, default)
    return getattr(job, name, default)


def summarize_jobs(jobs: Iterable[Any]) -> tuple[JobStatus, ...]:
    """Convert QOP JobData-like records into stable CLI-friendly rows."""
    return tuple(
        JobStatus(
            id=str(_job_field(job, "id", "")),
            status=str(_job_field(job, "status", "")),
            description=str(_job_field(job, "description", "") or ""),
            is_simulation=bool(_job_field(job, "is_simulation", False)),
        )
        for job in jobs
    )


def query_qop_status(
    qmm: Any,
    *,
    active_only: bool = True,
    statuses: Iterable[str] = ACTIVE_JOB_STATUSES,
) -> QopStatus:
    """Query QOP for open QMs and jobs without modifying server state."""
    open_qms = tuple(str(qm_id) for qm_id in qmm.list_open_qms())
    jobs = qmm.get_jobs(status=tuple(statuses) if active_only else ())
    return QopStatus(open_qms=open_qms, jobs=summarize_jobs(jobs))


def query_profile_qop_status(
    *,
    profile_name: str | None = None,
    qubit: str | None = None,
    active_only: bool = True,
) -> QopStatus:
    """Build an in-memory machine for the selected profile and query QOP."""
    from quam_config import create_machine

    machine = create_machine(profile_name=profile_name, qubit=qubit)
    qmm = machine.connect()
    return query_qop_status(qmm, active_only=active_only)
