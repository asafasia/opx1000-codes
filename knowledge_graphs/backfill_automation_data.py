from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUTOMATION_DATA_ROOT = ROOT / "knowledge_graphs" / "automation_data"


def _artifact_name(record: dict[str, Any]) -> str:
    run_directory = Path(record.get("summary", {}).get("run_directory", ""))
    run_id = run_directory.name if run_directory.name else "unknown_run"
    center_hz = int(float(record["center_frequency_hz"]))
    return f"{run_id}_{record['kind']}_{center_hz}_amp_{record['amplitude_factor']}"


def _copy_figures(source_directory: Path, target_directory: Path) -> list[str]:
    try:
        sources = sorted(source_directory.iterdir()) if source_directory.exists() else []
    except OSError:
        return []

    copied: list[str] = []
    for source in sources:
        if not source.is_file():
            continue
        target_directory.mkdir(parents=True, exist_ok=True)
        target = target_directory / source.name
        try:
            shutil.copy2(source, target)
        except OSError:
            continue
        copied.append(str(target.relative_to(ROOT)))
    return copied


def publish_record(qubit_name: str, record: dict[str, Any]) -> int:
    artifact_dir = (
        AUTOMATION_DATA_ROOT
        / qubit_name
        / "frequency_discovery"
        / _artifact_name(record)
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    figure_files: list[str] = []
    run_directory_value = record.get("summary", {}).get("run_directory", "")
    if run_directory_value:
        figure_files = _copy_figures(
            Path(run_directory_value) / "figures",
            artifact_dir / "figures",
        )

    record["artifact_directory"] = str(artifact_dir.relative_to(ROOT))
    record["figure_directory"] = str((artifact_dir / "figures").relative_to(ROOT))
    record["figure_files"] = figure_files

    with (artifact_dir / "record.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return len(figure_files)


def main() -> None:
    logs = sorted((ROOT / "knowledge_graphs").glob("q*_frequency_search_log.jsonl"))
    record_count = 0
    figure_count = 0
    for log_path in logs:
        qubit_name = log_path.name.split("_frequency_search_log.jsonl")[0]
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            figure_count += publish_record(qubit_name, json.loads(line))
            record_count += 1
    print(
        f"published {record_count} records and copied {figure_count} "
        f"figure files from {len(logs)} logs"
    )


if __name__ == "__main__":
    main()
