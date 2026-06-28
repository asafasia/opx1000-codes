from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import q3_frequency_search_runner as runner


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_LOG = ROOT / "knowledge_graphs" / "q4_frequency_campaign_log.jsonl"


def _task(kind: str, **kwargs: Any) -> dict[str, Any]:
    return {"kind": kind, **kwargs}


def _q4_tasks(listed_frequency_hz: float) -> list[dict[str, Any]]:
    candidate_f01 = listed_frequency_hz
    lower_candidate = 4_280_400_000.0
    two_photon_candidate = 4_367_500_000.0
    high_candidate = 4_466_500_000.0

    return [
        _task("resonator", center_hz=listed_frequency_hz, amp=0.7, shots=120),
        _task("resonator", center_hz=listed_frequency_hz - 50e6, amp=0.7, shots=120),
        _task("resonator", center_hz=listed_frequency_hz + 50e6, amp=0.7, shots=120),
        _task("resonator", center_hz=lower_candidate, amp=0.7, shots=120),
        _task("resonator", center_hz=two_photon_candidate, amp=0.7, shots=120),
        _task("qubit", center_hz=candidate_f01, amp=0.45, span_mhz=240, shots=350),
        _task("qubit", center_hz=candidate_f01, amp=0.7, span_mhz=120, shots=350),
        _task("chevron", center_hz=candidate_f01, amp=1.2, amp_step=0.04, span_mhz=120, shots=180),
        _task("chevron", center_hz=candidate_f01, amp=1.6, amp_step=0.05, span_mhz=80, shots=220),
        _task("chevron", center_hz=high_candidate, amp=1.2, amp_step=0.04, span_mhz=90, shots=180),
        _task("chevron", center_hz=lower_candidate, amp=1.0, amp_step=0.05, span_mhz=90, shots=160),
        _task("chevron", center_hz=two_photon_candidate, amp=1.0, amp_step=0.05, span_mhz=70, shots=160),
        _task("rabi", center_hz=candidate_f01, amp=1.2, amp_step=0.02, shots=300),
        _task("rabi", center_hz=high_candidate, amp=1.2, amp_step=0.02, shots=300),
        _task("resonator", center_hz=candidate_f01, amp=1.0, shots=120),
    ]


def _run_task(qubit_name: str, task: dict[str, Any]) -> dict[str, Any]:
    kind = task["kind"]
    if kind == "resonator":
        return runner.run_resonator(
            qubit_name,
            task["center_hz"],
            task["amp"],
            task["shots"],
        )
    if kind == "qubit":
        return runner.run_qubit(
            qubit_name,
            task["center_hz"],
            task["amp"],
            task["span_mhz"],
            task["shots"],
        )
    if kind == "rabi":
        return runner.run_power_rabi(
            qubit_name,
            task["center_hz"],
            task["amp"],
            task["amp_step"],
            task["shots"],
        )
    if kind == "chevron":
        return runner.run_power_rabi_chevron(
            qubit_name,
            task["center_hz"],
            task["amp"],
            task["amp_step"],
            task["span_mhz"],
            task["shots"],
        )
    raise ValueError(f"Unknown task kind {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubit", default="q4")
    parser.add_argument("--max-runs", type=int, default=14)
    args = parser.parse_args()

    listed_frequency_hz = runner._listed_frequency_hz(args.qubit)
    tasks = _q4_tasks(listed_frequency_hz)[: args.max_runs]
    CAMPAIGN_LOG.parent.mkdir(parents=True, exist_ok=True)

    for index, task in enumerate(tasks, start=1):
        print(f"\n=== q4 campaign run {index}/{len(tasks)}: {task} ===", flush=True)
        campaign_record: dict[str, Any] = {
            "campaign": "q4_frequency_discovery_chevron",
            "index": index,
            "max_runs": len(tasks),
            "task": task,
        }
        try:
            result = _run_task(args.qubit, task)
            campaign_record["result"] = result
            campaign_record["status"] = "completed"
        except Exception as exc:
            campaign_record["status"] = "failed"
            campaign_record["error"] = repr(exc)
            campaign_record["traceback"] = traceback.format_exc()
            print(campaign_record["traceback"], flush=True)

        with CAMPAIGN_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(campaign_record, sort_keys=True) + "\n")

    print(f"Campaign log: {CAMPAIGN_LOG}", flush=True)


if __name__ == "__main__":
    main()
