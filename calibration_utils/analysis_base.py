"""Shared analysis contract for calibration utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert common analysis values into JSON-safe built-in objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return {
            str(key): to_jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def dataset_summary(dataset: Any | None) -> dict[str, Any] | None:
    """Return compact metadata for an xarray-like dataset without raw arrays."""
    if dataset is None:
        return None
    coords = getattr(dataset, "coords", {})
    data_vars = getattr(dataset, "data_vars", {})
    sizes = getattr(dataset, "sizes", {})
    attrs = getattr(dataset, "attrs", {})
    return {
        "dims": {str(name): int(size) for name, size in dict(sizes).items()},
        "coords": sorted(str(name) for name in coords.keys()),
        "data_vars": sorted(str(name) for name in data_vars.keys()),
        "attrs": to_jsonable(attrs),
    }


@dataclass
class AnalysisResult:
    """Structured result returned by calibration analysis classes.

    The datasets remain accessible in Python, while :meth:`to_dict` provides a
    compact JSON representation suitable for the calibration run directory.
    """

    ds_processed: Any | None = None
    ds_fit: Any | None = None
    fit_results: Mapping[str, Any] = field(default_factory=dict)
    outcomes: Mapping[str, str] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_results": to_jsonable(self.fit_results),
            "outcomes": to_jsonable(self.outcomes),
            "summary": to_jsonable(self.summary),
            "datasets": {
                "processed": dataset_summary(self.ds_processed),
                "fit": dataset_summary(self.ds_fit),
            },
        }


class BaseAnalysis(ABC):
    """Base class for calibration analysis utilities."""

    def __init__(self, node: Any) -> None:
        self.node = node

    def run(self, ds: Any) -> AnalysisResult:
        """Process, fit, log, and package a raw calibration dataset."""
        ds_processed = self.process(ds)
        ds_fit, fit_results = self.fit(ds_processed)
        plain_fit_results = self.to_plain_fit_results(fit_results)
        result = AnalysisResult(
            ds_processed=ds_processed,
            ds_fit=ds_fit,
            fit_results=plain_fit_results,
            outcomes=self.outcomes(plain_fit_results),
            summary=self.summary(ds_fit, plain_fit_results),
        )
        self.log(result)
        return result

    def process(self, ds: Any) -> Any:
        return ds

    @abstractmethod
    def fit(self, ds: Any) -> tuple[Any, Mapping[str, Any]]:
        """Return fitted dataset and per-qubit fit results."""

    def to_plain_fit_results(self, fit_results: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(qubit_name): to_jsonable(fit_result)
            for qubit_name, fit_result in fit_results.items()
        }

    def outcomes(self, fit_results: Mapping[str, Any]) -> dict[str, str]:
        outcomes = {}
        for qubit_name, fit_result in fit_results.items():
            success = bool(fit_result.get("success", False)) if isinstance(fit_result, Mapping) else False
            outcomes[str(qubit_name)] = "successful" if success else "failed"
        return outcomes

    def summary(self, ds_fit: Any, fit_results: Mapping[str, Any]) -> dict[str, Any]:
        successes = sum(
            1
            for fit_result in fit_results.values()
            if isinstance(fit_result, Mapping) and bool(fit_result.get("success", False))
        )
        return {
            "num_results": len(fit_results),
            "num_successful": successes,
            "num_failed": len(fit_results) - successes,
        }

    def log(self, result: AnalysisResult) -> None:
        """Optional logging hook for concrete analyses."""
