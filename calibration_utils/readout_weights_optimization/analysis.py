from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

from profiles.loader import PROFILES_ROOT


def normalize_complex_trace(trace: np.ndarray) -> np.ndarray:
    """Normalize a complex trace so its largest absolute point is one."""
    trace = np.asarray(trace, dtype=np.complex128)
    norm = np.sqrt(np.sum(np.abs(trace) ** 2))
    if not np.isfinite(norm) or norm == 0:
        return np.zeros_like(trace)
    normalized = trace / norm
    peak = np.max(np.abs(normalized))
    if not np.isfinite(peak) or peak == 0:
        return np.zeros_like(trace)
    return normalized / peak


def kernel_to_segments(weights: np.ndarray, slice_length_ns: int) -> list[list[float | int]]:
    """Convert one weight per demod slice into profile integration-weight segments."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("Profile kernel weights must be one-dimensional.")
    if slice_length_ns <= 0:
        raise ValueError("slice_length_ns must be positive.")
    return [[float(weight), int(slice_length_ns)] for weight in weights]


def build_profile_kernel(optimal_trace: np.ndarray) -> np.ndarray:
    """Return the real profile kernel supported by the current profile schema."""
    kernel = np.real(np.asarray(optimal_trace, dtype=np.complex128))
    peak = np.max(np.abs(kernel)) if kernel.size else 0
    if not np.isfinite(peak) or peak == 0:
        return np.zeros_like(kernel, dtype=float)
    return kernel / peak


def _data_array(ds: xr.Dataset, name: str) -> xr.DataArray:
    if name in ds:
        return ds[name]

    numbered = sorted(
        key for key in ds.data_vars if key.startswith(name) and key[len(name) :].isdigit()
    )
    if not numbered:
        raise KeyError(f"Dataset is missing {name!r} traces.")
    arrays = [ds[key] for key in numbered]
    qubit_coord = ds.coords.get("qubit", np.arange(len(arrays)))
    return xr.concat(
        arrays,
        dim=xr.DataArray(qubit_coord[: len(arrays)], dims="qubit", name="qubit"),
    )


def process_sliced_traces(ds: xr.Dataset, *, slice_length_ns: int) -> xr.Dataset:
    """Combine sliced demod components into ground/excited traces and kernels."""
    if slice_length_ns <= 0:
        raise ValueError("slice_length_ns must be positive.")

    ig = _data_array(ds, "IIg") + _data_array(ds, "IQg")
    qg = _data_array(ds, "QIg") + _data_array(ds, "QQg")
    ie = _data_array(ds, "IIe") + _data_array(ds, "IQe")
    qe = _data_array(ds, "QIe") + _data_array(ds, "QQe")

    trace_delta = (ie + 1j * qe) - (ig + 1j * qg)
    optimal = xr.apply_ufunc(
        normalize_complex_trace,
        trace_delta,
        input_core_dims=[["time_slice"]],
        output_core_dims=[["time_slice"]],
        vectorize=True,
        output_dtypes=[np.complex128],
    )
    profile_kernel = xr.apply_ufunc(
        build_profile_kernel,
        optimal,
        input_core_dims=[["time_slice"]],
        output_core_dims=[["time_slice"]],
        vectorize=True,
        output_dtypes=[float],
    )

    time_slice = np.arange(1, optimal.sizes["time_slice"] + 1)
    time_ns = time_slice * slice_length_ns
    return xr.Dataset(
        {
            "Ig": ig,
            "Qg": qg,
            "Ie": ie,
            "Qe": qe,
            "ground_trace": ig + 1j * qg,
            "excited_trace": ie + 1j * qe,
            "subtracted_trace": trace_delta,
            "optimal_complex_trace": optimal,
            "profile_kernel": profile_kernel,
        },
        coords={**ds.coords, "time_slice": time_slice, "time_ns": ("time_slice", time_ns)},
        attrs={"slice_length_ns": int(slice_length_ns)},
    )


def save_kernel_artifacts(
    *,
    profile_name: str,
    experiment_name: str,
    analysed: xr.Dataset,
    parameters: Mapping[str, Any] | Any,
    root: Path = PROFILES_ROOT,
    now: datetime | None = None,
) -> Path:
    """Save traces and calculated kernels under profiles/<profile>/kernels/."""
    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    output_directory = root / profile_name / "kernels"
    output_directory.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "profile_name": profile_name,
        "timestamp": timestamp,
        "slice_length_ns": int(analysed.attrs["slice_length_ns"]),
        "qubits": [str(value) for value in analysed.qubit.values],
    }
    if hasattr(parameters, "model_dump"):
        metadata["parameters"] = parameters.model_dump(mode="json")
    elif isinstance(parameters, Mapping):
        metadata["parameters"] = dict(parameters)

    for qubit in analysed.qubit.values:
        selected = analysed.sel(qubit=qubit)
        np.savez(
            output_directory / f"{qubit}_readout_kernel.npz",
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
            time_ns=selected.time_ns.values,
            Ig=selected.Ig.values,
            Qg=selected.Qg.values,
            Ie=selected.Ie.values,
            Qe=selected.Qe.values,
            ground_trace=selected.ground_trace.values,
            excited_trace=selected.excited_trace.values,
            subtracted_trace=selected.subtracted_trace.values,
            optimal_complex_trace=selected.optimal_complex_trace.values,
            profile_kernel=selected.profile_kernel.values,
        )

    return output_directory
