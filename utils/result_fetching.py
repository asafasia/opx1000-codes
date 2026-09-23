"""Fetch calibration results with live progress and Ctrl+C recovery."""

import re
import time

import numpy as np
import xarray as xr
from qualibration_libs.data import XarrayDataFetcher


class AcquisitionError(RuntimeError):
    """A job failed or produced invalid measurement results."""


class AcquisitionStopped(RuntimeError):
    """The user stopped acquisition before any complete measurements existed."""


def _values(value):
    """Unwrap SDK save/save_all results without squeezing length-one axes."""
    if isinstance(value, dict):
        value = value["value"]
    value = np.asarray(value)
    if value.dtype.names and "value" in value.dtype.names:
        value = value["value"]
    return value


def _counter_value(handle):
    """Read an available scalar counter without waiting for measurement buffers."""
    if handle is None or handle.count_so_far() < 1:
        return None
    value = handle.fetch_all()
    if value is None:
        return None
    value = _values(value)
    return int(value.reshape(-1)[-1]) if value.size else None


def _read_dataset(handles, axes, *, job_id, interrupted, partial_axis):
    names = [name for name in handles.keys()
             if name != "n" and name not in XarrayDataFetcher.ignore_handles]
    if not names:
        if interrupted:
            raise AcquisitionStopped("Stopped before a complete sweep was available; no results to plot.")
        raise AcquisitionError(f"Job {job_id} produced no measurement result streams.")
    missing = [name for name in names if handles.get(name).count_so_far() < 1]
    if missing:
        if interrupted:
            raise AcquisitionStopped("Stopped before a complete sweep was available; no results to plot.")
        raise AcquisitionError(f"Job {job_id} finished without results for: {', '.join(missing)}.")
    lost = [name for name in names if handles.get(name).has_dataloss()]
    if lost:
        raise AcquisitionError(f"Job {job_id} reported data loss in: {', '.join(lost)}.")

    # Fetch directly: the upstream live fetcher waits for all requested handles
    # in its constructor and cannot adapt its axes to an interrupted run.
    raw = {name: handles.get(name).fetch_all(flat_struct=True) for name in names}
    if any(value is None for value in raw.values()):
        raise AcquisitionError(f"Job {job_id} has unavailable measurement results.")
    raw = {name: _values(value) for name, value in raw.items()}
    coords = {name: xr.DataArray(np.asarray(axis), dims=name,
                                attrs=dict(getattr(axis, "attrs", {})))
              for name, axis in axes.items()}
    target_axis = next(iter(coords))
    if target_axis not in {"qubit", "qubit_pair"}:
        raise AcquisitionError("The first result axis must be qubit or qubit_pair.")
    dims = [name for name in coords if name != target_axis]
    shape = tuple(coords[name].size for name in dims)
    acquired = None
    if interrupted and partial_axis is not None:
        if not dims or dims[0] != partial_axis:
            raise AcquisitionError("Partial results require the outermost retained axis.")
        for name, value in raw.items():
            if value.ndim != len(shape) or value.shape[1:] != shape[1:] or value.shape[0] > shape[0]:
                raise AcquisitionError(f"Unexpected result shape for {name}: {value.shape}, expected (*, {shape[1:]}).")
        # Streams may finish a different number of iterations when stopped.
        # Keep their common complete prefix; never pad missing shots with zeroes.
        acquired = min(value.shape[0] for value in raw.values())
        if acquired == 0:
            raise AcquisitionStopped("Stopped before a complete sweep was available; no results to plot.")
        coords[partial_axis] = coords[partial_axis][:acquired]
        raw = {name: value[:acquired] for name, value in raw.items()}
        shape = (acquired, *shape[1:])
    for name, value in raw.items():
        if value.shape != shape:
            raise AcquisitionError(f"Unexpected result shape for {name}: {value.shape}, expected {shape}.")

    grouped = {}
    variables = {}
    for name, value in raw.items():
        match = re.fullmatch(r"([a-zA-Z_]+)(\d+)", name)
        if match:
            label, index = match.groups()
            grouped.setdefault(label, {})[int(index)] = value
        else:
            variables[name] = (dims, value)
    target_count = coords[target_axis].size
    for label, values in grouped.items():
        if set(values) != set(range(1, target_count + 1)):
            raise AcquisitionError(f"Missing qubit results for {label}.")
        variables[label] = ([target_axis, *dims], np.stack([values[i] for i in range(1, target_count + 1)]))
    dataset = xr.Dataset(variables, coords=coords)
    if interrupted:
        dataset.attrs["acquisition_interrupted"] = True
        if acquired is not None:
            dataset.attrs.update(completed_iterations=acquired, partial_axis=partial_axis)
    return dataset


def fetch_result_dataset(job, axes, *, on_progress=None, poll_interval=0.2, partial_axis=None):
    """Fetch results, or stop on Ctrl+C and recover completed sweep buffers.

    Genuine job failures and external cancellations still raise AcquisitionError.
    A second Ctrl+C during recovery propagates to the session cleanup.
    """
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    if partial_axis is None:
        first = next((name for name in axes if name not in {"qubit", "qubit_pair"}), None)
        if first in {"shot", "n_runs"}:
            partial_axis = first
    started = time.time()
    if on_progress is not None:
        on_progress(0, started)
    job_id = getattr(job, "id", "unknown")
    get_status = getattr(job, "get_status", None)
    interrupted = False
    try:
        handles = job.result_handles
        counter = handles.get("n")
        last_counter = 0
        while True:
            if callable(get_status):
                status = get_status()
                if status in {"Canceled", "Error"}:
                    raise AcquisitionError(f"Job {job_id} ended with status {status}; no complete dataset was fetched.")
                if status not in {"In queue", "Running", "Processing", "Done"}:
                    raise AcquisitionError(f"Job {job_id} returned unexpected status {status!r}.")
                done = status == "Done"
            else:
                done = not handles.is_processing()
            count = _counter_value(counter)
            if on_progress is not None and count is not None and count != last_counter:
                on_progress(count, started)
                last_counter = count
            if done:
                break
            time.sleep(poll_interval)
        if handles.wait_for_all_values() is False:
            raise AcquisitionError(f"Job {job_id} closed result streams before completion.")
        return _read_dataset(handles, axes, job_id=job_id, interrupted=False, partial_axis=partial_axis)
    except KeyboardInterrupt:
        print("\nStopping acquisition; fetching completed measurements...", flush=True)
        interrupted = True
        status = get_status() if callable(get_status) else None
        if status == "Error":
            raise AcquisitionError(f"Job {job_id} ended with status Error.") from None
        if status not in {"Done", "Canceled"}:
            cancel = getattr(job, "cancel", None) or getattr(job, "halt", None)
            if not callable(cancel):
                raise AcquisitionError(f"Job {job_id} cannot be stopped by this SDK.") from None
            cancel()
        handles = job.result_handles
        # False is expected for a user-canceled job; existing full buffers remain
        # readable. Bound this wait, and allow a second Ctrl+C to abort recovery.
        handles.wait_for_all_values(timeout=10)
        if callable(get_status) and get_status() == "Error":
            raise AcquisitionError(f"Job {job_id} ended with status Error.") from None
    return _read_dataset(handles, axes, job_id=job_id, interrupted=interrupted, partial_axis=partial_axis)
