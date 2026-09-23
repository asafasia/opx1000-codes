"""Shared acquisition layout, QUA stream reduction, and shot-level analysis."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from qualibrate.core.parameters import RunnableParameters

from utils.experiment_readout import readout_states


class StateAcquisitionParameters(RunnableParameters):
    acquisition: Literal["averaged", "single_shot"] = "averaged"
    """Average on the controller, or retain each measured state / IQ shot."""


def single_shot_acquisition(parameters):
    single_shot = getattr(parameters, "acquisition", "averaged") == "single_shot"
    if single_shot and parameters.num_shots < 1:
        raise ValueError("Single-shot acquisition requires num_shots >= 1")
    return single_shot


def annotate_acquisition(dataset, parameters):
    dataset.attrs["acquisition"] = getattr(parameters, "acquisition", "averaged")
    if "state" in dataset and "shot" in dataset.state.dims:
        dataset.state.attrs = {
            "long_name": "discriminated state label",
            "state_labels": ", ".join(
                f"{i}={label}" for i, label in enumerate(readout_states(parameters))
            ),
        }
    return dataset


def calculate_populations(dataset, parameters):
    """Preserve labels in state_shots and expose P(e) as state for existing fits.

    Detect the shot dimension so historical averaged data remains compatible.
    Calling again preserves already computed or mitigated populations.
    """
    if "state" not in dataset or "shot" not in dataset.state.dims:
        return dataset
    labels = dataset.state
    states = list(dataset.attrs.get("readout_states", readout_states(parameters)))
    if states not in (["g", "e"], ["g", "e", "f"]):
        raise ValueError("Invalid readout_states for single-shot populations")
    if labels.sizes["shot"] == 0 or not np.isin(labels.values, range(len(states))).all():
        raise ValueError("Single-shot data must contain valid integer state labels")
    populations = {
        f"population_{label}": (labels == i).mean("shot")
        for i, label in enumerate(states)
    }
    for label in states:
        populations[f"population_{label}"].attrs = {
            "long_name": f"{label}-state population", "population_state": label,
        }
    return dataset.assign(
        state_shots=labels.copy(deep=True),
        state=populations["population_e"].copy(deep=True),
        **populations,
    )


def prepare_acquisition_dataset(dataset, parameters):
    """Keep raw shots alongside the populations / mean IQ expected by fits.

    Only the shared ``shot`` dimension is reduced. IQ-cloud calibrations keep
    their existing ``n_runs`` dimension, which their distribution fits require.
    Repeated calls preserve processed and mitigated data.
    """
    dataset = calculate_populations(dataset, parameters)
    additions = {}
    for name, values in dataset.data_vars.items():
        if "shot" not in values.dims or name.endswith("_shots") or name == "state":
            continue
        if values.sizes["shot"] == 0:
            raise ValueError("Single-shot data must contain at least one shot")
        additions[f"{name}_shots"] = values.copy(deep=True)
        additions[name] = values.mean("shot", keep_attrs=True)
    return dataset.assign(additions) if additions else dataset


@dataclass(frozen=True)
class AcquisitionLayout:
    """QUA loop axes in outer-to-inner order, including exactly one shot axis.

    Every buffer wraps the already-built inner dimensions. Thus reducing axis
    zero immediately after the shot buffer works even for an interior shot loop.
    An outer shot loop uses a running average to retain live averaged results.
    """

    loops: tuple[tuple[str, int], ...]
    shot_axis: str = "shot"
    single_shot: bool = False

    def __post_init__(self):
        names = [name for name, _ in self.loops]
        if len(set(names)) != len(names) or names.count(self.shot_axis) != 1:
            raise ValueError("Acquisition loops must be unique and include one shot axis")
        if any(not isinstance(size, (int, np.integer)) or size < 1 for _, size in self.loops):
            raise ValueError("Acquisition loop lengths must be positive integers")

    def save(self, stream, name, *, save_all=False):
        from qm.qua import FUNCTIONS

        for index in range(len(self.loops) - 1, -1, -1):
            axis, size = self.loops[index]
            if index == 0 and not save_all and (self.single_shot or axis != self.shot_axis):
                # Retain each completed outer iteration so Ctrl+C can recover it.
                # save_all supplies this dimension without waiting for a full run.
                stream.save_all(name)
                return
            if axis == self.shot_axis and not self.single_shot:
                if index == 0 and not save_all:
                    stream = stream.average()
                else:
                    stream = stream.buffer(size).map(FUNCTIONS.average(0))
            else:
                stream = stream.buffer(size)
        if save_all:
            stream.save_all(name)
        else:
            stream.save(name)
