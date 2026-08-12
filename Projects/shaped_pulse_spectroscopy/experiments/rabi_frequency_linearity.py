"""Measure the constant-drive voltage-to-Rabi-frequency relationship."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from quam.components.pulses import WaveformPulse

from calibration_utils.parameters import QubitsExperimentNodeParameters
from calibrations.base import BaseCalibration, CalibrationOptions
from shaped_pulse_spectroscopy.rabi_linearity import (
    plot_rabi_linearity,
    process_and_fit_dataset,
)
from quam_config import Quam, create_machine
from utils.plotting_settings import plot_per_qubit

DESCRIPTION = """
        CONSTANT-ENVELOPE RABI-FREQUENCY LINEARITY
This experiment sweeps the duration and absolute voltage of a constant qubit
drive. It independently fits the time-domain Rabi frequency at every voltage
and compares it with the linear conversion inferred from the square x180 pulse.
It never updates the device profile.
"""

MAX_SAFE_AMPLITUDE_V = 0.7


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    min_drive_amplitude_v: float = 0.05
    max_drive_amplitude_v: float = 0.60
    drive_amplitude_step_v: float = 0.05
    min_duration_ns: int = 16
    max_duration_ns: int = 800
    duration_step_ns: int = 4
    minimum_fit_r_squared: float = 0.80
    maximum_relative_error: float = 0.10
    operation: str = "rabi_linearity_const"


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass


def sweep_values(parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
    if parameters.drive_amplitude_step_v <= 0:
        raise ValueError("drive_amplitude_step_v must be positive.")
    amplitudes = np.arange(
        parameters.min_drive_amplitude_v,
        parameters.max_drive_amplitude_v + parameters.drive_amplitude_step_v / 2,
        parameters.drive_amplitude_step_v,
        dtype=float,
    )
    if amplitudes.size == 0 or np.any(amplitudes <= 0):
        raise ValueError("Drive amplitudes must be non-empty and positive.")
    if np.max(np.abs(amplitudes)) > MAX_SAFE_AMPLITUDE_V:
        raise ValueError(
            f"Constant drive amplitude exceeds the {MAX_SAFE_AMPLITUDE_V:g} V safety limit."
        )
    for name in ("min_duration_ns", "max_duration_ns", "duration_step_ns"):
        if int(getattr(parameters, name)) % 4:
            raise ValueError(f"{name} must be divisible by the 4 ns QUA clock.")
    durations = np.arange(
        parameters.min_duration_ns,
        parameters.max_duration_ns + parameters.duration_step_ns,
        parameters.duration_step_ns,
        dtype=int,
    )
    if durations.size < 8 or np.any(durations < 16):
        raise ValueError("Use at least eight pulse durations, all at least 16 ns.")
    return amplitudes, durations


class RabiFrequencyLinearity(BaseCalibration[Parameters, Quam]):
    """Constant-envelope duration-by-voltage Rabi experiment."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        *,
        name: str = "rabi_frequency_linearity",
        profile_name: str | None = None,
        qubit: str | None = None,
        auto_connect: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            description=DESCRIPTION,
            parameters=parameters,
            machine=machine,
            profile_name=profile_name,
            qubit=qubit,
            auto_connect=auto_connect,
            **kwargs,
        )

    def create_qua_program(self):
        qubits = self.get_qubits()
        amplitudes, durations_ns = sweep_values(self.parameters)
        base_amplitude = float(np.max(amplitudes))
        amplitude_factors = amplitudes / base_amplitude
        duration_cycles = durations_ns // 4
        for qubit in qubits:
            qubit.xy.operations[self.parameters.operation] = WaveformPulse(
                waveform_I=[base_amplitude] * 16,
                waveform_Q=[0.0] * 16,
            )

        self.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "drive_amplitude_v": xr.DataArray(
                amplitudes,
                attrs={"long_name": "constant drive amplitude", "units": "V"},
            ),
            "pulse_duration_ns": xr.DataArray(
                durations_ns,
                attrs={"long_name": "pulse duration", "units": "ns"},
            ),
        }

        with program() as qua_program:
            I, I_st, Q, Q_st, n, n_st = self.machine.declare_qua_variables()
            if self.parameters.use_state_discrimination:
                state = [declare(int) for _ in qubits]
                state_st = [declare_stream() for _ in qubits]
            a = declare(fixed)
            t = declare(int)

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    self.machine.initialize_qpu(target=qubit)
                align()
                with for_(n, 0, n < self.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(a, amplitude_factors)):
                        with for_(*from_array(t, duration_cycles)):
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency
                                )
                                qubit.reset(
                                    self.parameters.reset_type, self.parameters.simulate
                                )
                            align()
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play(
                                    self.parameters.operation,
                                    amplitude_scale=a,
                                    duration=t,
                                )
                            align()
                            for index, qubit in multiplexed_qubits.items():
                                if self.parameters.use_state_discrimination:
                                    qubit.readout_state(state[index])
                                    save(state[index], state_st[index])
                                else:
                                    qubit.resonator.measure(
                                        "readout", qua_vars=(I[index], Q[index])
                                    )
                                    save(I[index], I_st[index])
                                    save(Q[index], Q_st[index])
                            align()

            with stream_processing():
                n_st.save("n")
                for index in range(len(qubits)):
                    if self.parameters.use_state_discrimination:
                        state_st[index].buffer(len(durations_ns)).buffer(
                            len(amplitudes)
                        ).average().save(f"state{index + 1}")
                    else:
                        I_st[index].buffer(len(durations_ns)).buffer(
                            len(amplitudes)
                        ).average().save(f"I{index + 1}")
                        Q_st[index].buffer(len(durations_ns)).buffer(
                            len(amplitudes)
                        ).average().save(f"Q{index + 1}")

        self.namespace["qua_program"] = qua_program
        return qua_program

    def analyse_data(self) -> None:
        qubits = self.namespace.get("qubits") or self.get_qubits()
        fitted = process_and_fit_dataset(
            self.results["ds_raw"],
            qubits,
            use_state_discrimination=self.parameters.use_state_discrimination,
        )
        self.results["ds_raw"] = fitted
        self.results["fit_results"] = {}
        for qubit in qubits:
            selected = fitted.sel(qubit=qubit.name)
            good_fit = (
                np.asarray(selected.rabi_fit_r_squared)
                >= self.parameters.minimum_fit_r_squared
            )
            errors = np.abs(np.asarray(selected.rabi_relative_error)[good_fit])
            max_error = float(np.max(errors)) if errors.size else np.nan
            success = bool(
                errors.size >= 3 and max_error <= self.parameters.maximum_relative_error
            )
            self.results["fit_results"][qubit.name] = {
                "success": success,
                "maximum_relative_error": max_error,
                "valid_amplitude_points": int(errors.size),
            }
            self.outcomes[qubit.name] = "successful" if success else "failed"

    def plot_data(self) -> None:
        figures = plot_per_qubit(
            plot_rabi_linearity,
            self.results["ds_raw"],
            self.namespace["qubits"],
            figure_name="rabi_frequency_linearity",
        )
        plt.show()
        self.results["figures"] = figures

    def update_state(self) -> None:
        """This diagnostic deliberately never changes the device profile."""


if __name__ == "__main__":
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.use_readout_mitigation = True
    parameters.reset_type = "active"
    parameters.simulate = False

    options = CalibrationOptions(
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
    )
    calibration = RabiFrequencyLinearity(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
        auto_connect=not parameters.simulate,
    )
    calibration.run()
