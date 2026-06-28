"""Class-based echo-Lorentzian amplitude sweep at zero detuning."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from calibrations_v2.base import BaseCalibration, CalibrationOptions
from lorentzian import (
    install_lorentzian_operation,
    plot_amplitude_sweep,
    process_amplitude_dataset,
)
from parameters import Parameters
from quam_config import Quam, create_machine
from utils.plotting_settings import plot_per_qubit

DESCRIPTION = """
        ECHO LORENTZIAN - AMPLITUDE AT ZERO DETUNING
This calibration plays the selected Lorentzian-like qubit pulse at the qubit
frequency and sweeps only the waveform amplitude.
"""


def validate_readout_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> None:
    variables = set(ds.data_vars)
    expected = {"state"} if use_state_discrimination else {"I", "Q"}
    unexpected = {"I", "Q"} if use_state_discrimination else {"state"}
    missing = expected - variables
    present_unexpected = unexpected & variables
    if missing or present_unexpected:
        raise RuntimeError(
            "Echo-Lorentzian amplitude readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(variables)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


class EchoLorentzianAmplitude(BaseCalibration[Parameters, Quam]):
    """Echo-Lorentzian amplitude sweep implemented with the v2 lifecycle."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        *,
        name: str = "echo_lorentzian_amplitude",
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
        num_qubits = len(qubits)
        operation = self.parameters.operation
        install_lorentzian_operation(self)
        play_duration = self.namespace["lorentzian_play_duration_cycles"]

        amps = np.arange(
            self.parameters.min_amp_factor,
            self.parameters.max_amp_factor,
            self.parameters.amp_factor_step,
        )
        if amps.size == 0:
            raise ValueError("Amplitude sweep is empty.")
        if np.any(np.abs(amps) >= 2):
            raise ValueError("QUA amplitude prefactors must stay within [-2, 2).")

        self.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "amp_prefactor": xr.DataArray(
                amps,
                attrs={"long_name": "Lorentzian amplitude prefactor"},
            ),
        }

        with program() as qua_program:
            I, I_st, Q, Q_st, n, n_st = self.machine.declare_qua_variables()
            if self.parameters.use_state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [declare_stream() for _ in range(num_qubits)]
            a = declare(fixed)

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    self.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < self.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(a, amps)):
                        for qubit in multiplexed_qubits.values():
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                            qubit.reset(
                                self.parameters.reset_type,
                                self.parameters.simulate,
                            )
                        align()

                        for qubit in multiplexed_qubits.values():
                            qubit.xy.play(
                                operation,
                                amplitude_scale=a,
                                duration=play_duration,
                            )
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            if self.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure(
                                    "readout", qua_vars=(I[i], Q[i])
                                )
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    if self.parameters.use_state_discrimination:
                        state_st[i].buffer(len(amps)).average().save(f"state{i + 1}")
                    else:
                        I_st[i].buffer(len(amps)).average().save(f"I{i + 1}")
                        Q_st[i].buffer(len(amps)).average().save(f"Q{i + 1}")

        self.namespace["qua_program"] = qua_program
        return qua_program

    def execute_qua_program(self) -> None:
        super().execute_qua_program()
        validate_readout_dataset(
            self.results["ds_raw"],
            self.parameters.use_state_discrimination,
        )

    def analyse(self) -> None:
        validate_readout_dataset(
            self.results["ds_raw"],
            self.parameters.use_state_discrimination,
        )
        self.results["ds_raw"] = process_amplitude_dataset(self.results["ds_raw"], self)

    def plot_data(self) -> None:
        figures = plot_per_qubit(
            plot_amplitude_sweep,
            self.results["ds_raw"],
            self.namespace["qubits"],
            figure_name="echo_lorentzian_amplitude",
            use_state_discrimination=self.parameters.use_state_discrimination,
        )
        plt.show()
        self.results["figures"] = figures


if __name__ == "__main__":
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = True
    parameters.cutoff = 0.01
    parameters.num_shots = 1000
    parameters.lorentzian_length_in_ns = 80000
    parameters.waveform_template_length_in_ns = 2000
    parameters.lorentzian_peak_amplitude = 0.5
    parameters.min_amp_factor = 0.0
    parameters.max_amp_factor = 1.0
    parameters.amp_factor_step = 0.005

    options = CalibrationOptions()

    calibration = EchoLorentzianAmplitude(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q9"),
        auto_connect=True,
    )
    calibration.run()
