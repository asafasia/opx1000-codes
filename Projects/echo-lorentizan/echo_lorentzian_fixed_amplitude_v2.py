"""Class-based echo-Lorentzian spectroscopy at one selected amplitude."""

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
from qualang_tools.units import unit

from calibrations_v2.base import BaseCalibration, CalibrationOptions
from lorentzian import (
    install_lorentzian_operation,
    plot_raw_data,
    process_raw_dataset,
)
from parameters import Parameters
from quam_config import Quam, create_machine
from utils.plotting_settings import plot_per_qubit
from utils.rabi_amplitude import rabi_frequency_hz_to_amplitude

DESCRIPTION = """
        ECHO LORENTZIAN - FIXED AMPLITUDE SPECTROSCOPY
This calibration plays one selected Lorentzian-like qubit pulse amplitude and
sweeps only qubit-drive detuning. The selected amplitude may be supplied either
as a Lorentzian amplitude prefactor or as a Rabi frequency in MHz, converted
through the selected qubit's square x180 pulse calibration.
"""


def validate_readout_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> None:
    variables = set(ds.data_vars)
    expected = {"state"} if use_state_discrimination else {"I", "Q"}
    unexpected = {"I", "Q"} if use_state_discrimination else {"state"}
    missing = expected - variables
    present_unexpected = unexpected & variables
    if missing or present_unexpected:
        raise RuntimeError(
            "Echo-Lorentzian fixed-amplitude readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(variables)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


class EchoLorentzianFixedAmplitude(BaseCalibration[Parameters, Quam]):
    """Fixed-amplitude echo-Lorentzian detuning spectroscopy."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        *,
        name: str = "echo_lorentzian_fixed_amplitude",
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

    def fixed_amp_factor(self, qubits: Any) -> float:
        if self.parameters.fixed_amp_factor is not None:
            return float(self.parameters.fixed_amp_factor)
        if self.parameters.fixed_rabi_frequency_mhz is None:
            raise ValueError(
                "Set fixed_rabi_frequency_mhz or fixed_amp_factor for "
                "fixed-amplitude spectroscopy."
            )
        if len(qubits) != 1:
            raise ValueError(
                "fixed_rabi_frequency_mhz conversion currently expects one selected qubit."
            )
        qubit = list(qubits)[0]
        pi_pulse = qubit.xy.operations["x180"]
        full_amp = float(
            rabi_frequency_hz_to_amplitude(
                self.parameters.fixed_rabi_frequency_mhz * 1e6,
                float(pi_pulse.amplitude),
                float(pi_pulse.length),
            )
        )
        return full_amp / float(self.parameters.lorentzian_peak_amplitude)

    def create_qua_program(self):
        u = unit(coerce_to_integer=True)
        qubits = self.get_qubits()
        num_qubits = len(qubits)
        operation = self.parameters.operation
        amp_factor = self.fixed_amp_factor(qubits)
        if abs(amp_factor) >= 2:
            raise ValueError("QUA amplitude prefactors must stay within [-2, 2).")

        # Reuse waveform installation safety checks by exposing a one-point sweep.
        self.parameters.min_amp_factor = amp_factor
        self.parameters.max_amp_factor = amp_factor + 1e-6
        self.parameters.amp_factor_step = 1e-6
        install_lorentzian_operation(self)
        play_duration = self.namespace["lorentzian_play_duration_cycles"]
        amps = np.asarray([amp_factor], dtype=float)

        span = int(round(self.parameters.frequency_span_in_mhz * u.MHz))
        step = int(round(self.parameters.frequency_step_in_mhz * u.MHz))
        if step <= 0:
            raise ValueError("frequency_step_in_mhz must be positive.")
        dfs = np.arange(-span // 2, span // 2 + step, step, dtype=int)

        self.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "detuning": xr.DataArray(
                dfs,
                attrs={"long_name": "qubit detuning", "units": "Hz"},
            ),
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
            df = declare(int)

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    self.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < self.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, dfs)):
                        with for_(*from_array(a, amps)):
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency
                                )
                                qubit.reset(
                                    self.parameters.reset_type,
                                    self.parameters.simulate,
                                )
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency + df
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
                        state_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(
                            f"state{i + 1}"
                        )
                    else:
                        I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(
                            f"I{i + 1}"
                        )
                        Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(
                            f"Q{i + 1}"
                        )

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
        self.results["ds_raw"] = process_raw_dataset(self.results["ds_raw"], self)

    def plot_data(self) -> None:
        figures = plot_per_qubit(
            plot_raw_data,
            self.results["ds_raw"],
            self.namespace["qubits"],
            figure_name="echo_lorentzian_fixed_amplitude",
            use_state_discrimination=self.parameters.use_state_discrimination,
        )
        plt.show()
        self.results["figures"] = figures


if __name__ == "__main__":
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = False
    parameters.cutoff = 0.005
    parameters.fixed_rabi_frequency_mhz = 2.32
    parameters.num_shots = 100
    parameters.lorentzian_length_in_ns = 20000
    parameters.waveform_template_length_in_ns = 20000
    parameters.lorentzian_peak_amplitude = 0.2
    parameters.frequency_span_in_mhz = 1
    parameters.frequency_step_in_mhz = 1 / 99

    calibration = EchoLorentzianFixedAmplitude(
        parameters=parameters,
        options=CalibrationOptions(),
        machine=create_machine(qubit="q1"),
        auto_connect=True,
    )
    calibration.run()
