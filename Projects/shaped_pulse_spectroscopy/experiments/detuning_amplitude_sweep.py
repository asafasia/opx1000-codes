"""Class-based echo-Lorentzian frequency-versus-amplitude sweep."""

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
from qualang_tools.units import unit

from calibrations.base import BaseCalibration, CalibrationOptions
from shaped_pulse_spectroscopy.lorentzian import (
    amplitude_prefactors,
    install_lorentzian_operation,
    plot_raw_data,
    process_raw_dataset,
)
from shaped_pulse_spectroscopy.parameters import Parameters
from quam_config import Quam, create_machine
from utils.plotting_settings import plot_per_qubit

DESCRIPTION = """
        ECHO LORENTZIAN - FREQUENCY VS AMPLITUDE
This calibration plays a fixed-length Lorentzian-like qubit pulse while sweeping
both the qubit-drive detuning and the waveform amplitude. The pulse shape can be
the standard Lorentzian or the root-Lorentzian with tau derived from the cutoff
edge-to-peak ratio.
"""


def validate_readout_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> None:
    """Ensure fetched results match the requested readout mode."""
    variables = set(ds.data_vars)
    expected = {"state"} if use_state_discrimination else {"I", "Q"}
    unexpected = {"I", "Q"} if use_state_discrimination else {"state"}
    missing = expected - variables
    present_unexpected = unexpected & variables
    if missing or present_unexpected:
        raise RuntimeError(
            "Echo-Lorentzian readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(variables)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


class EchoLorentzian(BaseCalibration[Parameters, Quam]):
    """Echo-Lorentzian calibration implemented with the class-based calibration lifecycle."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        *,
        name: str = "echo_lorentzian",
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
        """Create the detuning-versus-amplitude Lorentzian sweep program."""
        u = unit(coerce_to_integer=True)
        qubits = self.get_qubits()
        num_qubits = len(qubits)
        operation = self.parameters.operation
        amps = amplitude_prefactors(self.parameters)
        if amps.size == 0:
            raise ValueError("Amplitude sweep is empty.")
        if np.any(np.abs(amps) >= 2):
            raise ValueError("QUA amplitude prefactors must stay within [-2, 2).")
        install_lorentzian_operation(self, amplitude_factors=amps)
        stark_chirps = self.namespace["lorentzian_stark_chirps"]
        play_duration = self.namespace["lorentzian_play_duration_cycles"]
        correction_enabled = bool(self.parameters.ac_stark_correction)

        span = int(round(self.parameters.frequency_span_in_mhz * u.MHz))
        points = getattr(self.parameters, "frequency_points", None)
        use_arbitrary_detunings = points is not None
        if points is not None:
            points = int(points)
            if points <= 0:
                raise ValueError("frequency_points must be positive.")
            dfs = np.rint(np.linspace(-span / 2, span / 2, points)).astype(int)
            if np.unique(dfs).size != dfs.size:
                raise ValueError(
                    "frequency_points creates duplicate integer detunings; "
                    "increase frequency_span_in_mhz or reduce frequency_points."
                )
        else:
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
            if correction_enabled:
                rate_index = declare(int)
                base_chirp_rates = {
                    qubit.name: declare(int, value=stark_chirps[qubit.name]["rates"])
                    for qubit in qubits
                }
                scaled_chirp_rates = {
                    qubit.name: declare(
                        int, size=len(stark_chirps[qubit.name]["rates"])
                    )
                    for qubit in qubits
                }

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    self.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < self.parameters.num_shots, n + 1):
                    save(n, n_st)
                    detuning_loop = (
                        for_each_(df, dfs.tolist())
                        if use_arbitrary_detunings
                        else for_(*from_array(df, dfs))
                    )
                    with detuning_loop:
                        with for_(*from_array(a, amps)):
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency
                                )
                                qubit.reset(
                                    self.parameters.reset_type,
                                    self.parameters.simulate,
                                )
                                if correction_enabled:
                                    chirp = stark_chirps[qubit.name]
                                    reference = chirp["reference_amplitude_factor"]
                                    scale = (
                                        0.0
                                        if reference == 0.0
                                        else a * a / reference**2
                                    )
                                    with for_(
                                        rate_index,
                                        0,
                                        rate_index < len(chirp["rates"]),
                                        rate_index + 1,
                                    ):
                                        assign(
                                            scaled_chirp_rates[qubit.name][rate_index],
                                            Cast.mul_int_by_fixed(
                                                base_chirp_rates[qubit.name][
                                                    rate_index
                                                ],
                                                scale,
                                            ),
                                        )
                                    qubit.xy.update_frequency(
                                        qubit.xy.intermediate_frequency
                                        + df
                                        + Cast.mul_int_by_fixed(
                                            chirp["initial_frequency_offset_hz"],
                                            scale,
                                        )
                                    )
                                else:
                                    qubit.xy.update_frequency(
                                        qubit.xy.intermediate_frequency + df
                                    )
                            align()

                            for qubit in multiplexed_qubits.values():
                                if correction_enabled:
                                    chirp = stark_chirps[qubit.name]
                                    qubit.xy.play(
                                        operation,
                                        amplitude_scale=a,
                                        duration=play_duration,
                                        chirp=(
                                            scaled_chirp_rates[qubit.name],
                                            chirp["times_cycles"],
                                            chirp["units"],
                                        ),
                                    )
                                else:
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
            figure_name="echo_lorentzian",
            use_state_discrimination=self.parameters.use_state_discrimination,
        )
        plt.show()
        self.results["figures"] = figures


if __name__ == "__main__":
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.use_readout_mitigation = True

    parameters.simulate = False
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = False
    parameters.ac_stark_correction = False
    parameters.stark_kappa_mhz_inv = -0.01
    parameters.stark_chirp_max_error_hz = 100.0
    parameters.cutoff = 0.999
    parameters.num_shots = 1000
    parameters.lorentzian_length_in_ns = 20000
    parameters.waveform_template_length_in_ns = 20000
    parameters.lorentzian_peak_amplitude = 0.2
    parameters.min_amp_factor = 0.0
    parameters.max_amp_factor = 1
    parameters.amp_factor_step = 0.005
    parameters.amp_factor_points = None
    parameters.amp_factor_spacing = "linear"
    parameters.frequency_span_in_mhz = 50
    parameters.frequency_step_in_mhz = 0.1
    parameters.frequency_points = None
    parameters.fit_fwhm = False

    options = CalibrationOptions()

    calibration = EchoLorentzian(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
        auto_connect=not parameters.simulate,
    )
    calibration.run()
