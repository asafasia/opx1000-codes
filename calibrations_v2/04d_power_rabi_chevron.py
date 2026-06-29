"""Class-based v2 migration for 04d_power_rabi_chevron."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from calibration_utils.power_rabi_chevron import (
    Parameters,
    plot_raw_data,
    process_raw_dataset,
)
from quam_config import Quam, create_machine
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from utils.simulation import simulate_and_plot

if __package__ in {None, ""}:
    from calibrations_v2.core import BaseCalibration, CalibrationOptions
else:
    from .core import BaseCalibration, CalibrationOptions

description = """
        POWER RABI CHEVRON - FREQUENCY VS AMPLITUDE
This sequence plays a fixed-duration qubit operation while sweeping both its
amplitude and the qubit-drive frequency. It is the amplitude-sweep counterpart
of the duration-based Rabi chevron.

The experiment is intended for visual selection of a resonant frequency and a
useful pulse-amplitude range. It does not automatically update pulse parameters.
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
            "Power Rabi chevron readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(variables)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


class PowerRabiChevron(BaseCalibration[Parameters, Quam]):
    """v2 class migration for ``calibrations/04d_power_rabi_chevron.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="04d_power_rabi_chevron",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    def create_qua_program(self):
        node = self
        """Create the frequency-versus-amplitude Rabi-chevron QUA program."""
        u = unit(coerce_to_integer=True)
        node.namespace["qubits"] = qubits = get_qubits(node)
        num_qubits = len(qubits)
        operation = node.parameters.operation
        for qubit in qubits:
            if operation not in qubit.xy.operations:
                raise ValueError(
                    f"{qubit.name} does not define operation {operation!r}."
                )

        amps = np.arange(
            node.parameters.min_amp_factor,
            node.parameters.max_amp_factor,
            node.parameters.amp_factor_step,
        )
        if amps.size == 0:
            raise ValueError("Amplitude sweep is empty.")
        if np.any(np.abs(amps) >= 2):
            raise ValueError("QUA amplitude prefactors must stay within [-2, 2).")

        span = int(round(node.parameters.frequency_span_in_mhz * u.MHz))
        step = int(round(node.parameters.frequency_step_in_mhz * u.MHz))
        if step <= 0:
            raise ValueError("frequency_step_in_mhz must be positive.")
        dfs = np.arange(-span // 2, span // 2 + step, step, dtype=int)

        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "detuning": xr.DataArray(
                dfs, attrs={"long_name": "qubit detuning", "units": "Hz"}
            ),
            "amp_prefactor": xr.DataArray(
                amps,
                attrs={"long_name": "pulse amplitude prefactor"},
            ),
        }

        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            if node.parameters.use_state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [declare_stream() for _ in range(num_qubits)]
            a = declare(fixed)
            df = declare(int)

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, dfs)):
                        with for_(*from_array(a, amps)):
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency
                                )
                                qubit.reset(
                                    node.parameters.reset_type,
                                    node.parameters.simulate,
                                    # log_callable=node.log,
                                )
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency + df
                                )
                            align()

                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play(operation, amplitude_scale=a)
                            align()

                            for i, qubit in multiplexed_qubits.items():
                                if node.parameters.use_state_discrimination:
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
                    if node.parameters.use_state_discrimination:
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

        return node.namespace.get("qua_program")

    def simulate_qua_program(self):
        node = self
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        samples, figure, waveform_report = simulate_and_plot(
            qmm,
            config,
            node.namespace["qua_program"],
            node.parameters,
        )
        node.results["simulation"] = {
            "figure": figure,
            "waveform_report": waveform_report,
            "samples": samples,
        }

    def execute_qua_program(self):
        node = self
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program"])
            fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            for dataset in fetcher:
                progress_counter(
                    fetcher.get("n", 0),
                    node.parameters.num_shots,
                    start_time=fetcher.t_start,
                )
            node.log(job.execution_report())
        validate_readout_dataset(dataset, node.parameters.use_state_discrimination)
        node.results["ds_raw"] = dataset

    def save_raw_results(self):
        node = self
        output_directory = CalibrationSaver().save_xarray(
            node.name,
            node.results["ds_raw"],
            profile_name=current_profile_name(),
            parameters=node.parameters,
        )
        node.namespace["calibration_run_directory"] = output_directory
        node.log(f"Raw calibration results saved to {output_directory}")

    def load_data(self):
        node = self
        load_data_id = node.parameters.load_data_id
        node.load_from_id(load_data_id)
        node.parameters.load_data_id = load_data_id
        node.namespace["qubits"] = get_qubits(node)

    def analyse_data(self):
        node = self
        validate_readout_dataset(
            node.results["ds_raw"], node.parameters.use_state_discrimination
        )
        node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)

    def plot_data(self):
        node = self
        figures = plot_per_qubit(
            plot_raw_data,
            node.results["ds_raw"],
            node.namespace["qubits"],
            figure_name="power_rabi_chevron",
            use_state_discrimination=node.parameters.use_state_discrimination,
        )
        node.results["figures"] = figures
        if "calibration_run_directory" in node.namespace:
            figures_directory = CalibrationSaver().save_figures(
                node.namespace["calibration_run_directory"],
                node.results["figures"],
            )
            node.log(f"Calibration figures saved to {figures_directory}")
        plt.show()


if __name__ == "__main__":
    parameters = Parameters()

    parameters.operation = "saturation"
    parameters.frequency_span_in_mhz = 500
    parameters.frequency_step_in_mhz = 2
    parameters.min_amp_factor = 0
    parameters.amp_factor_step = 0.05
    parameters.max_amp_factor = 1
    parameters.num_shots = 20
    parameters.use_state_discrimination = False

    options = CalibrationOptions()

    calibration = PowerRabiChevron(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q3"),
    )
    calibration.run()
