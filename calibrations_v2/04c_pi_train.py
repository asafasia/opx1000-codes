"""Class-based v2 migration for 04c_pi_train."""

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
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from calibration_utils.pi_train import Parameters, plot_pi_train, process_raw_dataset
from quam_config import Quam, create_machine
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration, CalibrationOptions
else:
    from .base import BaseCalibration, CalibrationOptions

description = """
        PI TRAIN
For every pulse-count point, reset the qubit, apply that many consecutive
selected gates, and measure the resulting state. x180 produces an alternating
ground/excited response; x90 produces a four-gate repeating pattern.

This is a diagnostic experiment and does not update machine parameters.
"""








def validate_readout_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> None:
    """Ensure fetched results match the requested readout mode."""
    expected = {"state"} if use_state_discrimination else {"I", "Q"}
    unexpected = {"I", "Q"} if use_state_discrimination else {"state"}
    missing = expected - set(ds.data_vars)
    present_unexpected = unexpected & set(ds.data_vars)
    if missing or present_unexpected:
        raise RuntimeError(
            "Pi-train readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(ds.data_vars)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


# %% {Create_QUA_program}
# %% {Simulate}
# %% {Execute}
# %% {Load_historical_data}
# %% {Save_raw_results}
# %% {Analyse_data}
# %% {Plot_data}

class PiTrain(BaseCalibration[Parameters, Quam]):
    """v2 class migration for ``calibrations/04c_pi_train.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="04c_pi_train",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )
    def create_qua_program(self):
        node = self
        """Create the pi-train pulse-count sweep."""
        if node.parameters.max_number_of_pulses < 1:
            raise ValueError("max_number_of_pulses must be at least 1.")

        node.namespace["qubits"] = qubits = get_qubits(node)
        num_qubits = len(qubits)
        operation = node.parameters.operation
        for qubit in qubits:
            if operation not in qubit.xy.operations:
                raise ValueError(f"{qubit.name} does not define operation {operation!r}.")
        pulse_counts = np.arange(node.parameters.max_number_of_pulses + 1, dtype=int)
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "number_of_pulses": xr.DataArray(
                pulse_counts,
                attrs={"long_name": f"number of consecutive {operation} gates"},
            ),
        }

        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            pulse_count = declare(int)
            count = declare(int)
            if node.parameters.use_state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [declare_stream() for _ in range(num_qubits)]

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(pulse_count, pulse_counts)):
                        for _, qubit in multiplexed_qubits.items():
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                # log_callable=node.log,
                            )
                            # qubit.wait(10000)  # Wait for reset to complete
                        align()

                        for _, qubit in multiplexed_qubits.items():
                            with for_(count, 0, count < pulse_count, count + 1):
                                qubit.xy.play(operation)

                        align()

                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

                        align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    if node.parameters.use_state_discrimination:
                        state_st[i].buffer(len(pulse_counts)).average().save(f"state{i + 1}")
                    else:
                        I_st[i].buffer(len(pulse_counts)).average().save(f"I{i + 1}")
                        Q_st[i].buffer(len(pulse_counts)).average().save(f"Q{i + 1}")


        return node.namespace.get("qua_program")
    def simulate_qua_program(self):
        node = self
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        samples, figure, waveform_report = simulate_and_plot(
            qmm, config, node.namespace["qua_program"], node.parameters
        )
        node.results["simulation"] = {
            "figure": figure,
            "waveform_report": waveform_report,
            "samples": samples,
        }
        plt.show()


    def execute_qua_program(self):
        node = self
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program"])
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )
            node.log(job.execution_report())
        validate_readout_dataset(dataset, node.parameters.use_state_discrimination)
        node.results["ds_raw"] = dataset


    def load_data(self):
        node = self
        load_data_id = node.parameters.load_data_id
        node.load_from_id(load_data_id)
        node.parameters.load_data_id = load_data_id
        node.namespace["qubits"] = get_qubits(node)


    def save_raw_results(self):
        node = self
        output_directory = CalibrationSaver().save_xarray(
            node.name,
            node.results["ds_raw"],
            profile_name=current_profile_name(),
        )
        node.namespace["calibration_run_directory"] = output_directory
        node.log(f"Raw calibration results saved to {output_directory}")


    def analyse_data(self):
        node = self
        validate_readout_dataset(node.results["ds_raw"], node.parameters.use_state_discrimination)
        node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
        node.outcomes = {qubit.name: "successful" for qubit in node.namespace["qubits"]}


    def plot_data(self):
        node = self
        figures = plot_per_qubit(
            plot_pi_train,
            node.results["ds_raw"],
            node.namespace["qubits"],
            node.parameters.use_state_discrimination,
            node.parameters.operation,
            figure_name="pi_train",
        )
        plt.show()
        node.results["figures"] = figures
        if "calibration_run_directory" in node.namespace:
            figures_directory = CalibrationSaver().save_figures(
                node.namespace["calibration_run_directory"],
                node.results["figures"],
            )
            node.log(f"Calibration figures saved to {figures_directory}")


if __name__ == "__main__":
    parameters = Parameters()

    parameters.num_shots = 3000
    parameters.use_state_discrimination = True
    parameters.operation = "x90"
    parameters.reset_type = 'active'
    parameters.max_number_of_pulses = 150

    options = CalibrationOptions()

    calibration = PiTrain(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q9"),
    )
    calibration.run()
