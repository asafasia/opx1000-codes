"""Class-based v2 migration for 12_Qubit_Spectroscopy_ef."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.qubit_spectroscopy import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam, create_machine
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from profiles import ProfileUpdater

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration, CalibrationOptions
else:
    from .base import BaseCalibration, CalibrationOptions

description = """
        QUBIT SPECTROSCOPY E TO F
This sequence involves preparing the excited state then sending a saturation pulse to the qubit around its e->f transition,
and then measuring the state of the resonator across various qubit drive frequencies.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually
from the node parameters.

The data is post-processed to determine the qubit second transition resonance frequency.

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended when using external mixers or the Octave.

Prerequisites:
    - Having calibrated single qubit gates.
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).

State update:
    - The qubit e->f frequency: qubit.f_12.
    - The qubit anharmonicity: qubit.anharmonicity.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
# %% {Create_QUA_program}
# %% {Simulate}
# %% {Execute}
# %% {Save_raw_results}
# %% {Load_data}
# %% {Analyse_data}
# %% {Plot_data}
# %% {Update_state}
# %% {Propose_profile_update}
# %% {Save_results}
# %%


class QubitSpectroscopyEf(BaseCalibration[Parameters, Quam]):
    """v2 class migration for ``calibrations/12_Qubit_Spectroscopy_ef.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="12_Qubit_Spectroscopy_ef",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    def create_qua_program(self):
        node = self
        """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
        # Class containing tools to help handle units and conversions.
        u = unit(coerce_to_integer=True)
        # Get the active qubits from the node and organize them by batches
        node.namespace["qubits"] = qubits = get_qubits(node)
        num_qubits = len(qubits)

        operation = node.parameters.operation  # The qubit operation to play
        n_avg = node.parameters.num_shots  # The number of averages
        # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
        operation_len = node.parameters.operation_len_in_ns
        # pre-factor to the value defined in the config - restricted to [-2; 2)
        operation_amp = node.parameters.operation_amplitude_factor
        # Qubit detuning sweep with respect to their resonance frequencies
        span = node.parameters.frequency_span_in_mhz * u.MHz
        step = node.parameters.frequency_step_in_mhz * u.MHz
        dfs = np.arange(-span // 2, +span // 2, step)

        # Register the sweep axes to be added to the dataset when fetching data
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "detuning": xr.DataArray(
                dfs, attrs={"long_name": "readout frequency", "units": "Hz"}
            ),
        }

        with program() as node.namespace["qua_program"]:
            # Macro to declare I, Q, n and their respective streams for a given number of qubit
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            df = declare(int)  # QUA variable for the qubit frequency

            for multiplexed_qubits in qubits.batch():
                # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, dfs)):
                        for i, qubit in multiplexed_qubits.items():
                            # Get the duration of the operation from the node parameters or the state
                            duration = (
                                operation_len
                                if operation_len is not None
                                else qubit.xy.operations[operation].length
                            )
                            # Wait for the qubit to thermalize (longer for proper |f> state thermalization)
                            # Reset the qubit frequency
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                            # Drive the qubit to the excited state
                            qubit.xy.play("x180")
                            # Update the qubit frequency to scan around the expected f_01
                            qubit.xy.update_frequency(
                                df
                                - qubit.anharmonicity
                                + qubit.xy.intermediate_frequency
                            )
                            # Play the saturation pulse
                            qubit.xy.play(
                                operation,
                                amplitude_scale=operation_amp,
                                duration=duration
                                >> 2,  # Bit shift by 2 as a fast division by 4 to convert from ns to clock cycles
                            )
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            # readout the resonator
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            # wait for the resonator to deplete
                            qubit.resonator.wait(node.machine.depletion_time * u.ns)

                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                        align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")

        return node.namespace.get("qua_program")

    def simulate_qua_program(self):
        node = self
        """Connect to the QOP and simulate the QUA program"""
        # Connect to the QOP
        qmm = node.machine.connect()
        # Get the config from the machine
        config = node.machine.generate_config()
        # Simulate the QUA program, generate the waveform report and plot the simulated samples
        samples, fig, wf_report = simulate_and_plot(
            qmm, config, node.namespace["qua_program"], node.parameters
        )
        # Store the figure, waveform report and simulated samples
        node.results["simulation"] = {
            "figure": fig,
            "wf_report": wf_report,
            "samples": samples,
        }

    def execute_qua_program(self):
        node = self
        """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
        # Connect to the QOP
        qmm = node.machine.connect()
        # Get the config from the machine
        config = node.machine.generate_config()
        # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            # The job is stored in the node namespace to be reused in the fetching_data run_action
            node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
            # Display the progress bar
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher["n"],
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )
            # Display the execution report to expose possible runtime errors
            node.log(job.execution_report())
        # Register the raw dataset
        node.results["ds_raw"] = dataset

    def save_raw_results(self):
        node = self
        """Save the acquired vectors and a snapshot of the selected profile."""
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
        """Load a previously acquired dataset."""
        load_data_id = node.parameters.load_data_id
        # Load the specified dataset
        node.load_from_id(node.parameters.load_data_id)
        node.parameters.load_data_id = load_data_id
        # Get the active qubits from the loaded node parameters
        node.namespace["qubits"] = get_qubits(node)

    def analyse_data(self):
        node = self
        """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
        node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
        node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
        node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

        # Log the relevant information extracted from the data analysis
        log_fitted_results(node.results["fit_results"], log_callable=node.log)
        node.outcomes = {
            qubit_name: ("successful" if fit_result["success"] else "failed")
            for qubit_name, fit_result in node.results["fit_results"].items()
        }

    def plot_data(self):
        node = self
        """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
        figures = plot_per_qubit(
            plot_raw_data_with_fit,
            node.results["ds_raw"],
            node.namespace["qubits"],
            node.results["ds_fit"],
            figure_name="amplitude",
            transition="ef",
            operation=node.parameters.operation,
            operation_amplitude_factor=node.parameters.operation_amplitude_factor,
            operation_len_in_ns=node.parameters.operation_len_in_ns,
        )
        plt.show()
        node.results["figures"] = figures
        if "calibration_run_directory" in node.namespace:
            figures_directory = CalibrationSaver().save_figures(
                node.namespace["calibration_run_directory"], node.results["figures"]
            )
            node.log(f"Calibration figures saved to {figures_directory}")

    def update_state(self):
        node = self
        """Update the relevant parameters if the qubit data analysis was successful."""
        with node.record_state_updates():
            for q in node.namespace["qubits"]:
                if node.outcomes[q.name] == "failed":
                    continue
                fitted_ef_frequency = float(
                    node.results["fit_results"][q.name]["frequency"]
                )
                q.f_12 = fitted_ef_frequency
                q.anharmonicity = float(q.f_01) - fitted_ef_frequency

    def propose_profile_update(self):
        node = self
        """Stage fitted e-f frequencies and anharmonicities for confirmation."""
        updates = {}
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] != "successful":
                continue
            fitted_ef_frequency = float(
                node.results["fit_results"][q.name]["frequency"]
            )
            updates[f"qubits.json.qubits.{q.name}.frequencies_hz.qubit_f12"] = (
                fitted_ef_frequency
            )
            updates[f"qubits.json.qubits.{q.name}.transmon.anharmonicity_hz"] = (
                float(q.f_01) - fitted_ef_frequency
            )

        if updates:
            proposal = ProfileUpdater().stage(
                node.name,
                updates,
                profile_name=current_profile_name(),
            )
            ProfileUpdater().confirm_and_apply(proposal)


if __name__ == "__main__":
    parameters = Parameters()

    parameters.num_shots = 100
    parameters.frequency_span_in_mhz = 200
    parameters.frequency_step_in_mhz = 2
    parameters.operation_amplitude_factor = 0.9

    options = CalibrationOptions()

    calibration = QubitSpectroscopyEf(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
