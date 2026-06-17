"""Class-based v2 migration for 08a_readout_frequency_optimization."""

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
from dataclasses import asdict
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from quam_config import Quam, create_machine
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from profiles import ProfileUpdater
from calibration_utils.readout_frequency_optimization import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_distances_with_fit,
    plot_IQ_abs_with_fit,
)
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration
else:
    from .base import BaseCalibration

description = """
        READOUT OPTIMISATION: FREQUENCY
The sequence consists in measuring the state of the resonator after thermalization (qubit in |g>) and after
playing a pi pulse to the qubit (qubit in |e>) successively while sweeping the readout frequency.
The 'I' & 'Q' quadratures when the qubit is in |g> and |e> are extracted to derive the readout fidelity.
The optimal readout frequency is chosen as to maximize the state discrimination Signal-to-Noise Ratio (SNR).

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).

State update:
    - The readout frequency: qubit.resonator.f_01 & qubit.resonator.RF_frequency
    - The dispersive shift: qubit.chi
"""




# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
# %% {Create_QUA_program}
# %% {Simulate}
# %% {Execute}
# %% {Save_raw_results}
# %% {Load_historical_data}
# %% {Analyse_data}
# %% {Plot_data}
# %% {Update_state}
# %% {Propose_profile_update}

class ReadoutFrequencyOptimization(BaseCalibration[Parameters, Quam]):
    """v2 class migration for ``calibrations/08a_readout_frequency_optimization.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="08a_readout_frequency_optimization",
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

        n_avg = node.parameters.num_shots  # The number of averages
        # The frequency sweep around the resonator resonance frequency
        span = node.parameters.frequency_span_in_mhz * u.MHz
        step = node.parameters.frequency_step_in_mhz * u.MHz
        dfs = np.arange(-span / 2, +span / 2, step)
        # Register the sweep axes to be added to the dataset when fetching data
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
        }

        with program() as node.namespace["qua_program"]:
            n = declare(int)
            I_g = [declare(fixed) for _ in range(num_qubits)]
            Q_g = [declare(fixed) for _ in range(num_qubits)]
            I_e = [declare(fixed) for _ in range(num_qubits)]
            Q_e = [declare(fixed) for _ in range(num_qubits)]
            df = declare(int)
            I_g_st = [declare_stream() for _ in range(num_qubits)]
            Q_g_st = [declare_stream() for _ in range(num_qubits)]
            I_e_st = [declare_stream() for _ in range(num_qubits)]
            Q_e_st = [declare_stream() for _ in range(num_qubits)]
            n_st = declare_stream()

            for multiplexed_qubits in qubits.batch():
                # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, dfs)):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            # Update the resonator frequencies & reset the qubits
                            update_frequency(qubit.resonator.name, df + qubit.resonator.intermediate_frequency)
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                # log_callable=node.log,
                            )

                        align()

                        # Qubit readout - |g> state
                        for i, qubit in multiplexed_qubits.items():
                            # Measure the state of the resonators
                            qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
                            # save data to their respective streams
                            save(I_g[i], I_g_st[i])
                            save(Q_g[i], Q_g_st[i])
                        align()

                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                # log_callable=node.log,
                            )
                        align()
                        # Qubit readout - |e> state
                        for i, qubit in multiplexed_qubits.items():
                            # Play the x180 gate to put the qubits in the excited state
                            qubit.xy.play("x180")
                            # Align the elements to measure after playing the qubit pulses.
                            qubit.align()
                            # Measure the state of the resonators
                            qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
                            # save data to their respective streams
                            save(I_e[i], I_e_st[i])
                            save(Q_e[i], Q_e_st[i])

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    I_g_st[i].buffer(len(dfs)).average().save(f"I_g{i + 1}")
                    Q_g_st[i].buffer(len(dfs)).average().save(f"Q_g{i + 1}")
                    I_e_st[i].buffer(len(dfs)).average().save(f"I_e{i + 1}")
                    Q_e_st[i].buffer(len(dfs)).average().save(f"Q_e{i + 1}")


        return node.namespace.get("qua_program")
    def simulate_qua_program(self):
        node = self
        """Connect to the QOP and simulate the QUA program"""
        # Connect to the QOP
        qmm = node.machine.connect()
        # Get the config from the machine
        config = node.machine.generate_config()
        # Simulate the QUA program, generate the waveform report and plot the simulated samples
        samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
        # Store the figure, waveform report and simulated samples
        node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


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
                    data_fetcher.get("n", 0),
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
            plot_distances_with_fit,
            node.results["ds_raw"],
            node.namespace["qubits"],
            node.results["ds_fit"],
            figure_name="distances",
        )
        figures.update(
            plot_per_qubit(
                plot_IQ_abs_with_fit,
                node.results["ds_raw"],
                node.namespace["qubits"],
                node.results["ds_fit"],
                figure_name="iq_abs",
            )
        )
        plt.show()
        node.results["figures"] = figures
        if "calibration_run_directory" in node.namespace:
            figures_directory = CalibrationSaver().save_figures(
                node.namespace["calibration_run_directory"],
                node.results["figures"],
            )
            node.log(f"Calibration figures saved to {figures_directory}")


    def update_state(self):
        node = self
        """Update fitted values that are not currently represented in the profile."""
        with node.record_state_updates():
            for q in node.namespace["qubits"]:
                if node.results["fit_results"][q.name]["success"]:
                    q.chi = node.results["fit_results"][q.name]["chi"]


    def propose_profile_update(self):
        node = self
        """Stage fitted readout frequencies and apply them only after confirmation."""
        updates = {
            f"qubits.json.qubits.{q.name}.frequencies_hz.resonator": float(
                node.results["fit_results"][q.name]["optimal_frequency"]
            )
            for q in node.namespace["qubits"]
            if node.results["fit_results"][q.name]["success"]
        }
        if updates:
            proposal = ProfileUpdater().stage(
                node.name,
                updates,
                profile_name=current_profile_name(),
            )
            ProfileUpdater().confirm_and_apply(proposal)


if __name__ == "__main__":
    parameters = Parameters()

    calibration = ReadoutFrequencyOptimization(
        parameters=parameters,
        machine=create_machine(qubit="q9"),
    )
    calibration.run()
