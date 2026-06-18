"""Class-based v2 migration for 08b_readout_power_optimization."""

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
from profiles import ProfileUpdater
from utils.plotting_settings import plot_per_qubit
from calibration_utils.readout_power_optimization import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.iq_blobs.plotting import plot_iq_blobs, plot_confusion_matrices
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration, CalibrationOptions
else:
    from .base import BaseCalibration, CalibrationOptions

description = """
        READOUT POWER OPTIMIZATION
The sequence consists in measuring the state of the resonator after thermalization (qubit in |g>) and after
playing a pi pulse to the qubit (qubit in |e>) successively while sweeping the readout amplitude.
The 'I' & 'Q' quadratures when the qubit is in |g> and |e> are extracted to derive the readout fidelity.
The optimal readout amplitude is chosen as to maximize the readout fidelity.

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).

State update:
    - The readout amplitude: qubit.resonator.operations["readout"].amplitude
    - The integration weight angle: qubit.resonator.operations["readout"].integration_weights_angle
    - the ge discrimination threshold: qubit.resonator.operations["readout"].threshold
    - the Repeat Until Success threshold: qubit.resonator.operations["readout"].rus_exit_threshold
    - The confusion matrix: qubit.resonator.operations["readout"].confusion_matrix
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
# %% {Save_results}


class ReadoutPowerOptimization(BaseCalibration[Parameters, Quam]):
    """v2 class migration for ``calibrations/08b_readout_power_optimization.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="08b_readout_power_optimization",
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

        n_runs = node.parameters.num_shots  # Number of runs
        amps = np.linspace(
            node.parameters.start_amp, node.parameters.end_amp, node.parameters.num_amps
        )
        # Register the sweep axes to be added to the dataset when fetching data
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "n_runs": xr.DataArray(
                np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}
            ),
            "amp_prefactor": xr.DataArray(
                amps, attrs={"long_name": "readout amplitude", "units": ""}
            ),
        }
        with program() as node.namespace["qua_program"]:
            Ig, Ig_st, Qg, Qg_st, n, n_st = node.machine.declare_qua_variables()
            Ie, Ie_st, Qe, Qe_st, _, _ = node.machine.declare_qua_variables()
            a = declare(fixed)

            for multiplexed_qubits in qubits.batch():
                # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_runs, n + 1):
                    # ground iq blobs for all qubits
                    save(n, n_st)
                    with for_(*from_array(a, amps)):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                # log_callable=node.log,
                            )
                        align()
                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            # Measure the state of the resonators
                            qubit.resonator.measure(
                                "readout", qua_vars=(Ig[i], Qg[i]), amplitude_scale=a
                            )
                            # save data to their respective streams
                            save(Ig[i], Ig_st[i])
                            save(Qg[i], Qg_st[i])

                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                # log_callable=node.log,
                            )
                        align()
                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            # Play the x180 gate to put the qubits in the excited state
                            qubit.xy.play("x180")
                            # Align the elements to measure after playing the qubit pulses.
                            qubit.align()
                            # Measure the state of the resonators
                            qubit.resonator.measure(
                                "readout", qua_vars=(Ie[i], Qe[i]), amplitude_scale=a
                            )
                            # save data to their respective streams
                            save(Ie[i], Ie_st[i])
                            save(Qe[i], Qe_st[i])

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    Ig_st[i].buffer(len(amps)).buffer(n_runs).save(f"Ig{i + 1}")
                    Qg_st[i].buffer(len(amps)).buffer(n_runs).save(f"Qg{i + 1}")
                    Ie_st[i].buffer(len(amps)).buffer(n_runs).save(f"Ie{i + 1}")
                    Qe_st[i].buffer(len(amps)).buffer(n_runs).save(f"Qe{i + 1}")

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
        node.results["ds_fit"], node.results["ds_iq_blobs"], fit_results = fit_raw_data(
            node.results["ds_raw"], node
        )
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
        )
        figures.update(
            plot_per_qubit(
                plot_iq_blobs,
                node.results["ds_raw"],
                node.namespace["qubits"],
                node.results["ds_iq_blobs"],
                figure_name="iq_blobs",
            )
        )
        figures.update(
            plot_per_qubit(
                plot_confusion_matrices,
                node.results["ds_raw"],
                node.namespace["qubits"],
                node.results["ds_iq_blobs"],
                figure_name="confusion_matrix",
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
        """Update the relevant parameters if the qubit data analysis was successful."""
        with node.record_state_updates():
            for q in node.namespace["qubits"]:
                if node.outcomes[q.name] == "failed":
                    continue

                fit_results = node.results["fit_results"][q.name]
                operation = q.resonator.operations["readout"]
                operation.integration_weights_angle -= float(fit_results["iw_angle"])
                operation.threshold = (
                    float(fit_results["ge_threshold"]) * operation.length / 2**12
                )
                operation.rus_exit_threshold = (
                    float(fit_results["rus_threshold"]) * operation.length / 2**12
                )
                operation.amplitude = float(fit_results["optimal_amplitude"])
                q.resonator.confusion_matrix = fit_results["confusion_matrix"]

    def propose_profile_update(self):
        node = self
        """Stage best readout fidelity in profile metrics."""
        if node.parameters.reset_type not in {"active", "thermal"}:
            return

        updates = {
            f"metrics.json.qubits.{q.name}.readout.fidelity_percent.{node.parameters.reset_type}": float(
                node.results["fit_results"][q.name]["readout_fidelity"]
            )
            for q in node.namespace["qubits"]
            if node.outcomes[q.name] == "successful"
        }
        if updates:
            proposal = ProfileUpdater().stage(
                node.name, updates, profile_name=current_profile_name()
            )
            ProfileUpdater().confirm_and_apply(proposal)


if __name__ == "__main__":
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.start_amp = 0
    parameters.end_amp = 2
    parameters.num_amps = 100
    parameters.num_shots = 100

    options = CalibrationOptions()

    calibration = ReadoutPowerOptimization(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
