"""Class-based v2 migration for 06a_ramsey."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from quam_config import Quam, create_machine
from calibration_utils.ramsey import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_io import CalibrationSaver, current_profile_name
from profiles import ProfileUpdater
from utils.plotting_settings import plot_per_qubit

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration, CalibrationOptions
else:
    from .base import BaseCalibration, CalibrationOptions

description = """
        RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90/y90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing gates on resonance as opposed to the detuned Ramsey.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).
    - (optional) Having optimized the readout parameters (nodes 08a, 08b and 08c).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The qubit 0->1 frequency: qubit.f_01 & qubit.xy.RF_frequency
    - T2*: qubit.T2ramsey.
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


class Ramsey(BaseCalibration[Parameters, Quam]):
    """v2 class migration for ``calibrations/06a_ramsey.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="06a_ramsey",
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

        n_avg = node.parameters.num_shots

        idle_times = get_idle_times_in_clock_cycles(node.parameters)
        detuning = node.parameters.frequency_detuning_in_mhz * u.MHz

        detuning_signs = [-1, 1]
        # Register the sweep axes to be added to the dataset when fetching data
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "idle_time": xr.DataArray(
                4 * idle_times, attrs={"long_name": "idle times", "units": "ns"}
            ),
            "detuning_signs": xr.DataArray(
                detuning_signs, attrs={"long_name": "detuning signs"}
            ),
        }
        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            idle_time = declare(int)
            detuning_sign = declare(int)
            virtual_detuning_phases = [declare(fixed) for _ in range(num_qubits)]

            if node.parameters.use_state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [declare_stream() for _ in range(num_qubits)]

            for multiplexed_qubits in qubits.batch():
                # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)

                    with for_each_(idle_time, idle_times):
                        with for_(*from_array(detuning_sign, detuning_signs)):
                            # Qubit initialization
                            for i, qubit in multiplexed_qubits.items():
                                reset_frame(qubit.xy.name)
                                qubit.reset(
                                    node.parameters.reset_type,
                                    node.parameters.simulate,
                                    # log_callable=node.log,
                                )
                            align()
                            # Qubit manipulation
                            for i, qubit in multiplexed_qubits.items():
                                with if_(detuning_sign == 1):
                                    assign(
                                        virtual_detuning_phases[i],
                                        Cast.mul_fixed_by_int(
                                            detuning * 1e-9, 4 * idle_time
                                        ),
                                    )
                                with else_():
                                    assign(
                                        virtual_detuning_phases[i],
                                        Cast.mul_fixed_by_int(
                                            -detuning * 1e-9, 4 * idle_time
                                        ),
                                    )

                                with strict_timing_():
                                    qubit.xy.play("x90")
                                    qubit.xy.frame_rotation_2pi(
                                        virtual_detuning_phases[i]
                                    )
                                    qubit.xy.wait(idle_time)
                                    qubit.xy.play("x90")

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
                        state_st[i].buffer(len(detuning_signs)).buffer(
                            len(idle_times)
                        ).average().save(f"state{i + 1}")
                    else:
                        I_st[i].buffer(len(detuning_signs)).buffer(
                            len(idle_times)
                        ).average().save(f"I{i + 1}")
                        Q_st[i].buffer(len(detuning_signs)).buffer(
                            len(idle_times)
                        ).average().save(f"Q{i + 1}")

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
                if node.results["fit_results"][q.name]["success"]:
                    q.f_01 -= float(node.results["fit_results"][q.name]["freq_offset"])
                    q.xy.RF_frequency -= float(
                        node.results["fit_results"][q.name]["freq_offset"]
                    )
                    q.T2ramsey = float(node.results["fit_results"][q.name]["decay"])

    def propose_profile_update(self):
        node = self
        """Stage fitted Ramsey T2 values in profile metrics."""
        updates = {
            f"metrics.json.qubits.{q.name}.coherence.t2_ramsey_ns": float(
                node.results["fit_results"][q.name]["decay"]
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

    parameters.num_shots = 2000
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.max_wait_time_in_ns = 15e3
    parameters.wait_time_num_points = 250
    parameters.frequency_detuning_in_mhz = 2

    options = CalibrationOptions()

    calibration = Ramsey(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q9"),
    )
    calibration.run()
