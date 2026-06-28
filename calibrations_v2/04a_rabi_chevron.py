"""Class-based v2 migration for 04a_rabi_chevron."""

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
from quam_config import Quam
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from calibration_utils.rabi_chevron import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates
from quam_config import create_machine

if __package__ in {None, ""}:
    from calibrations_v2.core import BaseCalibration, CalibrationOptions
else:
    from .core import BaseCalibration, CalibrationOptions

description = """
        RABI CHEVRON - DURATION VS AMPLITUDE
This sequence involves executing the qubit x180 pulse and measuring the state
of the resonator across various qubit intermediate frequencies and pulse durations.
Analyzing the results allows for determining the qubit and estimating the x180 pulse duration for a specific amplitude.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit frequency (node 03a_qubit_spectroscopy.py and/or 03b_qubit_spectroscopy_vs_flux.py).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - Manually set the x180 pulse duration qubit.xy.operation["x180"].length.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
def validate_readout_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> None:
    """Ensure fetched results match the requested readout mode."""
    variables = set(ds.data_vars)
    expected = {"state"} if use_state_discrimination else {"I", "Q"}
    unexpected = {"I", "Q"} if use_state_discrimination else {"state"}
    missing = expected - variables
    present_unexpected = unexpected & variables
    if missing or present_unexpected:
        raise RuntimeError(
            "Rabi readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(variables)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


# %% {Create_QUA_program}
# %% {Simulate}
# %% {Execute}
# %% {Save_raw_results}
# %% {Load_historical_data}
# %% {Analyse_data}
# %% {Plot_data}
# %% {Update_state}
# %% {Save_results}


class RabiChevron(BaseCalibration[Parameters, Quam]):
    """v2 class migration for ``calibrations/04a_rabi_chevron.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="04a_rabi_chevron",
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

        # Update the readout power to match the desired range, this change will be reverted at the end of the node.
        node.namespace["tracked_qubits"] = []
        for q in qubits:
            with tracked_updates(q, auto_revert=False) as q:
                q.xy.operations["x180"].length = 16
            node.namespace["tracked_qubits"].append(q)

        n_avg = node.parameters.num_shots  # The number of averages
        state_discrimination = node.parameters.use_state_discrimination
        # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
        pulse_durations = np.arange(
            node.parameters.min_wait_time_in_ns,
            node.parameters.max_wait_time_in_ns,
            node.parameters.time_step_in_ns,
        )
        # Qubit detuning sweep with respect to their resonance frequencies
        span = node.parameters.frequency_span_in_mhz * u.MHz
        step = node.parameters.frequency_step_in_mhz * u.MHz
        dfs = np.arange(-span // 2, +span // 2, step)

        # Register the sweep axes to be added to the dataset when fetching data
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "detuning": xr.DataArray(
                dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}
            ),
            "pulse_duration": xr.DataArray(
                pulse_durations,
                attrs={"long_name": "qubit pulse duration", "units": "ns"},
            ),
        }

        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            if state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [declare_stream() for _ in range(num_qubits)]
            t = declare(int)
            df = declare(int)

            for multiplexed_qubits in qubits.batch():
                # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, dfs)):
                        with for_(*from_array(t, pulse_durations // 4)):
                            # Qubit initialization
                            for i, qubit in multiplexed_qubits.items():
                                # Set the xy drive frequency back to the qubit frequency before reset.
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency
                                )
                                qubit.reset(
                                    node.parameters.reset_type,
                                    node.parameters.simulate,
                                    # log_callable=node.log,
                                )

                                # Update the xy drive frequency
                                qubit.xy.update_frequency(
                                    df + qubit.xy.intermediate_frequency
                                )
                            align()
                            # Qubit manipulation
                            for i, qubit in multiplexed_qubits.items():
                                qubit.xy.play("x180", duration=t)
                            align()

                            # Qubit readout
                            for i, qubit in multiplexed_qubits.items():
                                if state_discrimination:
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
                        state_st[i].buffer(len(pulse_durations)).buffer(
                            len(dfs)
                        ).average().save(f"state{i + 1}")
                    else:
                        I_st[i].buffer(len(pulse_durations)).buffer(
                            len(dfs)
                        ).average().save(f"I{i + 1}")
                        Q_st[i].buffer(len(pulse_durations)).buffer(
                            len(dfs)
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
        validate_readout_dataset(dataset, node.parameters.use_state_discrimination)
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
        validate_readout_dataset(
            node.results["ds_raw"], node.parameters.use_state_discrimination
        )
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
            use_state_discrimination=node.parameters.use_state_discrimination,
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

        # Revert the change done at the beginning of the node
        for tracked_qubit in node.namespace.get("tracked_qubits", []):
            tracked_qubit.revert_changes()

        with node.record_state_updates():
            for q in node.namespace["qubits"]:
                if node.outcomes[q.name] == "failed":
                    continue


if __name__ == "__main__":
    parameters = Parameters()

    parameters.use_state_discrimination = True
    parameters.reset_type = "thermal"
    parameters.use_state_discrimination = False
    parameters.num_shots = 100

    options = CalibrationOptions()

    calibration = RabiChevron(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q3"),
    )
    calibration.run()
