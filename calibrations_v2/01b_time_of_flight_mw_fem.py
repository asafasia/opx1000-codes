"""Class-based v2 migration for 01b_time_of_flight_mw_fem."""

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
from pathlib import Path
from dataclasses import asdict
from qm import generate_qua_script
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from quam_config import Quam, create_machine
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from calibration_utils.time_of_flight_mw import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_single_run_with_fit,
    plot_averaged_run_with_fit,
)
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates
from quam_builder.tools.power_tools import calculate_voltage_scaling_factor

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration, CalibrationOptions
else:
    from .base import BaseCalibration, CalibrationOptions

description = """
        TIME OF FLIGHT - MW FEM
This sequence involves sending a readout pulse and capturing the raw ADC traces.
The data undergoes post-processing to calibrate three distinct parameters:
    - Time of Flight: This represents the internal processing time and the propagation
      delay of the readout pulse. Its value can be adjusted in the configuration under
      "time_of_flight". This value is utilized to offset the acquisition window relative
      to when the readout pulse is dispatched.

    - Analog Inputs Gain: If a signal is constrained by digitization or if it saturates
      the ADC, the variable gain of the OPX analog input, ranging from -12 dB to 20 dB,
      can be modified to fit the signal within the ADC range of +/-0.5V.
      
Prerequisites:
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).

State update:
    - The time of flight: qubit.resonator.time_of_flight
"""




# Create the machine directly from profiles/main without loading state.json.





def select_full_scale_power_dbm(power_in_dbm: float, max_amplitude: float = 1) -> int:
    """Select the lowest valid QOP 3.2 MW-FEM full-scale power for a target power."""
    allowed_full_scale_powers = np.arange(-11, 11, 3)
    compatible_powers = [
        int(full_scale_power)
        for full_scale_power in allowed_full_scale_powers
        if calculate_voltage_scaling_factor(full_scale_power, power_in_dbm) <= max_amplitude
    ]
    if not compatible_powers:
        raise ValueError(
            f"Cannot reach {power_in_dbm} dBm with max_amplitude={max_amplitude}. "
            "Lower readout_amplitude_in_dBm or increase max_amplitude."
        )
    return compatible_powers[0]


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
# %% {QUA_program}
# %% {Simulate}
# %% {Execute}
# %% {Save_raw_results}
# %% {Data_loading_and_dataset_creation}
# %% {Data_analysis}
# %% {Plotting}
# %% {Update_state}
# %% {Save_results}

class TimeOfFlightMwFem(BaseCalibration[Parameters, Quam]):
    """v2 class migration for ``calibrations/01b_time_of_flight_mw_fem.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="01b_time_of_flight_mw_fem",
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

        node.namespace["tracked_resonators"] = []
        full_scale_power_dbm = select_full_scale_power_dbm(node.parameters.readout_amplitude_in_dBm)
        for q in qubits:
            resonator = q.resonator
            # make temporary updates before running the program and revert at the end.
            with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
                if node.parameters.time_of_flight_in_ns is not None:
                    resonator.time_of_flight = node.parameters.time_of_flight_in_ns
                resonator.operations["readout"].length = node.parameters.readout_length_in_ns
                resonator.set_output_power(
                    power_in_dbm=node.parameters.readout_amplitude_in_dBm,
                    full_scale_power_dbm=full_scale_power_dbm,
                    operation="readout",
                )
                node.namespace["tracked_resonators"].append(resonator)

        # Register the sweep axes to be added to the dataset when fetching data
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "readout_time": xr.DataArray(
                np.arange(0, node.parameters.readout_length_in_ns, 1),
                attrs={"long_name": "readout time", "units": "ns"},
            ),
        }

        with program() as node.namespace["qua_program"]:
            n = declare(int)  # QUA variable for the averaging loop
            n_st = declare_stream()
            adc_st = [declare_stream(adc_trace=True) for _ in range(num_qubits)]  # The stream to store the raw ADC trace

            for multiplexed_qubits in qubits.batch():
                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    for i, qubit in multiplexed_qubits.items():
                        # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
                        reset_if_phase(qubit.resonator.name)
                        qubit.wait(27000)  # Wait for the time of flight before sending
                        # Measure the resonator (send a readout pulse and record the raw ADC trace)
                        qubit.resonator.measure("readout", stream=adc_st[i])
                        # Wait for the resonator to deplete
                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                    align()

            with stream_processing():
                n_st.save("n")
                for i, qubit in enumerate(node.namespace["qubits"]):
                    if qubit.resonator.opx_input.port_id == 1:
                        stream = adc_st[i].input1()
                    else:
                        stream = adc_st[i].input2()
                    # Will save average:
                    stream.real().average().save(f"adcI{i + 1}")
                    stream.image().average().save(f"adcQ{i + 1}")
                    # Will save only last run:
                    stream.real().save(f"adc_single_runI{i + 1}")
                    stream.image().save(f"adc_single_runQ{i + 1}")

        debug_directory = Path(__file__).resolve().parents[1] / "debug"
        debug_directory.mkdir(exist_ok=True)
        debug_file = debug_directory / Path(__file__).name
        config = node.machine.generate_config()
        with debug_file.open("w") as source_file:
            print(generate_qua_script(node.namespace["qua_program"], config), file=source_file)
        node.log(f"Serialized QUA debug script saved to {debug_file}")


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
            plot_single_run_with_fit,
            node.results["ds_raw"],
            node.namespace["qubits"],
            node.results["ds_fit"],
            figure_name="single_run",
        )
        figures.update(
            plot_per_qubit(
                plot_averaged_run_with_fit,
                node.results["ds_raw"],
                node.namespace["qubits"],
                node.results["ds_fit"],
                figure_name="averaged_run",
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

        # Revert the change done at the beginning of the node
        for tracked_resonator in node.namespace.get("tracked_resonators", []):
            tracked_resonator.revert_changes()

        with node.record_state_updates():
            for q in node.namespace["qubits"]:
                if not node.results["fit_results"][q.name]["success"]:
                    continue

                fit_result = node.results["fit_results"][q.name]
                if node.parameters.time_of_flight_in_ns is not None:
                    q.resonator.time_of_flight = node.parameters.time_of_flight_in_ns + fit_result["tof_to_add"]
                else:
                    q.resonator.time_of_flight += fit_result["tof_to_add"]




if __name__ == "__main__":
    parameters = Parameters()

    options = CalibrationOptions()

    calibration = TimeOfFlightMwFem(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q9"),
    )
    calibration.run()
