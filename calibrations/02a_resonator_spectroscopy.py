# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam, create_machine
from calibration_utils.resonator_spectroscopy import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_amplitude,
)
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from profiles import ProfileUpdater
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        1D RESONATOR SPECTROSCOPY
This sequence performs two separate resonator-frequency scans. The first measures the resonator while the qubit
remains in the ground state. The second measures the resonator after applying the selected qubit operation. Saturation
is continuously applied during readout, while x180_const is completed before readout. The overlaid responses expose
the dispersive shift.
The data is then post-processed to determine the resonator resonance frequency.
This frequency is used to update the readout frequency in the state.

Prerequisites:
    - Having calibrated the IQ mixer/Octave connected to the readout line (node 01a_mixer_calibration.py).
    - Having calibrated the time of flight, offsets, and gains (node 01a_time_of_flight.py).
    - Having initialized the QUAM state parameters for the readout pulse amplitude and duration, and the resonators depletion time.
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The readout frequency: qubit.resonator.f_01 & qubit.resonator.RF_frequency
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="02a_resonator_spectroscopy",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)

node.machine = create_machine(qubit='q7')

node.machine.connect()  # Connect to the machine to fetch the qubits information and populate the node namespace if needed

node.machine.qmm.close_all_qms()


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q9"]
    node.parameters.qubit_operation = 'saturation'
    node.parameters.num_shots= 600
    node.parameters.frequency_span_in_mhz = 30
    node.parameters.frequency_step_in_mhz = 0.3
    pass


    

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Extract the sweep parameters and axes from the node parameters
    n_runs = node.parameters.num_shots
    selected_operation = node.parameters.qubit_operation
    qua_operation = "x180" if selected_operation == "x180_const" else selected_operation
    # The frequency sweep around the resonator resonance frequency
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)
    for qubit in qubits:
        if qua_operation not in qubit.xy.operations:
            raise ValueError(f"{qubit.name} does not define qubit operation {qua_operation!r}.")
        if selected_operation == "saturation":
            saturation_length = qubit.xy.operations["saturation"].length
            readout_length = qubit.resonator.operations["readout"].length
            required_length = node.parameters.saturation_lead_time_in_ns + readout_length
            if saturation_length < required_length:
                raise ValueError(
                    f"{qubit.name} saturation pulse is {saturation_length} ns, but at least "
                    f"{required_length} ns is required to cover the lead-in and readout."
                )
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.arange(n_runs), attrs={"long_name": "shot index"}),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        Ig, Ig_st, Qg, Qg_st, n, n_st = node.machine.declare_qua_variables()
        Im, Im_st, Qm, Qm_st, _, _ = node.machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the readout frequency

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)

                # Complete ground-state resonator spectroscopy scan.
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        rr = qubit.resonator
                        # Update the resonator frequencies for all resonators
                        rr.update_frequency(df + rr.intermediate_frequency)
                        # Measure the resonator
                        rr.measure("readout", qua_vars=(Ig[i], Qg[i]))
                        # wait for the resonator to deplete
                        # rr.wait(rr.depletion_time * u.ns)

                        save(Ig[i], Ig_st[i])
                        save(Qg[i], Qg_st[i])
                        qubit.wait(25000)

                    align()

                # Complete the driven-state resonator spectroscopy scan.
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        rr = qubit.resonator
                        rr.update_frequency(df + rr.intermediate_frequency)
                        if selected_operation == "saturation":
                            align(qubit.xy.name, rr.name)
                            qubit.xy.play(
                                qua_operation,
                                amplitude_scale=node.parameters.saturation_amplitude_factor,
                            )
                            rr.wait(node.parameters.saturation_lead_time_in_ns * u.ns)
                        else:
                            qubit.xy.play(
                                qua_operation,
                                amplitude_scale=node.parameters.saturation_amplitude_factor,
                            )
                            qubit.align()
                        rr.measure("readout", qua_vars=(Im[i], Qm[i]))
                        # rr.wait(rr.depletion_time * u.ns)
                        save(Im[i], Im_st[i])
                        save(Qm[i], Qm_st[i])
                        # qubit.reset_qubit_thermal()
                        qubit.wait(25000)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                Ig_st[i].buffer(len(dfs)).buffer(n_runs).save(f"Ig{i + 1}")
                Qg_st[i].buffer(len(dfs)).buffer(n_runs).save(f"Qg{i + 1}")
                Im_st[i].buffer(len(dfs)).buffer(n_runs).save(f"Im{i + 1}")
                Qm_st[i].buffer(len(dfs)).buffer(n_runs).save(f"Qm{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}
    plt.show()


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
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


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Save_raw_results}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def save_raw_results(node: QualibrationNode[Parameters, Quam]):
    """Save the acquired vectors and a snapshot of the selected profile."""
    output_directory = CalibrationSaver().save_xarray(
        node.name,
        node.results["ds_raw"],
        profile_name=current_profile_name(),
    )
    node.namespace["calibration_run_directory"] = output_directory
    node.log(f"Raw calibration results saved to {output_directory}")


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
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


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot mean resonator responses and shot-level IQ separation."""
    figures = plot_per_qubit(
        plot_raw_amplitude,
        node.results["ds_raw"],
        node.namespace["qubits"],
        figure_name="amplitude",
    )
    plt.show()
    node.results["figures"] = figures
    if "calibration_run_directory" in node.namespace:
        figures_directory = CalibrationSaver().save_figures(
            node.namespace["calibration_run_directory"],
            node.results["figures"],
        )
        node.log(f"Calibration figures saved to {figures_directory}")



        


# %% {Propose_profile_update}
@node.run_action(skip_if=node.parameters.simulate)
def propose_profile_update(node: QualibrationNode[Parameters, Quam]):
    """Stage fitted resonator frequencies and apply them only after confirmation."""
    updates = {
        f"qubits.json.qubits.{q.name}.frequencies_hz.resonator": float(
            node.results["fit_results"][q.name]["frequency"]
        )
        for q in node.namespace["qubits"]
        if node.outcomes[q.name] == "successful"
    }
    if updates:
        proposal = ProfileUpdater().stage(node.name, updates, profile_name=current_profile_name())
        ProfileUpdater().confirm_and_apply(proposal)


# # %% {Save_results}
# @node.run_action()
# def save_results(node: QualibrationNode[Parameters, Quam]):
#     node.save()
