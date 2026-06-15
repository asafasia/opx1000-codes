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
from calibration_utils.qubit_spectroscopy import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from profiles import ProfileUpdater
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive frequencies.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually
from the node parameters.

The data is post-processed to determine the qubit resonance frequency and the width of the peak.

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended when using external mixers or the Octave.
Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The qubit 0->1 frequency: qubit.f_01 & qubit.xy.RF_frequency
    - The integration weight angle to get the state discrimination along the 'I' quadrature: qubit.resonator.operations["readout"].integration_weights_angle.
    - (optional) The saturation pulse amplitude to get the targeted fwhm: qubit.xy.operations["saturation"].amplitude.
    - (optional) The guessed x180/x90 pulse amplitude: qubit.xy.operations["x180"/"x90"].amplitude.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="03a_qubit_spectroscopy",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.use_state_discrimination = False
    node.parameters.num_shots = 1000
    node.parameters.operation_amplitude_factor = 0.01
    node.parameters.frequency_span_in_mhz = 30
    node.parameters.frequency_step_in_mhz = 0.3
    # node.parameters.simulate = True
    pass


# Create the machine directly from profiles/main without loading state.json.
node.machine = create_machine(qubit='q9')

node.machine.connect()  # Connect to the machine to fetch the qubits information and populate the node namespace if needed

node.machine.qmm.close_all_qms()


def validate_readout_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> None:
    """Ensure fetched results match the requested readout mode."""
    variables = set(ds.data_vars)
    expected = {"state"} if use_state_discrimination else {"I", "Q"}
    unexpected = {"I", "Q"} if use_state_discrimination else {"state"}
    missing = expected - variables
    present_unexpected = unexpected & variables
    if missing or present_unexpected:
        raise RuntimeError(
            "Qubit-spectroscopy readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(variables)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
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
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
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
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            # log_callable=node.log,
                        )
                        # Get the duration of the operation from the node parameters or the state
                        duration = (
                            operation_len
                            if operation_len is not None
                            else qubit.xy.operations[operation].length
                        )
                        # Update the qubit frequency
                        qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                        # Play the saturation pulse
                        qubit.xy.play(
                            operation,
                            amplitude_scale=operation_amp,
                            duration=duration // 4,
                        )
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        # readout the resonator
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

                    align()

        with stream_processing():
            # pass
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(dfs)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
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
    plt.show()


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
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
    validate_readout_dataset(dataset, node.parameters.use_state_discrimination)
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
    validate_readout_dataset(node.results["ds_raw"], node.parameters.use_state_discrimination)
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
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    figures = plot_per_qubit(
        plot_raw_data_with_fit,
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        figure_name="amplitude",
        use_state_discrimination=node.parameters.use_state_discrimination,
        operation=node.parameters.operation,
        operation_amplitude_factor=node.parameters.operation_amplitude_factor,
        operation_len_in_ns=node.parameters.operation_len_in_ns,
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
    """Stage fitted qubit frequencies and apply them only after confirmation."""
    updates = {
        f"qubits.json.qubits.{q.name}.frequencies_hz.qubit_f01": float(
            node.results["fit_results"][q.name]["frequency"]
        )
        for q in node.namespace["qubits"]
        if node.outcomes[q.name] == "successful"
    }
    if updates:
        proposal = ProfileUpdater().stage(node.name, updates, profile_name=current_profile_name())
        ProfileUpdater().confirm_and_apply(proposal)


# %% {Save_results}
# @node.run_action()
# def save_results(node: QualibrationNode[Parameters, Quam]):
#     node.save()
