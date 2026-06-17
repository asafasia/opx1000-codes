# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import dataclasses
import os
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam, create_machine
from profiles import load_profile
from calibration_utils.power_rabi import (
    Parameters,
    get_number_of_pulses,
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

# %% {Description}
description = """
        POWER RABI
This sequence calibrates either the GE or EF transition. For transition="ge", it repeatedly executes
the selected GE qubit pulse (such as x180, x180_drag, x180_cosine, x90...) 'N' times and measures
the state across pulse amplitudes and number of pulses. For transition="ef", it first prepares |e>
with x180 and then calibrates a single EF_x180 pulse across amplitude.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit frequency (node 03a_qubit_spectroscopy.py).
    - Having set the qubit gates duration (qubit.xy.operations["x180"].length).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - GE: the pulse amplitude corresponding to the specified operation (x180, x90...)
      (qubit.xy.operations[operation].amplitude).
    - EF: the EF_x180 pulse amplitude (qubit.xy.operations["EF_x180"].amplitude).
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="04b_power_rabi",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine=create_machine(),
)

node.machine = create_machine(qubit="q9")

node.machine.connect()  # Connect to the machine to fetch the qubits information and populate the node namespace if needed

node.machine.qmm.close_all_qms()


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.use_state_discrimination = True
    node.parameters.reset_type = "thermal"  # "active" or "thermal"
    node.parameters.num_shots = 500
    node.parameters.transition = "ge"
    node.parameters.pi_repetitions = 4
    # node.parameters.operation = "x180_drag"
    pass


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


def active_operation(parameters: Parameters) -> str:
    """Return the operation calibrated by the selected transition."""
    return "EF_x180" if parameters.transition == "ef" else parameters.operation


def has_gef_readout_calibration(qubit) -> bool:
    """Return whether the qubit has the data needed for dedicated GEF readout."""
    return (
        callable(getattr(qubit, "readout_state_gef", None))
        and getattr(qubit.resonator, "GEF_frequency_shift", None) is not None
    )


def ensure_operation_available(qubit, operation: str, transition: str) -> None:
    """Validate GE operations and provide a default EF_x180 pulse when missing."""
    if operation in qubit.xy.operations:
        return
    if transition == "ef" and operation == "EF_x180":
        x180 = qubit.xy.operations["x180"]
        qubit.xy.operations["EF_x180"] = (
            dataclasses.replace(x180, alpha=0.0)
            if hasattr(x180, "alpha")
            else dataclasses.replace(x180)
        )
        return
    raise ValueError(f"{qubit.name} does not define operation {operation!r}.")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # The number of averages
    operation = active_operation(node.parameters)  # The qubit operation to play
    for qubit in qubits:
        ensure_operation_available(qubit, operation, node.parameters.transition)
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    # Number of applied Rabi pulses sweep
    N_pi_vec = get_number_of_pulses(node.parameters)
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "nb_of_pulses": xr.DataArray(N_pi_vec, attrs={"long_name": "number of pulses"}),
        "amp_prefactor": xr.DataArray(
            amps, attrs={"long_name": "pulse amplitude prefactor"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        npi = declare(int)  # QUA variable for the number of qubit pulses
        count = declare(int)  # QUA variable for counting repeated transition pulses

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(npi, N_pi_vec)):
                    with for_(*from_array(a, amps)):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                # log_callable=node.log,
                            )

                        align()

                        # Qubit manipulation
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.transition == "ef":
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency
                                )
                                qubit.xy.play("x180")
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency
                                    - qubit.anharmonicity
                                )
                                with for_(count, 0, count < npi, count + 1):
                                    qubit.xy.play("EF_x180", amplitude_scale=a)
                            else:
                                # Loop for error amplification (perform many qubit pulses)
                                with for_(count, 0, count < npi, count + 1):
                                    qubit.xy.play(operation, amplitude_scale=a)
                        align()

                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                if node.parameters.transition == "ef":
                                    if has_gef_readout_calibration(qubit):
                                        qubit.readout_state_gef(state[i])
                                    else:
                                        qubit.readout_state(state[i])
                                else:
                                    qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure(
                                    "readout", qua_vars=(I[i], Q[i])
                                )
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

                        # Return the qubit to the ground state before the next shot.
                        align()

        with stream_processing():
            n_st.save("n")
            for i, qubit in enumerate(qubits):
                if operation.endswith("x180") or operation.startswith("x180_"):
                    if node.parameters.use_state_discrimination:
                        state_st[i].buffer(len(amps)).buffer(
                            len(N_pi_vec)
                        ).average().save(f"state{i + 1}")
                    else:
                        I_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(
                            f"I{i + 1}"
                        )
                        Q_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(
                            f"Q{i + 1}"
                        )

                elif operation in ["x90", "-x90", "y90", "-y90"]:
                    if node.parameters.use_state_discrimination:
                        state_st[i].buffer(len(amps)).buffer(
                            len(N_pi_vec)
                        ).average().save(f"state{i + 1}")
                    else:
                        I_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(
                            f"I{i + 1}"
                        )
                        Q_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(
                            f"Q{i + 1}"
                        )
                else:
                    raise ValueError(f"Unrecognized operation {operation}.")


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
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
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
    """Stage fitted pulse amplitudes and apply them only after confirmation."""
    updates = {}
    profile_name = current_profile_name()
    qubit_profiles = load_profile(profile_name)["qubits"]["qubits"]
    operation = active_operation(node.parameters)
    for q in node.namespace["qubits"]:
        if node.outcomes[q.name] != "successful":
            continue
        if operation not in qubit_profiles[q.name]["operations"]:
            node.log(
                f"Profile update skipped: operation {operation!r} "
                "does not have a dedicated profile pulse."
            )
            continue
        amplitude = float(node.results["fit_results"][q.name]["opt_amp"])
        pulse_name = qubit_profiles[q.name]["operations"][operation]
        updates[f"pulses.json.pulses.{q.name}.{pulse_name}.amplitude"] = amplitude
    if updates:
        proposal = ProfileUpdater().stage(node.name, updates, profile_name=profile_name)
        ProfileUpdater().confirm_and_apply(proposal)


# # %% {Save_results}
# @node.run_action()
# def save_results(node: QualibrationNode[Parameters, Quam]):
#     node.save()
