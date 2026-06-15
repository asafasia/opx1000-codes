# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam, create_machine
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from profiles import ProfileUpdater
from calibration_utils.iq_blobs import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_iq_blobs_dashboard,
)
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit in
the |g> state) and then after applying a x180 (pi) pulse to the qubit (bringing the qubit to the |e> state).
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the
      separation between |g> and |e> states aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination (at the center between the two blobs).
    - The repeat-until-success threshold, set equal to the state-discrimination threshold.
    - The readout confusion matrix, which is also influenced by the x180 pulse fidelity.

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).

State update:
    - The integration weight angle: qubit.resonator.operations["readout"].integration_weights_angle
    - the ge discrimination threshold: qubit.resonator.operations["readout"].threshold
    - the Repeat Until Success threshold: qubit.resonator.operations["readout"].rus_exit_threshold
    - The confusion matrix: qubit.resonator.operations["readout"].confusion_matrix
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="07_iq_blobs",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine=create_machine(),
)


node.machine = create_machine(qubit='q9')

node.machine.connect()
node.machine.qmm.close_all_qms()


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """
    Allow the user to locally set the node parameters for debugging purposes, or
    execution in the Python IDE.
    """
    # node.parameters.reset_type = "active"
    node.parameters.qubit_operation = "x180"

    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q10"]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Create the sweep axes and generate the QUA program from the pulse sequence and the
    node parameters.
    """
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_runs = node.parameters.num_shots  # Number of runs
    operation = node.parameters.operation
    selected_qubit_operation = node.parameters.qubit_operation
    qua_qubit_operation = "x180" if selected_qubit_operation == "x180_const" else selected_qubit_operation
    if node.parameters.pi_repetitions < 1:
        raise ValueError("pi_repetitions must be a positive integer.")
    if node.parameters.xy_to_readout_delay_in_ns < 0:
        raise ValueError("xy_to_readout_delay_in_ns cannot be negative.")
    for qubit in qubits:
        if qua_qubit_operation not in qubit.xy.operations:
            raise ValueError(
                f"{qubit.name} does not define qubit operation {qua_qubit_operation!r}."
            )
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}),
    }

    with program() as node.namespace["qua_program"]:
        I_g, I_g_st, Q_g, Q_g_st, n, n_st = node.machine.declare_qua_variables()
        I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            # Acquire the ground and prepared clouds in independent shot loops.
            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)
                for qubit in multiplexed_qubits.values():
                    qubit.reset(
                        node.parameters.reset_type,
                        node.parameters.simulate,
                        # log_callable=node.log,
                    )
                align()
                for i, qubit in multiplexed_qubits.items():
                    qubit.resonator.measure(operation, qua_vars=(I_g[i], Q_g[i]))

                    save(I_g[i], I_g_st[i])
                    save(Q_g[i], Q_g_st[i])
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                align()

            with for_(n, 0, n < n_runs, n + 1):
                for qubit in multiplexed_qubits.values():
                    qubit.reset(
                        node.parameters.reset_type,
                        node.parameters.simulate,
                        # log_callable=node.log,
                    )
                align()

                for qubit in multiplexed_qubits.values():
                    repetitions = (
                        node.parameters.pi_repetitions
                        if selected_qubit_operation == "x180_const"
                        else 1
                    )
                    for _ in range(repetitions):
                        qubit.xy.play(
                            qua_qubit_operation,
                            amplitude_scale=node.parameters.qubit_amplitude_factor,
                        )
                # Synchronize XY and resonator timelines, then delay readout explicitly.
                align()
                for qubit in multiplexed_qubits.values():
                    qubit.resonator.wait(node.parameters.xy_to_readout_delay_in_ns * u.ns)
                for i, qubit in multiplexed_qubits.items():
                    qubit.resonator.measure(operation, qua_vars=(I_e[i], Q_e[i]))
                    save(I_e[i], I_e_st[i])
                    save(Q_e[i], Q_e_st[i])
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_g_st[i].buffer(n_runs).save(f"Ig{i + 1}")
                Q_g_st[i].buffer(n_runs).save(f"Qg{i + 1}")
                I_e_st[i].buffer(n_runs).save(f"Ie{i + 1}")
                Q_e_st[i].buffer(n_runs).save(f"Qe{i + 1}")


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


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw".
    """
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
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)


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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """
    Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary.
    """
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
    """
    Plot the raw and fitted data in specific figures whose shape is given by
    qubit.grid_location.
    """
    figures = plot_per_qubit(
        plot_iq_blobs_dashboard,
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        figure_name="iq_blobs_dashboard",
    )
    plt.show()
    node.results["figures"] = figures
    if "calibration_run_directory" in node.namespace:
        figures_directory = CalibrationSaver().save_figures(
            node.namespace["calibration_run_directory"],
            node.results["figures"],
        )
        node.log(f"Calibration figures saved to {figures_directory}")


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            fit_result = node.results["fit_results"][q.name]
            if not all(
                np.isfinite(fit_result[name])
                for name in ("iw_angle", "ge_threshold", "rus_threshold")
            ):
                node.log(f"Skipping {q.name} update because a fitted readout parameter is not finite.")
                continue

            if node.outcomes[q.name] == "failed":
                node.log(f"{q.name} failed IQ-blob quality checks; its fitted parameters can still be reviewed.")
            operation = q.resonator.operations[node.parameters.operation]
            operation.integration_weights_angle -= float(fit_result["iw_angle"])
            # Convert the thresholds back to demod units
            operation.threshold = float(fit_result["ge_threshold"]) * operation.length / 2**12
            operation.rus_exit_threshold = float(fit_result["rus_threshold"]) * operation.length / 2**12
            if node.parameters.operation == "readout":
                q.resonator.confusion_matrix = fit_result["confusion_matrix"]


# %% {Propose_profile_update}
@node.run_action(skip_if=node.parameters.simulate)
def propose_profile_update(node: QualibrationNode[Parameters, Quam]):
    """Stage the fitted readout angle and threshold for successful qubits."""
    if node.parameters.operation != "readout":
        node.log(
            f"Profile update skipped: operation {node.parameters.operation!r} "
            "does not use the profile's default readout parameters."
        )
        return

    updates = {}
    reset_metric_key = (
        node.parameters.reset_type
        if node.parameters.reset_type in {"active", "thermal"}
        else None
    )
    for q in node.namespace["qubits"]:
        fit_result = node.results["fit_results"][q.name]
        if not all(
            np.isfinite(fit_result[name])
            for name in ("iw_angle", "ge_threshold", "rus_threshold")
        ):
            continue
        operation = q.resonator.operations["readout"]
        updates[f"qubits.json.qubits.{q.name}.readout.integration_weights_angle_rad"] = float(
            operation.integration_weights_angle
        )
        updates[f"qubits.json.qubits.{q.name}.readout.threshold"] = float(operation.threshold)
        updates[f"qubits.json.qubits.{q.name}.readout.rus_exit_threshold"] = float(
            operation.rus_exit_threshold
        )
        if reset_metric_key is not None:
            updates[
                f"metrics.json.qubits.{q.name}.readout.fidelity_percent.{reset_metric_key}"
            ] = float(fit_result["readout_fidelity"])

    if updates:
        failed_qubits = [q.name for q in node.namespace["qubits"] if node.outcomes[q.name] == "failed"]
        if failed_qubits:
            node.log(
                "WARNING: proposing fitted parameters despite failed IQ-blob quality checks for "
                + ", ".join(failed_qubits)
            )
        proposal = ProfileUpdater().stage(node.name, updates, profile_name=current_profile_name())
        ProfileUpdater().confirm_and_apply(proposal)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
