# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm import SimulationConfig
from qm.exceptions import QMSimulationError
from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam, create_machine
from saver import CalibrationSaver, current_profile_name
from updater import ProfileUpdater
from calibration_utils.iq_blobs import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    log_blob_diagnostics,
    plot_iq_blobs_dashboard,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.data import XarrayDataFetcher
from utils.plotting_settings import FIGURE_SIZE
from utils.simulation import plot_waveform_report_safely

# %% {Description}
description = """
        IQ BLOBS - SEPARATE ACQUISITIONS
This diagnostic calibration acquires the ground and excited IQ clouds in two
independent QUA programs and jobs. It then combines both acquisitions and uses
the standard IQ-blobs analysis and dashboard.

This node intentionally does not update the device state.
"""

node = QualibrationNode[Parameters, Quam](
    name="07_iq_blobs_separate",
    description=description,
    parameters=Parameters(),
)

node.machine = create_machine()

@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local debugging parameter overrides."""
    # node.parameters.reset_type = "active"
    # node.parameters.qubits = ["q9"]
    # node.parameters.simulate = True
    # node.parameters.samples = 1000
    # node.parameters.qubit_operation = 'x180_const'
    node.parameters.qubit_operation = 'saturation'


def make_state_program(
    node: QualibrationNode[Parameters, Quam],
    state: str,
    n_runs: int | None = None,
    initialization_wait_in_ns: int = 200_000,
):
    """Build one independent IQ acquisition program for state 'g' or 'e'."""
    if state not in {"g", "e"}:
        raise ValueError("state must be 'g' or 'e'")

    u = unit(coerce_to_integer=True)
    qubits = node.namespace["qubits"]
    num_qubits = len(qubits)
    n_runs = node.parameters.num_shots if n_runs is None else n_runs
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

    with program() as qua_program:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)
                for qubit in multiplexed_qubits.values():
                    # qubit.resonator.wait(15000)  # 300 µs
                    # qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    pass
                align()

                if state == "e":
                
                    qubit.xy.play(
                        qua_qubit_operation,
                        amplitude_scale=node.parameters.qubit_amplitude_factor,
                    )
                    # Synchronize XY and resonator timelines before the explicit delay.
                    # align()
                    # for qubit in multiplexed_qubits.values():
                    #     qubit.resonator.wait(node.parameters.xy_to_readout_delay_in_ns * u.ns)
                    #     # qubit.reset_qubit_thermal()
                    align()

                for i, qubit in multiplexed_qubits.items():
                    qubit.resonator.measure(operation, qua_vars=(I[i], Q[i]))
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    # qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    qubit.resonator.wait(15000)  # 300 µs, to ensure the resonator is depleted before the next shot, even if the qubit is in |e> and T1 is long.
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(n_runs).save(f"I{i + 1}")
                Q_st[i].buffer(n_runs).save(f"Q{i + 1}")

    return qua_program


def acquire_state(
    node: QualibrationNode[Parameters, Quam], qmm, config: dict, state: str
) -> xr.Dataset:
    """Execute one state in its own QM session and rename its I/Q results."""
    node.log(f"Starting independent {state}-state acquisition")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_programs"][state])
        fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in fetcher:
            progress_counter(
                fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=fetcher.t_start,
            )
        node.log(f"{state}-state acquisition:\n{job.execution_report()}")
    suffix = "g" if state == "g" else "e"
    return dataset.rename({"I": f"I{suffix}", "Q": f"Q{suffix}"})


def simulate_state_program(node, qmm, config, qua_program):
    """Simulate and plot samples, falling back to the waveform report if needed."""
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)
    job = qmm.simulate(config, qua_program, simulation_config)
    # QOP v2 may return the simulated job before reports and samples are ready.
    job.wait_until("Done", timeout=node.parameters.timeout)
    wf_report = job.get_simulated_waveform_report()
    try:
        samples = job.get_simulated_samples()
        fig, axes = plt.subplots(
            nrows=len(samples.keys()),
            sharex=True,
            squeeze=False,
            figsize=FIGURE_SIZE,
        )
        for axis, controller in zip(axes.flat, samples.keys()):
            plt.sca(axis)
            samples[controller].plot()
            axis.set_title(controller)
        fig.tight_layout()
    except QMSimulationError:
        node.log(
            "QOP simulated the program but qm-qua could not pull analog samples; "
            "showing the waveform report instead."
        )
        fig = plot_waveform_report_safely(wf_report, samples=None)
        samples = None
    return samples, fig, wf_report


# %% {Create_QUA_programs}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_programs(node: QualibrationNode[Parameters, Quam]):
    """Create independent ground-state and excited-state QUA programs."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    n_runs = node.parameters.num_shots
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.arange(n_runs), attrs={"long_name": "shot index"}),
    }
    node.namespace["qua_programs"] = {
        state: make_state_program(node, state) for state in ("g", "e")
    }


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_programs(node: QualibrationNode[Parameters, Quam]):
    """Simulate short representative versions of both acquisition programs."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    simulations = {}
    for state in ("g", "e"):
        # Simulating all experimental shots can span seconds and overwhelm the
        # simulator sample pull. One shot contains the complete state sequence.
        qua_program = make_state_program(
            node,
            state,
            n_runs=1,
            initialization_wait_in_ns=100,
        )
        samples, fig, wf_report = simulate_state_program(node, qmm, config, qua_program)
        # Saving raw v2 waveform reports currently fails in qualang-tools for
        # MW-FEM reports. Keep the rendered figure, which is the useful output.
        simulations[state] = {"figure": fig}
    node.results["simulation"] = simulations
    plt.show()


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_programs(node: QualibrationNode[Parameters, Quam]):
    """Execute ground and excited acquisition as separate jobs, then merge them."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    ground = acquire_state(node, qmm, config, "g")
    excited = acquire_state(node, qmm, config, "e")

    node.results["ds_ground_raw"] = ground
    node.results["ds_excited_raw"] = excited
    node.results["ds_raw"] = process_raw_dataset(xr.merge([ground, excited]), node)


# %% {Save_raw_results}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def save_raw_results(node: QualibrationNode[Parameters, Quam]):
    """Save the merged acquisition and a snapshot of the selected profile."""
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
    """Load a previously acquired combined IQ dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Run the standard IQ-blobs analysis on the merged dataset."""
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {name: asdict(result) for name, result in fit_results.items()}
    log_blob_diagnostics(node.results["ds_raw"], log_callable=node.log)
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        name: ("successful" if result["success"] else "failed")
        for name, result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the standard combined IQ-blobs dashboard."""
    dashboard = plot_iq_blobs_dashboard(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
    )
    node.results["figures"] = {"iq_blobs_separate_dashboard": dashboard}
    plt.show()
    if "calibration_run_directory" in node.namespace:
        figures_directory = CalibrationSaver().save_figures(
            node.namespace["calibration_run_directory"],
            node.results["figures"],
        )
        node.log(f"Calibration figures saved to {figures_directory}")


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update fitted readout parameters in memory for successful qubits."""
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
            operation.threshold = float(fit_result["ge_threshold"]) * operation.length / 2**12
            operation.rus_exit_threshold = float(fit_result["rus_threshold"]) * operation.length / 2**12
            if node.parameters.operation == "readout":
                q.resonator.confusion_matrix = fit_result["confusion_matrix"]


# %% {Propose_profile_update}
@node.run_action(skip_if=node.parameters.simulate)
def propose_profile_update(node: QualibrationNode[Parameters, Quam]):
    """Ask before applying fitted readout angle and threshold to the profile."""
    if node.parameters.operation != "readout":
        node.log(
            f"Profile update skipped: operation {node.parameters.operation!r} "
            "does not use the profile's default readout parameters."
        )
        return

    updates = {}
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
    """Save the two raw acquisitions, merged analysis, and dashboard."""
    node.save()
