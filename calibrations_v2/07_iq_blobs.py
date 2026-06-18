"""Class-based v2 migration for 07_iq_blobs."""

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
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from quam_config import Quam, create_machine
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from profiles import ProfileUpdater
from calibration_utils.iq_blobs import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    save_fit_results,
    plot_iq_blobs_dashboard,
)
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from quam.components.pulses import SquareReadoutPulse

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration, CalibrationOptions
else:
    from .base import BaseCalibration, CalibrationOptions

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


class IqBlobs(BaseCalibration[Parameters, Quam]):
    """v2 class migration for ``calibrations/07_iq_blobs.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="07_iq_blobs",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    def create_qua_program(self):
        node = self
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
        states = list(node.parameters.states)
        selected_qubit_operation = node.parameters.qubit_operation
        qua_qubit_operation = (
            "x180"
            if selected_qubit_operation == "x180_const"
            else selected_qubit_operation
        )
        if states not in (["g", "e"], ["g", "e", "f"]):
            raise ValueError('states must be either ["g", "e"] or ["g", "e", "f"].')
        if node.parameters.pi_repetitions < 1:
            raise ValueError("pi_repetitions must be a positive integer.")
        if node.parameters.xy_to_readout_delay_in_ns < 0:
            raise ValueError("xy_to_readout_delay_in_ns cannot be negative.")
        for qubit in qubits:
            if qua_qubit_operation not in qubit.xy.operations:
                raise ValueError(
                    f"{qubit.name} does not define qubit operation {qua_qubit_operation!r}."
                )
            if "f" in states and "EF_x180" not in qubit.xy.operations:
                raise ValueError(
                    f"{qubit.name} does not define qubit operation 'EF_x180'."
                )
            if (
                operation == "readout_GEF"
                and "readout_GEF" not in qubit.resonator.operations
            ):
                readout_op = qubit.resonator.operations["readout"]
                new_length = int(
                    round(readout_op.length * 1.5 / 4) * 4
                )  # multiple of 4 ns
                qubit.resonator.operations["readout_GEF"] = SquareReadoutPulse(
                    length=new_length,
                    amplitude=readout_op.amplitude,
                    digital_marker=readout_op.digital_marker,
                    axis_angle=readout_op.axis_angle,
                    threshold=None,
                    rus_exit_threshold=None,
                    integration_weights=[[1.0, new_length]],
                    integration_weights_angle=readout_op.integration_weights_angle,
                )
        # Register the sweep axes to be added to the dataset when fetching data
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "n_runs": xr.DataArray(
                np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}
            ),
        }

        with program() as node.namespace["qua_program"]:
            I_g, I_g_st, Q_g, Q_g_st, n, n_st = node.machine.declare_qua_variables()
            I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.declare_qua_variables()
            if "f" in states:
                I_f, I_f_st, Q_f, Q_f_st, _, _ = node.machine.declare_qua_variables()

            for multiplexed_qubits in qubits.batch():
                # Acquire the ground and prepared clouds in independent shot loops.
                if "f" in states:
                    for qubit in multiplexed_qubits.values():
                        shift = (
                            qubit.resonator.GEF_frequency_shift
                            if qubit.resonator.GEF_frequency_shift is not None
                            else 0
                        )
                        qubit.resonator.update_frequency(
                            qubit.resonator.intermediate_frequency + shift
                        )

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
                        qubit.resonator.wait(
                            node.parameters.xy_to_readout_delay_in_ns * u.ns
                        )
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure(operation, qua_vars=(I_e[i], Q_e[i]))
                        save(I_e[i], I_e_st[i])
                        save(Q_e[i], Q_e_st[i])
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    align()

                if "f" in states:
                    with for_(n, 0, n < n_runs, n + 1):
                        for qubit in multiplexed_qubits.values():
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                # log_callable=node.log,
                            )
                        align()

                        for qubit in multiplexed_qubits.values():
                            qubit.xy.play("x180")
                            update_frequency(
                                qubit.xy.name,
                                qubit.xy.intermediate_frequency - qubit.anharmonicity,
                            )
                            qubit.xy.play("EF_x180")
                            update_frequency(
                                qubit.xy.name, qubit.xy.intermediate_frequency
                            )
                        align()
                        for qubit in multiplexed_qubits.values():
                            qubit.resonator.wait(
                                node.parameters.xy_to_readout_delay_in_ns * u.ns
                            )
                        for i, qubit in multiplexed_qubits.items():
                            qubit.resonator.measure(
                                operation, qua_vars=(I_f[i], Q_f[i])
                            )
                            save(I_f[i], I_f_st[i])
                            save(Q_f[i], Q_f_st[i])
                            qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    I_g_st[i].buffer(n_runs).save(f"Ig{i + 1}")
                    Q_g_st[i].buffer(n_runs).save(f"Qg{i + 1}")
                    I_e_st[i].buffer(n_runs).save(f"Ie{i + 1}")
                    Q_e_st[i].buffer(n_runs).save(f"Qe{i + 1}")
                    if "f" in states:
                        I_f_st[i].buffer(n_runs).save(f"If{i + 1}")
                        Q_f_st[i].buffer(n_runs).save(f"Qf{i + 1}")

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
        """
        Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"
        and the fitted results in the "fit_results" dictionary.
        """
        node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
        node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
        if "calibration_run_directory" in node.namespace:
            output_path = save_fit_results(
                node.namespace["calibration_run_directory"],
                node.results["fit_results"],
            )
            node.log(f"Fit results saved to {output_path}")

        # Log the relevant information extracted from the data analysis
        log_fitted_results(node.results["fit_results"], log_callable=node.log)
        node.outcomes = {
            qubit_name: ("successful" if fit_result["success"] else "failed")
            for qubit_name, fit_result in node.results["fit_results"].items()
        }

    def plot_data(self):
        node = self
        """
        Plot the raw and fitted data in specific figures whose shape is given by
        qubit.grid_location.
        """
        figures = plot_per_qubit(
            plot_iq_blobs_dashboard,
            node.results["ds_raw"],
            node.namespace["qubits"],
            node.results["ds_fit"],
            run_metadata={
                "operation": node.parameters.operation,
                "reset_type": node.parameters.reset_type,
                "num_shots": node.parameters.num_shots,
                "pi_repetitions": node.parameters.pi_repetitions,
                "states": node.parameters.states,
                "qubit_operation": node.parameters.qubit_operation,
            },
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

    def update_state(self):
        node = self
        """Update the relevant parameters if the qubit data analysis was successful."""
        with node.record_state_updates():
            for q in node.namespace["qubits"]:
                fit_result = node.results["fit_results"][q.name]
                if not all(
                    np.isfinite(fit_result[name])
                    for name in ("iw_angle", "ge_threshold", "rus_threshold")
                ):
                    node.log(
                        f"Skipping {q.name} update because a fitted readout parameter is not finite."
                    )
                    continue

                if node.outcomes[q.name] == "failed":
                    node.log(
                        f"{q.name} failed IQ-blob quality checks; its fitted parameters can still be reviewed."
                    )
                operation = q.resonator.operations[node.parameters.operation]
                operation.integration_weights_angle -= float(fit_result["iw_angle"])
                # Convert the thresholds back to demod units
                operation.threshold = (
                    float(fit_result["ge_threshold"]) * operation.length / 2**12
                )
                operation.rus_exit_threshold = (
                    float(fit_result["rus_threshold"]) * operation.length / 2**12
                )
                if node.parameters.operation == "readout":
                    q.resonator.confusion_matrix = fit_result["confusion_matrix"]

    def propose_profile_update(self):
        node = self
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
            updates[
                f"qubits.json.qubits.{q.name}.readout.integration_weights_angle_rad"
            ] = float(operation.integration_weights_angle)
            updates[f"qubits.json.qubits.{q.name}.readout.threshold"] = float(
                operation.threshold
            )
            updates[f"qubits.json.qubits.{q.name}.readout.rus_exit_threshold"] = float(
                operation.rus_exit_threshold
            )
            if reset_metric_key is not None:
                updates[
                    f"metrics.json.qubits.{q.name}.readout.fidelity_percent.{reset_metric_key}"
                ] = float(fit_result["readout_fidelity"])

        if updates:
            failed_qubits = [
                q.name
                for q in node.namespace["qubits"]
                if node.outcomes[q.name] == "failed"
            ]
            if failed_qubits:
                node.log(
                    "WARNING: proposing fitted parameters despite failed IQ-blob quality checks for "
                    + ", ".join(failed_qubits)
                )
            proposal = ProfileUpdater().stage(
                node.name, updates, profile_name=current_profile_name()
            )
            ProfileUpdater().confirm_and_apply(proposal)


if __name__ == "__main__":
    parameters = Parameters()

    parameters.qubit_operation = "x180_const"
    parameters.states = ["g", "e"]
    parameters.reset_type = "active"
    parameters.num_shots = 10000

    options = CalibrationOptions()

    calibration = IqBlobs(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
