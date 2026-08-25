"""Class-based calibration for 07_iq_blobs."""

from __future__ import annotations

from pprint import pprint
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.multi_user import qm_session
from calibrations.runtime_estimation import progress_counter
from quam_config import Quam, create_machine
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from utils.readout_macro import active_reset_configured
from profiles import ProfileUpdater
from calibration_utils.iq_blobs import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_iq_blobs_dashboard,
)
from calibration_utils.analysis_base import FunctionalAnalysis
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from quam.components.pulses import SquareReadoutPulse

if __package__ in {None, ""}:
    from calibrations.core import BaseCalibration, CalibrationOptions
else:
    from .core import BaseCalibration, CalibrationOptions

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
    - The binary fidelity/assignment matrix: qubit.resonator.confusion_matrix
"""


def _has_gef_centers(qubit) -> bool:
    centers = getattr(qubit.resonator, "gef_centers", None)
    if centers is None:
        return False
    try:
        centers_array = np.asarray(centers, dtype=float)
    except (TypeError, ValueError):
        return False
    return centers_array.shape == (3, 2) and bool(np.isfinite(centers_array).all())


def _copy_integration_weights(integration_weights):
    if integration_weights is None:
        return None
    return [
        [float(weight_segment[0]), int(weight_segment[1])]
        for weight_segment in integration_weights
    ]


def _rotate_iq_centers(centers, angle: float):
    """Express fitted centers in the IQ frame used after the IW-angle update."""
    centers = np.asarray(centers, dtype=float)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.column_stack(
        (
            centers[:, 0] * cosine - centers[:, 1] * sine,
            centers[:, 0] * sine + centers[:, 1] * cosine,
        )
    )


def reset_qubit_active_gef(qubit, max_attempts: int = 15) -> None:
    """Reset G/E/F through the profile-configured readout discriminator."""
    active_reset_configured(
        qubit,
        num_states=3,
        max_attempts=max_attempts,
    )


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
    """Class-based calibration for ``calibrations/07_iq_blobs.py``."""

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

    def create_analysis(self):
        return FunctionalAnalysis(
            self,
            # IQ blobs currently save voltage-scaled ds_raw during execution/load.
            fit=lambda ds: fit_raw_data(ds, self),
            log=lambda result: log_fitted_results(
                result.fit_results,
                log_callable=self.log,
            ),
        )

    def create_qua_program(self):
        node = self
        """
        Create the sweep axes and generate the QUA program from the pulse sequence and the
        node parameters.
        """
        # Get the active qubits from the node and organize them by batches
        node.namespace["qubits"] = qubits = get_qubits(node)
        num_qubits = len(qubits)

        n_runs = node.parameters.num_shots  # Number of runs
        operation = node.parameters.operation
        states = list(node.parameters.states)
        reset_type = node.parameters.reset_type
        use_gef_active_reset = "f" in states and reset_type == "active"
        use_simple_active_gef_reset = reset_type == "active_gef"
        selected_qubit_operation = node.parameters.qubit_operation
        qua_qubit_operation = (
            "x180"
            if selected_qubit_operation == "x180_const"
            else selected_qubit_operation
        )
        valid_states = {"g", "e", "f"}
        if (
            len(states) not in (2, 3)
            or len(set(states)) != len(states)
            or set(states) - valid_states
        ):
            raise ValueError(
                'states must be a unique two-state pair from ["g", "e", "f"] or ["g", "e", "f"].'
            )
        if node.parameters.pi_repetitions < 1:
            raise ValueError("pi_repetitions must be a positive integer.")
        if node.parameters.active_gef_reset_attempts < 1:
            raise ValueError("active_gef_reset_attempts must be a positive integer.")
        for qubit in qubits:
            if (
                use_gef_active_reset or use_simple_active_gef_reset
            ) and not _has_gef_centers(qubit):
                raise ValueError(
                    f"{qubit.name} active_gef reset requires qubit.resonator.gef_centers. "
                    "Run IQ blobs with states ['g', 'e', 'f'] and reset_type='thermal' first."
                )
            if "e" in states and qua_qubit_operation not in qubit.xy.operations:
                raise ValueError(
                    f"{qubit.name} does not define qubit operation {qua_qubit_operation!r}."
                )
            if "f" in states:
                if "x180" not in qubit.xy.operations:
                    raise ValueError(
                        f"{qubit.name} does not define qubit operation 'x180'."
                    )
                if "EF_x180" not in qubit.xy.operations:
                    raise ValueError(
                        f"{qubit.name} does not define qubit operation 'EF_x180'."
                    )
            if (
                operation == "readout_GEF"
                or use_gef_active_reset
                or use_simple_active_gef_reset
            ) and "readout_GEF" not in qubit.resonator.operations:
                readout_op = qubit.resonator.operations["readout"]
                qubit.resonator.operations["readout_GEF"] = SquareReadoutPulse(
                    length=readout_op.length,
                    amplitude=readout_op.amplitude,
                    digital_marker=readout_op.digital_marker,
                    axis_angle=readout_op.axis_angle,
                    threshold=readout_op.threshold,
                    rus_exit_threshold=readout_op.rus_exit_threshold,
                    integration_weights=_copy_integration_weights(
                        readout_op.integration_weights
                    ),
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

            def measure_cloud(qubit, i_quadrature, q_quadrature):
                uses_gef_frequency = operation == "readout_GEF"
                if uses_gef_frequency:
                    qubit.resonator.update_frequency(
                        int(
                            qubit.resonator.intermediate_frequency
                            + qubit.resonator.GEF_frequency_shift
                        )
                    )
                qubit.resonator.measure(
                    operation,
                    qua_vars=(i_quadrature, q_quadrature),
                )
                if uses_gef_frequency:
                    qubit.resonator.update_frequency(
                        qubit.resonator.intermediate_frequency
                    )

            if use_gef_active_reset:
                reset_state = [declare(int) for _ in range(num_qubits)]
                reset_attempt = declare(int)

                def reset_qubit(qubit, qubit_index):
                    with for_(
                        reset_attempt,
                        0,
                        reset_attempt < node.parameters.active_gef_reset_attempts,
                        reset_attempt + 1,
                    ):
                        qubit.readout_state_gef(reset_state[qubit_index])
                        align()
                        with if_(reset_state[qubit_index] == 1):
                            update_frequency(
                                qubit.xy.name,
                                int(qubit.xy.intermediate_frequency),
                                keep_phase=True,
                            )
                            qubit.xy.play("x180")
                        with if_(reset_state[qubit_index] == 2):
                            update_frequency(
                                qubit.xy.name,
                                int(
                                    qubit.xy.intermediate_frequency
                                    - qubit.anharmonicity
                                ),
                                keep_phase=True,
                            )
                            qubit.xy.play("EF_x180")
                            update_frequency(
                                qubit.xy.name,
                                int(qubit.xy.intermediate_frequency),
                                keep_phase=True,
                            )
                            qubit.xy.play("x180")
                        align()

            else:

                def reset_qubit(qubit, qubit_index):
                    if reset_type == "active_gef":
                        reset_qubit_active_gef(
                            qubit,
                            max_attempts=node.parameters.active_gef_reset_attempts,
                        )
                    else:
                        qubit.reset(
                            reset_type,
                            node.parameters.simulate,
                            # log_callable=node.log,
                        )

            for multiplexed_qubits in qubits.batch():
                save_n_state = states[0]
                # Acquire the selected clouds in independent shot loops.
                if "g" in states:
                    with for_(n, 0, n < n_runs, n + 1):
                        if save_n_state == "g":
                            save(n, n_st)
                        for i, qubit in multiplexed_qubits.items():
                            reset_qubit(qubit, i)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            measure_cloud(qubit, I_g[i], Q_g[i])

                            save(I_g[i], I_g_st[i])
                            save(Q_g[i], Q_g_st[i])
                        align()

                if "e" in states:
                    with for_(n, 0, n < n_runs, n + 1):
                        if save_n_state == "e":
                            save(n, n_st)
                        for i, qubit in multiplexed_qubits.items():
                            reset_qubit(qubit, i)
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
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            measure_cloud(qubit, I_e[i], Q_e[i])
                            save(I_e[i], I_e_st[i])
                            save(Q_e[i], Q_e_st[i])
                        align()

                if "f" in states:
                    with for_(n, 0, n < n_runs, n + 1):
                        if save_n_state == "f":
                            save(n, n_st)
                        for i, qubit in multiplexed_qubits.items():
                            reset_qubit(qubit, i)
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
                        for i, qubit in multiplexed_qubits.items():
                            measure_cloud(qubit, I_f[i], Q_f[i])
                            save(I_f[i], I_f_st[i])
                            save(Q_f[i], Q_f_st[i])
                        align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    if "g" in states:
                        I_g_st[i].buffer(n_runs).save(f"Ig{i + 1}")
                        Q_g_st[i].buffer(n_runs).save(f"Qg{i + 1}")
                    if "e" in states:
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
                "readout_discriminator": getattr(
                    node.machine,
                    "readout_discriminator",
                    None,
                ),
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
                state_labels = [
                    str(state) for state in fit_result.get("state_labels", [])
                ]
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
                if state_labels in (
                    ["g", "e"],
                    ["g", "e", "f"],
                ):
                    centers = np.asarray(fit_result["center_matrix"], dtype=float)
                    if state_labels == ["g", "e"]:
                        centers = _rotate_iq_centers(
                            centers,
                            float(fit_result["iw_angle"]),
                        )
                    if np.isfinite(centers).all():
                        q.resonator.gef_centers = (
                            centers * operation.length / 2**12
                        ).tolist()
                    else:
                        node.log(
                            f"Skipping {q.name} IQ-center update because fitted centers are not finite."
                        )
                if state_labels != ["g", "e"]:
                    node.log(
                        f"Skipping {q.name} readout state update because acquired states "
                        f"were {state_labels}, not ['g', 'e']."
                    )
                    continue
                operation.integration_weights_angle -= float(fit_result["iw_angle"])
                # Convert the thresholds back to demod units
                operation.threshold = (
                    float(fit_result["ge_threshold"]) * operation.length / 2**12
                )
                operation.rus_exit_threshold = (
                    float(fit_result["rus_threshold"]) * operation.length / 2**12
                )
                if node.parameters.operation == "readout":
                    q.resonator.confusion_matrix = fit_result["fidelity_matrix"]

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
            state_labels = [str(state) for state in fit_result.get("state_labels", [])]
            if state_labels in (
                ["g", "e"],
                ["g", "e", "f"],
            ):
                centers = np.asarray(fit_result["center_matrix"], dtype=float)
                if state_labels == ["g", "e"]:
                    centers = _rotate_iq_centers(
                        centers,
                        float(fit_result["iw_angle"]),
                    )
                if np.isfinite(centers).all():
                    operation = q.resonator.operations["readout"]
                    updates[f"qubits.json.qubits.{q.name}.readout.gef_centers"] = (
                        centers * operation.length / 2**12
                    ).tolist()
                else:
                    node.log(
                        f"Profile IQ-center update skipped for {q.name}: fitted centers are not finite."
                    )
            if state_labels == ["g", "e", "f"]:
                continue
            if state_labels != ["g", "e"]:
                node.log(
                    f"Profile update skipped for {q.name}: acquired states "
                    f"were {state_labels}, not ['g', 'e']."
                )
                continue
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
            updates[f"qubits.json.qubits.{q.name}.readout.confusion_matrix"] = (
                fit_result["fidelity_matrix"]
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

    parameters.qubit_operation = "x180"
    parameters.states = ["g", "e"]
    parameters.reset_type = "thermal"
    # parameters.active_gef_reset_attempts = 3
    parameters.num_shots = 10000

    options = CalibrationOptions()
    # options.ai_review = True

    machine = create_machine(qubit="q6")

    calibration = IqBlobs(
        parameters=parameters,
        options=options,
        machine=machine,
    )
    calibration.run()
