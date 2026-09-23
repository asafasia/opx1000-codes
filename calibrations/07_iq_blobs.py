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
from utils.qm_session import qm_session
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
from utils.simulation import simulate_and_plot
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
        node.namespace["qubits"] = qubits = self.get_qubits()
        num_qubits = len(qubits)

        n_runs = node.parameters.num_shots  # Number of runs
        operation = node.parameters.readout_operation
        node.parameters.operation = operation
        states = list(
            node.parameters.readout_states
            if node.parameters.states is None
            else node.parameters.states
        )
        node.namespace["prepared_states"] = states
        reset_type = node.parameters.reset_type
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
        if node.parameters.active_gef_reset_attempts is not None:
            if node.parameters.active_gef_reset_attempts < 1:
                raise ValueError("active_gef_reset_attempts must be positive")
            node.parameters.active_reset_max_attempts = (
                node.parameters.active_gef_reset_attempts
            )
        for qubit in qubits:
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
        # Register the sweep axes to be added to the dataset when fetching data
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "n_runs": xr.DataArray(
                np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}
            ),
        }

        self.configure_acquisition(shot_axis="n_runs", preserve_shots=True)

        with program() as node.namespace["qua_program"]:
            I_g, I_g_st, Q_g, Q_g_st, n, n_st = node.machine.declare_qua_variables()
            I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.declare_qua_variables()
            if "f" in states:
                I_f, I_f_st, Q_f, Q_f_st, _, _ = node.machine.declare_qua_variables()

            def measure_cloud(qubit, i_quadrature, q_quadrature):
                self.measure_readout(qubit, qua_vars=(i_quadrature, q_quadrature))

            def reset_qubit(qubit, qubit_index):
                self.reset_qubit(qubit)

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
                        self.save_acquisition_stream(I_g_st[i], f'Ig{i + 1}')
                        self.save_acquisition_stream(Q_g_st[i], f'Qg{i + 1}')
                    if "e" in states:
                        self.save_acquisition_stream(I_e_st[i], f'Ie{i + 1}')
                        self.save_acquisition_stream(Q_e_st[i], f'Qe{i + 1}')
                    if "f" in states:
                        self.save_acquisition_stream(I_f_st[i], f'If{i + 1}')
                        self.save_acquisition_stream(Q_f_st[i], f'Qf{i + 1}')

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
            # Wait for complete buffers while displaying the shot counter.
            dataset = self.fetch_result_dataset(job)
        # Register the raw dataset
        node.results["ds_raw"] = self.annotate_readout_dataset(dataset)
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
        node.namespace["qubits"] = self.get_qubits()

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
                "states": node.namespace.get("prepared_states", node.parameters.states),
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

    def _fitted_readout_settings(self, q):
        """Convert centers into the demodulation frame of this pulse only."""
        fit = self.results["fit_results"][q.name]
        states = list(fit.get("state_labels", []))
        expected = list(self.parameters.readout_states)
        if states != expected:
            self.log(
                f"{q.name}: no readout update proposed: prepared states {states} "
                f"do not match readout_states={expected} for "
                f"{self.parameters.readout_operation}. Set states=None to prepare "
                "the selected readout states; use readout_states=['g', 'e', 'f'] "
                "when calibrating the GEF pulse."
            )
            return None
        operation = q.resonator.operations[self.parameters.readout_operation]
        centers = np.asarray(fit["center_matrix"], dtype=float)
        if centers.shape != (len(states), 2) or not np.isfinite(centers).all():
            self.log(
                f"{q.name}: no readout update proposed: fitted IQ centers are missing or invalid."
            )
            return None
        values = {}
        if states == ["g", "e"]:
            if not all(
                np.isfinite(fit[key])
                for key in ("iw_angle", "ge_threshold", "rus_threshold")
            ):
                return None
            centers = _rotate_iq_centers(centers, float(fit["iw_angle"]))
            values.update(
                integration_weights_angle_rad=float(operation.integration_weights_angle)
                - float(fit["iw_angle"]),
                threshold=float(fit["ge_threshold"]) * operation.length / 2**12,
                rus_exit_threshold=float(fit["rus_threshold"])
                * operation.length
                / 2**12,
            )
        values["gef_centers"] = (centers * operation.length / 2**12).tolist()
        # Nearest-center assignment differs from the optimized binary threshold.
        matrix = (
            fit.get("confusion_matrix")
            if (
                len(states) == 3
                or getattr(q.resonator, "readout_discriminator", "quam")
                == "nearest_center"
            )
            else fit.get("fidelity_matrix")
        )
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape == (len(states), len(states)) and np.isfinite(matrix).all():
                values["confusion_matrix"] = matrix.tolist()
        from utils.readout_macro import readout_signature

        values["calibration_signature"] = readout_signature(
            q,
            self.parameters.readout_operation,
            angle=values.get("integration_weights_angle_rad"),
        )
        return values

    def update_state(self):
        from utils.readout_macro import readout_settings

        self.namespace["fitted_readout_settings"] = {}
        for q in self.namespace["qubits"]:
            values = self._fitted_readout_settings(q)
            if values is None:
                continue
            self.namespace["fitted_readout_settings"][q.name] = values
            settings = readout_settings(q, self.parameters.readout_operation)
            settings.update(values)
            operation = q.resonator.operations[self.parameters.readout_operation]
            if "integration_weights_angle_rad" in values:
                operation.integration_weights_angle = values[
                    "integration_weights_angle_rad"
                ]
                operation.threshold = values["threshold"]
                operation.rus_exit_threshold = values["rus_exit_threshold"]
            if self.parameters.readout_operation == "readout":
                q.resonator.gef_centers = values["gef_centers"]
                q.resonator.confusion_matrix = values.get("confusion_matrix")

    def profile_updates(self):
        def fitted_settings(q, fit):
            values = self.namespace.get("fitted_readout_settings", {}).get(q.name)
            return self._fitted_readout_settings(q) if values is None else values

        return self.readout_profile_updates(
            settings=fitted_settings, fidelity="readout_fidelity",
            successful_only=False,  # _fitted_readout_settings validates the acquired states and centers.
        )


if __name__ == "__main__":
    parameters = Parameters()
    parameters.center_method = "median"  # "median" or "mean" for IQ cloud centers.

    parameters.qubit_operation = "x180"
    parameters.readout_states = ["g", "e", "f"]  # Selects the readout_GEF pulse.
    parameters.states = None  # Prepare the same states as readout_states.
    parameters.reset_type = "active"  # Use "active" after GEF centers are calibrated.
    # parameters.active_gef_reset_attempts = 3

    parameters.num_shots = 10000

    options = CalibrationOptions(
        propose_profile_update=True,
        apply_profile_update=True,  # Ask for explicit "yes" before applying.
    )
    # options.ai_review = True

    machine = create_machine(qubit="q6")

    calibration = IqBlobs(
        parameters=parameters,
        options=options,
        machine=machine,
    )
    calibration.run()
