"""Class-based calibration for 02a_resonator_spectroscopy."""

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
from quam_config import Quam, create_machine
from calibration_utils.resonator_spectroscopy import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_iq_blobs_for_frequency,
    plot_raw_amplitude,
)
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from profiles import ProfileUpdater
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

if __package__ in {None, ""}:
    from calibrations.core import BaseCalibration, CalibrationOptions
else:
    from .core import BaseCalibration, CalibrationOptions

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


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
class ResonatorSpectroscopy(BaseCalibration[Parameters, Quam]):
    """Class-based calibration for ``calibrations/02a_resonator_spectroscopy.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="02a_resonator_spectroscopy",
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
        # Extract the sweep parameters and axes from the node parameters
        n_runs = node.parameters.num_shots
        states = list(node.parameters.states)
        valid_states = {"g", "e", "f"}
        if (
            len(states) not in (2, 3)
            or len(set(states)) != len(states)
            or set(states) - valid_states
        ):
            raise ValueError(
                "Resonator spectroscopy states must be a unique two-state pair from ['g', 'e', 'f'] or ['g', 'e', 'f']."
            )
        use_f_state = "f" in states
        selected_operation = node.parameters.qubit_operation
        qua_operation = (
            "x180" if selected_operation == "x180_const" else selected_operation
        )
        # The frequency sweep around the resonator resonance frequency
        span = node.parameters.frequency_span_in_mhz * u.MHz
        step = node.parameters.frequency_step_in_mhz * u.MHz
        dfs = np.arange(-span / 2, +span / 2, step)
        for qubit in qubits:
            if "e" in states and qua_operation not in qubit.xy.operations:
                raise ValueError(
                    f"{qubit.name} does not define qubit operation {qua_operation!r}."
                )
            if use_f_state:
                if "x180" not in qubit.xy.operations:
                    raise ValueError(
                        f"{qubit.name} does not define qubit operation 'x180'."
                    )
                if "EF_x180" not in qubit.xy.operations:
                    raise ValueError(
                        f"{qubit.name} does not define qubit operation 'EF_x180'."
                    )
                if getattr(qubit, "anharmonicity", None) is None:
                    raise ValueError(
                        f"{qubit.name} does not define anharmonicity required to prepare |f>."
                    )
            if "e" in states and selected_operation == "saturation":
                saturation_length = qubit.xy.operations["saturation"].length
                readout_length = qubit.resonator.operations["readout"].length
                required_length = (
                    node.parameters.saturation_lead_time_in_ns + readout_length
                )
                if saturation_length < required_length:
                    raise ValueError(
                        f"{qubit.name} saturation pulse is {saturation_length} ns, but at least "
                        f"{required_length} ns is required to cover the lead-in and readout."
                    )
        # Register the sweep axes to be added to the dataset when fetching data
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "n_runs": xr.DataArray(
                np.arange(n_runs), attrs={"long_name": "shot index"}
            ),
            "detuning": xr.DataArray(
                dfs, attrs={"long_name": "readout frequency", "units": "Hz"}
            ),
        }

        # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
        with program() as node.namespace["qua_program"]:
            Ig, Ig_st, Qg, Qg_st, n, n_st = node.machine.declare_qua_variables()
            Im, Im_st, Qm, Qm_st, _, _ = node.machine.declare_qua_variables()
            If, If_st, Qf, Qf_st, _, _ = node.machine.declare_qua_variables()
            df = declare(int)  # QUA variable for the readout frequency

            for multiplexed_qubits in qubits.batch():
                # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()
                with for_(n, 0, n < n_runs, n + 1):
                    save(n, n_st)

                    if "g" in states:
                        # Complete ground-state resonator spectroscopy scan.
                        with for_(*from_array(df, dfs)):
                            for i, qubit in multiplexed_qubits.items():
                                qubit.reset(
                                    "thermal",
                                    node.parameters.simulate,
                                    # log_callable=node.log,
                                )

                                rr = qubit.resonator
                                # Update the resonator frequencies for all resonators
                                rr.update_frequency(df + rr.intermediate_frequency)
                                # Measure the resonator
                                rr.measure("readout", qua_vars=(Ig[i], Qg[i]))
                                # wait for the resonator to deplete
                                # rr.wait(rr.depletion_time * u.ns)

                                save(Ig[i], Ig_st[i])
                                save(Qg[i], Qg_st[i])

                            align()

                    if "e" in states:
                        # Complete the driven-state resonator spectroscopy scan.
                        with for_(*from_array(df, dfs)):
                            for i, qubit in multiplexed_qubits.items():
                                qubit.reset(
                                    "thermal",
                                    node.parameters.simulate,
                                )

                                rr = qubit.resonator
                                rr.update_frequency(df + rr.intermediate_frequency)
                                if selected_operation == "saturation":
                                    align(qubit.xy.name, rr.name)
                                    qubit.xy.play(
                                        qua_operation,
                                        amplitude_scale=node.parameters.saturation_amplitude_factor,
                                    )
                                    rr.wait(
                                        node.parameters.saturation_lead_time_in_ns
                                        * u.ns
                                    )
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

                            align()

                    if use_f_state:
                        # Complete the f-state resonator spectroscopy scan.
                        with for_(*from_array(df, dfs)):
                            for i, qubit in multiplexed_qubits.items():
                                qubit.reset(
                                    "thermal",
                                    node.parameters.simulate,
                                )

                                rr = qubit.resonator
                                rr.update_frequency(df + rr.intermediate_frequency)
                                qubit.xy.play("x180")
                                update_frequency(
                                    qubit.xy.name,
                                    qubit.xy.intermediate_frequency
                                    - qubit.anharmonicity,
                                )
                                qubit.xy.play("EF_x180")
                                update_frequency(
                                    qubit.xy.name, qubit.xy.intermediate_frequency
                                )
                                qubit.align()
                                rr.measure("readout", qua_vars=(If[i], Qf[i]))
                                save(If[i], If_st[i])
                                save(Qf[i], Qf_st[i])

                            align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    if "g" in states:
                        Ig_st[i].buffer(len(dfs)).buffer(n_runs).save(f"Ig{i + 1}")
                        Qg_st[i].buffer(len(dfs)).buffer(n_runs).save(f"Qg{i + 1}")
                    if "e" in states:
                        Im_st[i].buffer(len(dfs)).buffer(n_runs).save(f"Im{i + 1}")
                        Qm_st[i].buffer(len(dfs)).buffer(n_runs).save(f"Qm{i + 1}")
                    if use_f_state:
                        If_st[i].buffer(len(dfs)).buffer(n_runs).save(f"If{i + 1}")
                        Qf_st[i].buffer(len(dfs)).buffer(n_runs).save(f"Qf{i + 1}")

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
        plt.show()

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

    def load_data(self):
        node = self
        """Load a previously acquired dataset."""
        load_data_id = node.parameters.load_data_id
        # Load the specified dataset
        node.load_from_id(node.parameters.load_data_id)
        node.parameters.load_data_id = load_data_id
        # Get the active qubits from the loaded node parameters
        node.namespace["qubits"] = get_qubits(node)

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
        """Plot mean resonator responses and shot-level IQ separation."""
        figures = plot_per_qubit(
            plot_raw_amplitude,
            node.results["ds_raw"],
            node.namespace["qubits"],
            figure_name="amplitude",
            qubit_operation=node.parameters.qubit_operation,
            saturation_amplitude_factor=node.parameters.saturation_amplitude_factor,
            saturation_lead_time_in_ns=node.parameters.saturation_lead_time_in_ns,
        )
        plt.show()
        node.results["figures"] = figures
        if "calibration_run_directory" in node.namespace:
            figures_directory = CalibrationSaver().save_figures(
                node.namespace["calibration_run_directory"],
                node.results["figures"],
            )
            node.log(f"Calibration figures saved to {figures_directory}")

    def plot_for_freq(self, frequency: float):
        """Plot shot-level IQ blobs at the nearest resonator-sweep frequency."""
        node = self
        if "ds_raw" not in node.results:
            raise ValueError(
                "No resonator spectroscopy data loaded. Run or load data first."
            )
        if "qubits" not in node.namespace:
            node.namespace["qubits"] = get_qubits(node)
        if "full_freq" not in node.results["ds_raw"].coords:
            node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)

        figure = plot_iq_blobs_for_frequency(
            node.results["ds_raw"],
            node.namespace["qubits"],
            frequency,
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()
        node.results.setdefault("figures", {})["frequency_iq_blobs"] = figure
        if "calibration_run_directory" in node.namespace:
            figures_directory = CalibrationSaver().save_figures(
                node.namespace["calibration_run_directory"],
                {"frequency_iq_blobs": figure},
            )
            node.log(f"Frequency IQ-blob figure saved to {figures_directory}")
        return figure

    def _plot_for_freq(self, frequency: float):
        """Backward-compatible notebook helper for plotting one frequency point."""
        return self.plot_for_freq(frequency)

    def propose_profile_update(self):
        node = self
        """Stage fitted resonator frequencies and apply them only after confirmation."""
        updates = {
            f"qubits.json.qubits.{q.name}.frequencies_hz.resonator": float(
                node.results["fit_results"][q.name]["frequency"]
            )
            for q in node.namespace["qubits"]
            if node.outcomes[q.name] == "successful"
        }
        if updates:
            proposal = ProfileUpdater().stage(
                node.name, updates, profile_name=current_profile_name()
            )
            ProfileUpdater().confirm_and_apply(proposal)


if __name__ == "__main__":
    parameters = Parameters()

    parameters.qubit_operation = "x180"
    parameters.num_shots = 200
    parameters.frequency_span_in_mhz = 20
    parameters.frequency_step_in_mhz = 0.1
    parameters.states = ["g", "f"]

    options = CalibrationOptions()
    # options.ai_review = True

    calibration = ResonatorSpectroscopy(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
    )
    calibration.run()

    # %%

    # calibration._plot_for_freq(
    #     frequency=calibration.results["fit_results"]["q1"]["frequency"]
    # )


# %%
