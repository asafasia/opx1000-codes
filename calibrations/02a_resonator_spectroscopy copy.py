# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam, create_machine
from calibration_utils.resonator_spectroscopy import Parameters
from saver import CalibrationSaver, current_profile_name
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from utils.plotting_settings import FIGURE_SIZE


# %% {Node initialisation}
description = """
        FIXED-FREQUENCY RESONATOR IQ BLOBS

This sequence measures the resonator at the current readout frequency only.
It does not perform a frequency sweep.

For each shot:
    1. Measure the resonator with the qubit in the ground state.
    2. Apply the selected qubit operation.
    3. Measure the resonator again.
    4. Store the ground and driven-state IQ points.

This is useful for checking the IQ separation between the ground state and the driven state.
"""

node = QualibrationNode[Parameters, Quam](
    name="02a_resonator_fixed_frequency_iq",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubits = ["q9"]
    node.parameters.qubit_operation = "x180"
    node.parameters.num_shots = 10000
    # node.parameters.simulate = True


node.machine = create_machine()
node.machine.connect()
node.machine.qmm.close_all_qms()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)

    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_runs = node.parameters.num_shots
    selected_operation = node.parameters.qubit_operation
    qua_operation = "x180" if selected_operation == "x180_const" else selected_operation

    for qubit in qubits:
        if qua_operation not in qubit.xy.operations:
            raise ValueError(f"{qubit.name} does not define qubit operation {qua_operation!r}.")

        if selected_operation == "saturation":
            saturation_length = qubit.xy.operations["saturation"].length
            readout_length = qubit.resonator.operations["readout"].length
            required_length = node.parameters.saturation_lead_time_in_ns + readout_length

            if saturation_length < required_length:
                raise ValueError(
                    f"{qubit.name} saturation pulse is {saturation_length} ns, "
                    f"but at least {required_length} ns is required."
                )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.arange(n_runs), attrs={"long_name": "shot index"}),
    }

    with program() as node.namespace["qua_program"]:
        Ig, Ig_st, Qg, Qg_st, n, n_st = node.machine.declare_qua_variables()
        Im, Im_st, Qm, Qm_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)

                # Ground-state readout
                for i, qubit in multiplexed_qubits.items():
                    rr = qubit.resonator

                    rr.measure("readout", qua_vars=(Ig[i], Qg[i]))
                    rr.wait(rr.depletion_time * u.ns)

                    save(Ig[i], Ig_st[i])
                    save(Qg[i], Qg_st[i])

                align()

                # Driven-state readout
                for i, qubit in multiplexed_qubits.items():
                    rr = qubit.resonator

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
                    rr.wait(200 * u.us)

                    save(Im[i], Im_st[i])
                    save(Qm[i], Qm_st[i])

                    qubit.reset_qubit_thermal()

                align()

        with stream_processing():
            n_st.save("n")

            for i in range(num_qubits):
                Ig_st[i].buffer(n_runs).save(f"Ig{i + 1}")
                Qg_st[i].buffer(n_runs).save(f"Qg{i + 1}")
                Im_st[i].buffer(n_runs).save(f"Im{i + 1}")
                Qm_st[i].buffer(n_runs).save(f"Qm{i + 1}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()

    samples, fig, wf_report = simulate_and_plot(
        qmm,
        config,
        node.namespace["qua_program"],
        node.parameters,
    )

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
    qmm = node.machine.connect()
    config = node.machine.generate_config()

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])

        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])

        dataset = None
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )

        node.log(job.execution_report())

    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id

    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Save_raw_results}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def save_raw_results(node: QualibrationNode[Parameters, Quam]):
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
    ds = node.results["ds_raw"]

    results = {}

    for qubit_index, qubit in enumerate(node.namespace["qubits"]):
        Ig = np.asarray(ds.Ig[qubit_index])
        Qg = np.asarray(ds.Qg[qubit_index])
        Im = np.asarray(ds.Im[qubit_index])
        Qm = np.asarray(ds.Qm[qubit_index])

        ground = Ig + 1j * Qg
        driven = Im + 1j * Qm

        ground_mean = np.mean(ground)
        driven_mean = np.mean(driven)

        separation = abs(driven_mean - ground_mean)

        results[qubit.name] = {
            "ground_mean_I": float(ground_mean.real),
            "ground_mean_Q": float(ground_mean.imag),
            "driven_mean_I": float(driven_mean.real),
            "driven_mean_Q": float(driven_mean.imag),
            "iq_separation": float(separation),
        }

        node.log(f"{qubit.name}: IQ separation = {separation:.5g}")

    node.results["iq_results"] = results
    node.outcomes = {
        qubit.name: "successful"
        for qubit in node.namespace["qubits"]
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    ds = node.results["ds_raw"]

    figures = {}

    for qubit_index, qubit in enumerate(node.namespace["qubits"]):
        Ig = np.asarray(ds.Ig[qubit_index])
        Qg = np.asarray(ds.Qg[qubit_index])
        Im = np.asarray(ds.Im[qubit_index])
        Qm = np.asarray(ds.Qm[qubit_index])

        ground_mean = np.mean(Ig) + 1j * np.mean(Qg)
        driven_mean = np.mean(Im) + 1j * np.mean(Qm)

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        ax.plot(Ig, Qg, ".", label="Ground", alpha=0.3, color="blue", markersize=4)
        ax.plot(Im, Qm, ".", label="Driven", alpha=0.3, color="orange",markersize=4)

        ax.plot(
            ground_mean.real,
            ground_mean.imag,
            "bx",
            label="Ground mean",
            markersize=10,
        )

        ax.plot(
            driven_mean.real,
            driven_mean.imag,
            "rx",
            label="Driven mean",
            markersize=10,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title(f"{qubit.name} fixed-frequency IQ blobs")
        ax.legend()

        figures[f"{qubit.name}_iq_blobs"] = fig

        plt.show()

    node.results["figures"] = figures

    if "calibration_run_directory" in node.namespace:
        figures_directory = CalibrationSaver().save_figures(
            node.namespace["calibration_run_directory"],
            node.results["figures"],
        )

        node.log(f"Calibration figures saved to {figures_directory}")
# %%
