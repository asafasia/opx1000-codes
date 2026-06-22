"""Class-based v2 migration for 06b_echo."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from calibration_io import CalibrationSaver, current_profile_name
from calibration_utils.T2echo import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from profiles import ProfileUpdater
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_idle_times_in_clock_cycles, get_qubits
from quam_config import Quam, create_machine
from utils.plotting_settings import plot_per_qubit
from utils.simulation import simulate_and_plot

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration, CalibrationOptions
else:
    from .base import BaseCalibration, CalibrationOptions


description = """
        T2 echo MEASUREMENT
The sequence consists in playing an echo sequence (x90 - idle_time - x180 - idle_time - -x90 - measurement) for
different idle times.
The qubit T2 echo is extracted by fitting the exponential decay of the measured quadratures/state.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit frequency precisely (node 06a_ramsey.py).
    - (optional) Having optimized the readout parameters (nodes 08a, 08b and 08c).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The qubit T2 echo: qubit.T2echo.
"""


class Echo(BaseCalibration[Parameters, Quam]):
    """v2 class migration for ``calibrations/06b_echo.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="06b_echo",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    def create_qua_program(self):
        node = self
        node.namespace["qubits"] = qubits = get_qubits(node)
        num_qubits = len(qubits)

        n_avg = node.parameters.num_shots
        idle_times = get_idle_times_in_clock_cycles(node.parameters)
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "idle_time": xr.DataArray(
                2 * 4 * idle_times,
                attrs={"long_name": "idle time", "units": "ns"},
            ),
        }

        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            idle_time = declare(int)

            if node.parameters.use_state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [declare_stream() for _ in range(num_qubits)]

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_each_(idle_time, idle_times):
                        for qubit in multiplexed_qubits.values():
                            reset_frame(qubit.xy.name)
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                            )
                        align()

                        for qubit in multiplexed_qubits.values():
                            with strict_timing_():

                                qubit.xy.play("x90")
                                qubit.xy.wait(idle_time)
                                qubit.xy.play("x180")
                                qubit.xy.wait(idle_time)
                                qubit.xy.play("-x90")
                                qubit.align()
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure(
                                    "readout", qua_vars=(I[i], Q[i])
                                )
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    if node.parameters.use_state_discrimination:
                        state_st[i].buffer(len(idle_times)).average().save(
                            f"state{i + 1}"
                        )
                    else:
                        I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                        Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")

        return node.namespace.get("qua_program")

    def simulate_qua_program(self):
        node = self
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        samples, fig, wf_report = simulate_and_plot(
            qmm, config, node.namespace["qua_program"], node.parameters
        )
        node.results["simulation"] = {
            "figure": fig,
            "wf_report": wf_report,
            "samples": samples,
        }

    def execute_qua_program(self):
        node = self
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

    def save_raw_results(self):
        node = self
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
        load_data_id = node.parameters.load_data_id
        node.load_from_id(node.parameters.load_data_id)
        node.parameters.load_data_id = load_data_id
        node.namespace["qubits"] = get_qubits(node)

    def analyse_data(self):
        node = self
        node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
        node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
        node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

        log_fitted_results(node.results["fit_results"], log_callable=node.log)
        node.outcomes = {
            qubit_name: ("successful" if fit_result["success"] else "failed")
            for qubit_name, fit_result in node.results["fit_results"].items()
        }

    def plot_data(self):
        node = self
        figures = plot_per_qubit(
            plot_raw_data_with_fit,
            node.results["ds_raw"],
            node.namespace["qubits"],
            node.results["ds_fit"],
            figure_name="raw_fit",
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
        with node.record_state_updates():
            for q in node.namespace["qubits"]:
                if node.outcomes[q.name] == "failed":
                    continue
                q.T2echo = node.results["fit_results"][q.name]["T2_echo"]

    def propose_profile_update(self):
        node = self
        updates = {
            f"metrics.json.qubits.{q.name}.coherence.t2_echo_ns": float(
                node.results["ds_fit"].sel(qubit=q.name).T2_echo.values
            )
            for q in node.namespace["qubits"]
            if node.outcomes[q.name] == "successful"
        }
        if updates:
            proposal = ProfileUpdater().stage(
                node.name,
                updates,
                profile_name=current_profile_name(),
            )
            ProfileUpdater().confirm_and_apply(proposal)


if __name__ == "__main__":
    parameters = Parameters()

    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.max_wait_time_in_ns = 3e3
    parameters.wait_time_num_points = 150
    parameters.log_or_linear_sweep = "linear"
    parameters.num_shots = 5000

    options = CalibrationOptions()

    calibration = Echo(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
