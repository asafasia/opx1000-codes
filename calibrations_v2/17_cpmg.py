"""Class-based v2 migration for CPMG dynamical decoupling."""

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

from calibration_io import CalibrationSaver, current_profile_name
from calibration_utils.cpmg import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    plot_t2_vs_order,
    process_raw_dataset,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_idle_times_in_clock_cycles, get_qubits
from quam_config import Quam, create_machine
from utils.plotting_settings import plot_per_qubit
from utils.simulation import simulate_and_plot

if __package__ in {None, ""}:
    from calibrations_v2.core import BaseCalibration, CalibrationOptions
else:
    from .core import BaseCalibration, CalibrationOptions


description = """
        CPMG DYNAMICAL DECOUPLING MEASUREMENT
The program plays a Carr-Purcell-Meiboom-Gill sequence:
x90 - [tau - y180 - tau] x N - -x90 - measurement.

The calibration sweeps the total evolution time and the number of refocusing pi
pulses N. For each N, tau is computed as total_evolution_time / (2N), then the
decay is fitted to extract T2_CPMG(N).

Prerequisites:
    - Calibrated resonator readout.
    - Calibrated x90, -x90 and y180 gates.
    - Precisely calibrated qubit frequency, usually from Ramsey.
"""


class CPMG(BaseCalibration[Parameters, Quam]):
    """CPMG dynamical-decoupling calibration in the v2 class style."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="17_cpmg",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    def create_qua_program(self):
        node = self
        node.namespace["qubits"] = qubits = get_qubits(node)
        num_qubits = len(qubits)

        n_avg = int(node.parameters.num_shots)
        requested_evolution_times = get_idle_times_in_clock_cycles(node.parameters)
        n_pi_values = np.asarray(node.parameters.n_pi_values, dtype=int)
        if n_avg <= 0:
            raise ValueError("num_shots must be positive.")
        if requested_evolution_times.size == 0:
            raise ValueError(
                "CPMG evolution-time sweep must contain at least one point."
            )
        if n_pi_values.size == 0 or np.any(n_pi_values <= 0):
            raise ValueError("n_pi_values must contain positive integers.")
        evolution_time_divisor = int(np.lcm.reduce(2 * n_pi_values))
        evolution_times = np.unique(
            (requested_evolution_times // evolution_time_divisor)
            * evolution_time_divisor
        ).astype(int)
        evolution_times = evolution_times[
            evolution_times >= 2 * int(np.max(n_pi_values)) * 4
        ]
        if evolution_times.size == 0:
            raise ValueError(
                "CPMG evolution-time sweep is empty after quantizing to values "
                "compatible with all n_pi_values. Increase min_wait_time_in_ns."
            )
        if int(np.min(evolution_times)) // (2 * int(np.max(n_pi_values))) < 4:
            raise ValueError(
                "CPMG tau would be shorter than 4 clock cycles. Increase "
                "min_wait_time_in_ns or reduce the largest n_pi_values entry."
            )

        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "n_pi": xr.DataArray(
                n_pi_values,
                attrs={"long_name": "CPMG pi pulses"},
            ),
            "evolution_time": xr.DataArray(
                4 * evolution_times,
                attrs={"long_name": "total evolution time", "units": "ns"},
            ),
        }

        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            tau = declare(int)
            tau_between_pulses = declare(int)

            if node.parameters.use_state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [declare_stream() for _ in range(num_qubits)]

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    for n_pi_value in n_pi_values:
                        tau_values = (evolution_times // (2 * int(n_pi_value))).astype(
                            int
                        )
                        with for_each_(tau, tau_values):
                            assign(tau_between_pulses, 2 * tau)
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
                                    qubit.xy.wait(tau)
                                    for pulse_index in range(int(n_pi_value)):
                                        qubit.xy.play("y180")
                                        if pulse_index < int(n_pi_value) - 1:
                                            qubit.xy.wait(tau_between_pulses)
                                        else:
                                            qubit.xy.wait(tau)
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
                        state_st[i].buffer(len(evolution_times)).buffer(
                            len(n_pi_values)
                        ).average().save(f"state{i + 1}")
                    else:
                        I_st[i].buffer(len(evolution_times)).buffer(
                            len(n_pi_values)
                        ).average().save(f"I{i + 1}")
                        Q_st[i].buffer(len(evolution_times)).buffer(
                            len(n_pi_values)
                        ).average().save(f"Q{i + 1}")

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
            qubit: ("successful" if any(result["success"].values()) else "failed")
            for qubit, result in node.results["fit_results"].items()
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
        figures.update(
            plot_per_qubit(
                plot_t2_vs_order,
                node.results["ds_raw"],
                node.namespace["qubits"],
                node.results["ds_fit"],
                figure_name="t2_vs_cpmg_order",
            )
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
        """CPMG reports driven-decoupling coherence and does not overwrite T2echo."""

    def propose_profile_update(self):
        """No profile field currently represents T2_CPMG(N)."""
        return False


if __name__ == "__main__":
    parameters = Parameters()

    parameters.num_shots = 1000
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.min_wait_time_in_ns = 512
    parameters.max_wait_time_in_ns = 50_000
    parameters.wait_time_num_points = 200
    parameters.log_or_linear_sweep = "linear"
    parameters.n_pi_values = [1, 2, 4, 8]

    options = CalibrationOptions()

    calibration = CPMG(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
