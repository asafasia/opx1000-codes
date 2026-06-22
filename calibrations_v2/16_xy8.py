"""Class-based v2 migration for XY8 dynamical decoupling."""

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
from calibration_utils.xy8 import (
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
    from calibrations_v2.base import BaseCalibration, CalibrationOptions
else:
    from .base import BaseCalibration, CalibrationOptions


description = """
        XY8 DYNAMICAL DECOUPLING MEASUREMENT
The program plays an XY8 sequence to measure qubit coherence under dynamical decoupling.
Each XY8 cycle uses the 8-pulse pattern X-Y-X-Y-Y-X-Y-X. The calibration sweeps both
the delay tau between pi pulses and the number of XY8 cycles N, then fits a decay for
each N to extract T2_XY8(N).

Prerequisites:
    - Calibrated resonator readout.
    - Calibrated x90, -x90, x180 and y180 gates.
    - Precisely calibrated qubit frequency, usually from Ramsey.
"""


class XY8(BaseCalibration[Parameters, Quam]):
    """XY8 dynamical-decoupling calibration in the v2 class style."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="16_xy8",
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
        evolution_times = get_idle_times_in_clock_cycles(node.parameters)
        n_xy8_values = np.asarray(node.parameters.n_xy8_values, dtype=int)
        if n_avg <= 0:
            raise ValueError("num_shots must be positive.")
        if evolution_times.size == 0:
            raise ValueError("XY8 evolution-time sweep must contain at least one point.")
        if n_xy8_values.size == 0 or np.any(n_xy8_values <= 0):
            raise ValueError("n_xy8_values must contain positive integers.")

        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "n_xy8": xr.DataArray(
                n_xy8_values,
                attrs={"long_name": "XY8 cycles"},
            ),
            "evolution_time": xr.DataArray(
                4 * evolution_times,
                attrs={"long_name": "total evolution time", "units": "ns"},
            ),
        }

        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            evolution_time = declare(int)
            pulse_delay = declare(int)
            edge_delay = declare(int)
            n_xy8 = declare(int)
            cycle = declare(int)

            if node.parameters.use_state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [declare_stream() for _ in range(num_qubits)]

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_each_(n_xy8, n_xy8_values):
                        with for_each_(evolution_time, evolution_times):
                            assign(pulse_delay, evolution_time / (8 * n_xy8))
                            assign(edge_delay, pulse_delay / 2)
                            for qubit in multiplexed_qubits.values():
                                reset_frame(qubit.xy.name)
                                qubit.reset(
                                    node.parameters.reset_type,
                                    node.parameters.simulate,
                                )
                            align()

                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x90")
                                with for_(cycle, 0, cycle < n_xy8, cycle + 1):
                                    qubit.xy.wait(edge_delay)
                                    qubit.xy.play("x180")
                                    qubit.xy.wait(pulse_delay)
                                    qubit.xy.play("y180")
                                    qubit.xy.wait(pulse_delay)
                                    qubit.xy.play("x180")
                                    qubit.xy.wait(pulse_delay)
                                    qubit.xy.play("y180")
                                    qubit.xy.wait(pulse_delay)
                                    qubit.xy.play("y180")
                                    qubit.xy.wait(pulse_delay)
                                    qubit.xy.play("x180")
                                    qubit.xy.wait(pulse_delay)
                                    qubit.xy.play("y180")
                                    qubit.xy.wait(pulse_delay)
                                    qubit.xy.play("x180")
                                    qubit.xy.wait(edge_delay)
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
                            len(n_xy8_values)
                        ).average().save(f"state{i + 1}")
                    else:
                        I_st[i].buffer(len(evolution_times)).buffer(
                            len(n_xy8_values)
                        ).average().save(f"I{i + 1}")
                        Q_st[i].buffer(len(evolution_times)).buffer(
                            len(n_xy8_values)
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
                figure_name="t2_vs_xy8_order",
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
        """XY8 reports driven-decoupling coherence and does not overwrite T2echo."""

    def propose_profile_update(self):
        """No profile field currently represents T2_XY8(N)."""
        return False


if __name__ == "__main__":
    parameters = Parameters()

    parameters.num_shots = 1000
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.min_wait_time_in_ns = 16
    parameters.max_wait_time_in_ns = 2000
    parameters.wait_time_num_points = 100
    parameters.log_or_linear_sweep = "linear"
    parameters.n_xy8_values = [1, 2, 4, 8]

    options = CalibrationOptions()

    calibration = XY8(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
