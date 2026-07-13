"""2D readout optimization over frequency and amplitude."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from calibration_io import CalibrationSaver, current_profile_name
from calibration_utils.readout_frequency_amplitude_optimization import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_optimization_maps,
    process_raw_dataset,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from quam_config import Quam, create_machine
from utils.simulation import simulate_and_plot

if __package__ in {None, ""}:
    from calibrations.core import BaseCalibration, CalibrationOptions
else:
    from .core import BaseCalibration, CalibrationOptions


description = """
        READOUT FREQUENCY-AMPLITUDE OPTIMIZATION
This sequence prepares |g> and |e> clouds while sweeping both readout frequency and
readout pulse amplitude. It produces 2D maps of state-center difference and
single-shot fidelity.
"""


class ReadoutFrequencyAmplitudeOptimization(BaseCalibration[Parameters, Quam]):
    """Shot-level readout optimization over frequency and amplitude."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="08c_readout_frequency_amplitude_optimization",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    def create_qua_program(self):
        node = self
        u = unit(coerce_to_integer=True)
        node.namespace["qubits"] = qubits = get_qubits(node)
        num_qubits = len(qubits)

        n_runs = node.parameters.num_shots
        span = node.parameters.frequency_span_in_mhz * u.MHz
        step = node.parameters.frequency_step_in_mhz * u.MHz
        dfs = np.arange(-span / 2, +span / 2, step)
        amps = np.linspace(
            node.parameters.start_amp,
            node.parameters.end_amp,
            node.parameters.num_amps,
        )
        selected_qubit_operation = node.parameters.qubit_operation
        qua_qubit_operation = (
            "x180"
            if selected_qubit_operation == "x180_const"
            else selected_qubit_operation
        )

        for qubit in qubits:
            if qua_qubit_operation not in qubit.xy.operations:
                raise ValueError(
                    f"{qubit.name} does not define qubit operation {qua_qubit_operation!r}."
                )

        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "n_runs": xr.DataArray(
                np.arange(n_runs), attrs={"long_name": "shot index"}
            ),
            "detuning": xr.DataArray(
                dfs, attrs={"long_name": "readout detuning", "units": "Hz"}
            ),
            "amp_prefactor": xr.DataArray(
                amps, attrs={"long_name": "readout amplitude prefactor", "units": ""}
            ),
        }

        with program() as node.namespace["qua_program"]:
            Ig, Ig_st, Qg, Qg_st, n, n_st = node.machine.declare_qua_variables()
            Ie, Ie_st, Qe, Qe_st, _, _ = node.machine.declare_qua_variables()
            df = declare(int)
            a = declare(fixed)

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_runs, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, dfs)):
                        for qubit in multiplexed_qubits.values():
                            qubit.resonator.update_frequency(
                                df + qubit.resonator.intermediate_frequency
                            )

                        with for_(*from_array(a, amps)):
                            for qubit in multiplexed_qubits.values():
                                qubit.reset(
                                    node.parameters.reset_type,
                                    node.parameters.simulate,
                                )
                            align()
                            for i, qubit in multiplexed_qubits.items():
                                qubit.resonator.measure(
                                    "readout",
                                    qua_vars=(Ig[i], Qg[i]),
                                    amplitude_scale=a,
                                )
                                save(Ig[i], Ig_st[i])
                                save(Qg[i], Qg_st[i])

                            for qubit in multiplexed_qubits.values():
                                qubit.reset(
                                    node.parameters.reset_type,
                                    node.parameters.simulate,
                                )
                            align()
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play(
                                    qua_qubit_operation,
                                    amplitude_scale=node.parameters.qubit_amplitude_factor,
                                )
                            align()
                            for i, qubit in multiplexed_qubits.items():
                                qubit.resonator.measure(
                                    "readout",
                                    qua_vars=(Ie[i], Qe[i]),
                                    amplitude_scale=a,
                                )
                                save(Ie[i], Ie_st[i])
                                save(Qe[i], Qe_st[i])
                            align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    Ig_st[i].buffer(len(amps)).buffer(len(dfs)).buffer(n_runs).save(
                        f"Ig{i + 1}"
                    )
                    Qg_st[i].buffer(len(amps)).buffer(len(dfs)).buffer(n_runs).save(
                        f"Qg{i + 1}"
                    )
                    Ie_st[i].buffer(len(amps)).buffer(len(dfs)).buffer(n_runs).save(
                        f"Ie{i + 1}"
                    )
                    Qe_st[i].buffer(len(amps)).buffer(len(dfs)).buffer(n_runs).save(
                        f"Qe{i + 1}"
                    )

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
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )
            node.log(job.execution_report())
        node.results["ds_raw"] = dataset

    def save_raw_results(self):
        output_directory = CalibrationSaver().save_xarray(
            self.name,
            self.results["ds_raw"],
            profile_name=current_profile_name(),
            parameters=self.parameters,
        )
        self.namespace["calibration_run_directory"] = output_directory
        self.log(f"Raw calibration results saved to {output_directory}")

    def load_data(self):
        load_data_id = self.parameters.load_data_id
        self.load_from_id(self.parameters.load_data_id)
        self.parameters.load_data_id = load_data_id
        self.namespace["qubits"] = get_qubits(self)

    def analyse_data(self):
        self.results["ds_raw"] = process_raw_dataset(self.results["ds_raw"], self)
        self.results["ds_fit"], fit_results = fit_raw_data(self.results["ds_raw"], self)
        self.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
        log_fitted_results(self.results["fit_results"], log_callable=self.log)
        self.outcomes = {
            qubit_name: ("successful" if fit_result["success"] else "failed")
            for qubit_name, fit_result in self.results["fit_results"].items()
        }

    def plot_data(self):
        self.results["figures"] = plot_optimization_maps(
            self.results["ds_raw"],
            self.namespace["qubits"],
            self.results["ds_fit"],
        )
        plt.show()
        if "calibration_run_directory" in self.namespace:
            figures_directory = CalibrationSaver().save_figures(
                self.namespace["calibration_run_directory"],
                self.results["figures"],
            )
            self.log(f"Calibration figures saved to {figures_directory}")


if __name__ == "__main__":
    parameters = Parameters()
    parameters.reset_type = "thermal"
    parameters.num_shots = 100
    parameters.frequency_span_in_mhz = 20
    parameters.frequency_step_in_mhz = 0.2
    parameters.start_amp = 0.1
    parameters.end_amp = 2.0
    parameters.num_amps = 10

    options = CalibrationOptions(
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
    )

    calibration = ReadoutFrequencyAmplitudeOptimization(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q7"),
    )
    calibration.run()
