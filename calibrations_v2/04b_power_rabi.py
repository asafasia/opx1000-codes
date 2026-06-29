"""Class-based Power Rabi calibration."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from calibration_utils.analysis_base import BaseAnalysis
from calibration_utils.power_rabi import (
    Parameters,
    fit_raw_data,
    get_number_of_pulses,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from profiles import load_profile
from quam_config import Quam, create_machine
from utils.plotting_settings import plot_per_qubit

if __package__ in {None, ""}:
    from calibrations_v2.core import BaseCalibration, CalibrationOptions
else:
    from .core import BaseCalibration, CalibrationOptions

DESCRIPTION = """
        POWER RABI
This sequence calibrates either the GE or EF transition. For transition="ge", it repeatedly executes
the selected GE qubit pulse across amplitude and pulse-count sweeps. For transition="ef", it first
prepares |e> with x180 and calibrates EF_x180 across amplitude.

State update:
    - GE: the pulse amplitude corresponding to the specified operation.
    - EF: the EF_x180 pulse amplitude.
"""


def validate_readout_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> None:
    """Ensure fetched results match the requested readout mode."""
    variables = set(ds.data_vars)
    expected = {"state"} if use_state_discrimination else {"I", "Q"}
    unexpected = {"I", "Q"} if use_state_discrimination else {"state"}
    missing = expected - variables
    present_unexpected = unexpected & variables
    if missing or present_unexpected:
        raise RuntimeError(
            "Rabi readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(variables)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


def active_operation(parameters: Parameters) -> str:
    """Return the operation calibrated by the selected transition."""
    return "EF_x180" if parameters.transition == "ef" else parameters.operation


def has_gef_readout_calibration(qubit: Any) -> bool:
    """Return whether the qubit has the data needed for dedicated GEF readout."""
    return (
        callable(getattr(qubit, "readout_state_gef", None))
        and getattr(qubit.resonator, "GEF_frequency_shift", None) is not None
    )


def ensure_operation_available(qubit: Any, operation: str, transition: str) -> None:
    """Validate GE operations and provide a default EF_x180 pulse when missing."""
    if operation in qubit.xy.operations:
        return
    if transition == "ef" and operation == "EF_x180":
        x180 = qubit.xy.operations["x180"]
        qubit.xy.operations["EF_x180"] = (
            dataclasses.replace(x180, alpha=0.0)
            if hasattr(x180, "alpha")
            else dataclasses.replace(x180)
        )
        return
    raise ValueError(f"{qubit.name} does not define operation {operation!r}.")


class PowerRabi(BaseCalibration[Parameters, Quam]):
    """Power Rabi calibration implemented with the v2 base lifecycle."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        *,
        name: str = "04b_power_rabi",
        profile_name: str | None = None,
        qubit: str | None = None,
        auto_connect: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            description=DESCRIPTION,
            parameters=parameters,
            machine=machine,
            profile_name=profile_name,
            qubit=qubit,
            auto_connect=auto_connect,
            **kwargs,
        )

    def create_analysis(self):
        return PowerRabiAnalysis(self)

    def create_qua_program(self):
        """Create the sweep axes and generate the QUA program."""
        qubits = self.get_qubits()
        num_qubits = len(qubits)
        n_avg = self.parameters.num_shots
        operation = active_operation(self.parameters)
        for qubit in qubits:
            ensure_operation_available(qubit, operation, self.parameters.transition)

        amps = np.arange(
            self.parameters.min_amp_factor,
            self.parameters.max_amp_factor,
            self.parameters.amp_factor_step,
        )
        n_pi_vec = get_number_of_pulses(self.parameters)
        self.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "nb_of_pulses": xr.DataArray(
                n_pi_vec,
                attrs={"long_name": "number of pulses"},
            ),
            "amp_prefactor": xr.DataArray(
                amps,
                attrs={"long_name": "pulse amplitude prefactor"},
            ),
        }

        with program() as qua_program:
            I, I_st, Q, Q_st, n, n_st = self.machine.declare_qua_variables()
            if self.parameters.use_state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [declare_stream() for _ in range(num_qubits)]
            a = declare(fixed)
            npi = declare(int)
            count = declare(int)

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    self.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(npi, n_pi_vec)):
                        with for_(*from_array(a, amps)):
                            for _, qubit in multiplexed_qubits.items():
                                qubit.reset(
                                    self.parameters.reset_type,
                                    self.parameters.simulate,
                                )

                            align()

                            for _, qubit in multiplexed_qubits.items():
                                if self.parameters.transition == "ef":
                                    qubit.xy.update_frequency(
                                        qubit.xy.intermediate_frequency
                                    )
                                    qubit.xy.play("x180")
                                    qubit.xy.update_frequency(
                                        qubit.xy.intermediate_frequency
                                        - qubit.anharmonicity
                                    )
                                    with for_(count, 0, count < npi, count + 1):
                                        qubit.xy.play("EF_x180", amplitude_scale=a)
                                else:
                                    with for_(count, 0, count < npi, count + 1):
                                        qubit.xy.play(operation, amplitude_scale=a)
                            align()

                            for i, qubit in multiplexed_qubits.items():
                                if self.parameters.use_state_discrimination:
                                    if self.parameters.transition == "ef":
                                        if has_gef_readout_calibration(qubit):
                                            qubit.readout_state_gef(state[i])
                                        else:
                                            qubit.readout_state(state[i])
                                    else:
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
                for i, _ in enumerate(qubits):
                    if operation.endswith("x180") or operation.startswith("x180_"):
                        self._save_streams(
                            i,
                            (
                                state_st
                                if self.parameters.use_state_discrimination
                                else None
                            ),
                            I_st,
                            Q_st,
                            amps,
                            n_pi_vec,
                        )
                    elif operation in ["x90", "-x90", "y90", "-y90"]:
                        self._save_streams(
                            i,
                            (
                                state_st
                                if self.parameters.use_state_discrimination
                                else None
                            ),
                            I_st,
                            Q_st,
                            amps,
                            n_pi_vec,
                        )
                    else:
                        raise ValueError(f"Unrecognized operation {operation}.")

        self.namespace["qua_program"] = qua_program
        return qua_program

    def execute_qua_program(self) -> None:
        super().execute_qua_program()
        validate_readout_dataset(
            self.results["ds_raw"],
            self.parameters.use_state_discrimination,
        )

    def plot_data(self) -> None:
        figures = plot_per_qubit(
            plot_raw_data_with_fit,
            self.results["ds_raw"],
            self.namespace["qubits"],
            self.results["ds_fit"],
            figure_name="amplitude",
            use_state_discrimination=self.parameters.use_state_discrimination,
        )
        plt.show()
        self.results["figures"] = figures

    def profile_updates(self) -> dict[str, float]:
        updates = {}
        profile_name = self.active_profile_name()
        qubit_profiles = load_profile(profile_name)["qubits"]["qubits"]
        operation = active_operation(self.parameters)
        for q in self.namespace["qubits"]:
            if self.outcomes.get(q.name) != "successful":
                continue
            if operation not in qubit_profiles[q.name]["operations"]:
                self.log(
                    f"Profile update skipped: operation {operation!r} "
                    "does not have a dedicated profile pulse."
                )
                continue
            amplitude = float(self.results["fit_results"][q.name]["opt_amp"])
            pulse_name = qubit_profiles[q.name]["operations"][operation]
            updates[f"pulses.json.pulses.{q.name}.{pulse_name}.amplitude"] = amplitude
        return updates

    @staticmethod
    def _save_streams(i, state_st, I_st, Q_st, amps, n_pi_vec) -> None:
        if state_st is not None:
            state_st[i].buffer(len(amps)).buffer(len(n_pi_vec)).average().save(
                f"state{i + 1}"
            )
        else:
            I_st[i].buffer(len(amps)).buffer(len(n_pi_vec)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(amps)).buffer(len(n_pi_vec)).average().save(f"Q{i + 1}")


class PowerRabiAnalysis(BaseAnalysis):
    """Shared analysis adapter for Power Rabi calibration data."""

    def process(self, ds):
        validate_readout_dataset(
            ds,
            self.node.parameters.use_state_discrimination,
        )
        return process_raw_dataset(ds, self.node)

    def fit(self, ds):
        return fit_raw_data(ds, self.node)

    def log(self, result):
        log_fitted_results(result.fit_results, log_callable=self.node.log)


if __name__ == "__main__":

    parameters = Parameters()
    parameters.reset_type = "active"
    parameters.use_state_discrimination = True
    parameters.num_shots = 1000
    parameters.transition = "ge"
    parameters.pi_repetitions = 4
    parameters.operation = "x180"

    options = CalibrationOptions()

    power_rabi = PowerRabi(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
        auto_connect=True,
    )
    power_rabi.run()
