"""Class-based calibration for 04d_power_rabi_chevron."""

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
from qm.qua import *
from qualang_tools.loops import from_array
from utils.qm_session import qm_session
from qualang_tools.units import unit
from calibration_utils.power_rabi_chevron import (
    Parameters,
    plot_raw_data,
    plot_stark_shift,
    fit_stark_shift,
    process_raw_dataset,
)
from calibration_utils.analysis_base import AnalysisResult
from quam_config import Quam, create_machine
from calibration_io import CalibrationSaver, current_profile_name
from utils.plotting_settings import plot_per_qubit
from utils.simulation import simulate_and_plot

if __package__ in {None, ""}:
    from calibrations.core import BaseCalibration, CalibrationOptions
else:
    from .core import BaseCalibration, CalibrationOptions

description = """
        POWER RABI CHEVRON - FREQUENCY VS AMPLITUDE
This sequence plays a fixed-duration qubit operation while sweeping both its
amplitude and the qubit-drive frequency. It is the amplitude-sweep counterpart
of the duration-based Rabi chevron.

The analysis fits Gaussian spectral centers versus amplitude squared and tests a free
power-law exponent. For a plain square drive it compares the apparent shift
with f_Rabi^2 / |f12-f01|. Weak, ambiguous and edge peaks are excluded.
It does not automatically update pulse parameters.
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
            "Power Rabi chevron readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(variables)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


class PowerRabiChevron(BaseCalibration[Parameters, Quam]):
    """Class-based calibration for ``calibrations/04d_power_rabi_chevron.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="04d_power_rabi_chevron",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    def create_qua_program(self):
        node = self
        """Create the frequency-versus-amplitude Rabi-chevron QUA program."""
        u = unit(coerce_to_integer=True)
        node.namespace["qubits"] = qubits = self.get_qubits()
        num_qubits = len(qubits)
        operation = node.parameters.operation
        for qubit in qubits:
            if operation not in qubit.xy.operations:
                raise ValueError(
                    f"{qubit.name} does not define operation {operation!r}."
                )

        amps = np.arange(
            node.parameters.min_amp_factor,
            node.parameters.max_amp_factor,
            node.parameters.amp_factor_step,
        )
        if amps.size == 0:
            raise ValueError("Amplitude sweep is empty.")
        if np.any(np.abs(amps) >= 2):
            raise ValueError("QUA amplitude prefactors must stay within [-2, 2).")

        span = int(round(node.parameters.frequency_span_in_mhz * u.MHz))
        step = int(round(node.parameters.frequency_step_in_mhz * u.MHz))
        if step <= 0:
            raise ValueError("frequency_step_in_mhz must be positive.")
        dfs = np.arange(-span // 2, span // 2 + step, step, dtype=int)

        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "detuning": xr.DataArray(
                dfs, attrs={"long_name": "qubit detuning", "units": "Hz"}
            ),
            "amp_prefactor": xr.DataArray(
                amps,
                attrs={"long_name": "pulse amplitude prefactor"},
            ),
        }

        self.configure_acquisition()

        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            if node.parameters.use_state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [self.declare_state_stream() for _ in range(num_qubits)]
            a = declare(fixed)
            df = declare(int)

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, dfs)):
                        with for_(*from_array(a, amps)):
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency
                                )
                                self.reset_qubit(qubit)
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency + df
                                )
                            align()

                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play(operation, amplitude_scale=a)
                            align()

                            for i, qubit in multiplexed_qubits.items():
                                if node.parameters.use_state_discrimination:
                                    self.readout_state(qubit, state[i])
                                    self.save_readout_state(state[i], state_st[i])
                                else:
                                    self.measure_readout(qubit, qua_vars=(I[i], Q[i]))
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])

                            align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    self.process_readout_streams(
                        i,
                        state_st if self.parameters.use_state_discrimination else None,
                        I_st,
                        Q_st,
                    )

        return node.namespace.get("qua_program")

    def simulate_qua_program(self):
        node = self
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        samples, figure, waveform_report = simulate_and_plot(
            qmm,
            config,
            node.namespace["qua_program"],
            node.parameters,
        )
        node.results["simulation"] = {
            "figure": figure,
            "waveform_report": waveform_report,
            "samples": samples,
        }

    def execute_qua_program(self):
        node = self
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program"])
            dataset = self.fetch_result_dataset(job)
        validate_readout_dataset(dataset, node.parameters.use_state_discrimination)
        node.results["ds_raw"] = self.annotate_readout_dataset(dataset)

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
        node.load_from_id(load_data_id)
        node.parameters.load_data_id = load_data_id
        node.namespace["qubits"] = self.get_qubits()

    def analyse_data(self):
        self.prepare_acquisition_results()
        node = self
        validate_readout_dataset(
            node.results["ds_raw"], node.parameters.use_state_discrimination
        )
        processed = process_raw_dataset(node.results["ds_raw"], node)
        ds_fit, reports = (
            fit_stark_shift(processed, node)
            if node.parameters.fit_stark_shift
            else (None, {})
        )
        node.results.pop("ds_fit", None)
        result = AnalysisResult(
            ds_processed=processed,
            ds_fit=ds_fit,
            fit_results=reports,
            outcomes={
                q: "successful" if r["success"] else "failed"
                for q, r in reports.items()
            },
            summary={
                "observable": "Gaussian center of fixed-duration excitation spectrum",
                "model": "peak_detuning_hz = intercept_hz + coefficient * amplitude**2",
                "theory": "shift_hz = C * f_Rabi_hz**2 / abs(f12_hz-f01_hz); weak square-drive C=0.5",
                "caveat": "Quadratic scaling alone cannot establish inverse-anharmonicity scaling. Physical comparison assumes calibrated pi area and linear drive gain.",
            },
        )
        node.apply_analysis_result(result)
        for name, report in reports.items():
            node.log(
                f"{name}: {report['n_fit_points']} spectral peaks used; "
                f"quadratic {report['quadratic_consistency']}; "
                f"power exponent={report['exponent']}; C={report['theory_coefficient']}. "
                f"{report['reason']}"
            )

    def save_analysis_result(self):
        saved = super().save_analysis_result()
        ds_fit = self.results.get("ds_fit")
        if saved and ds_fit is not None:
            # Persist numeric peaks, uncertainties, masks and rejection reasons,
            # not just the compact dataset summary in analysis_result.json.
            columns = [
                "peak_detuning_hz",
                "peak_frequency_hz",
                "peak_frequency_std_hz",
                "peak_valid",
                "peak_fit_used",
                "peak_status",
                "quadratic_fit_detuning_hz",
                "theory_shift_hz",
            ]
            columns += [
                name for name in ds_fit.data_vars if name.startswith("gaussian_")
            ]
            table = (
                ds_fit[columns].drop_dims("detuning", errors="ignore").to_dataframe()
            )
            output = (
                Path(self.namespace["calibration_run_directory"]) / "stark_peaks.csv"
            )
            table.to_csv(output)
            self.log(f"Spectral peaks and fit selection saved to {output}")
        return saved

    def plot_data(self):
        node = self
        figures = plot_per_qubit(
            plot_raw_data,
            node.results["ds_raw"],
            node.namespace["qubits"],
            figure_name="power_rabi_chevron",
            use_state_discrimination=node.parameters.use_state_discrimination,
            ds_fit=node.results.get("ds_fit"),
        )
        if node.results.get("ds_fit") is not None:
            figures.update(
                plot_per_qubit(
                    plot_stark_shift,
                    node.results["ds_fit"],
                    node.namespace["qubits"],
                    figure_name="ac_stark_shift",
                    fit_results=node.results["fit_results"],
                )
            )
        if node.parameters.use_state_discrimination:
            for index, label in enumerate(("g", "e", "f")):
                if f"population_{label}" in node.results["ds_raw"]:
                    figures.update(
                        plot_per_qubit(
                            plot_raw_data,
                            node.results["ds_raw"],
                            node.namespace["qubits"],
                            figure_name=f"power_rabi_chevron_P{index}",
                            use_state_discrimination=True,
                            population=label,
                        )
                    )
        node.results["figures"] = figures
        if "calibration_run_directory" in node.namespace:
            figures_directory = CalibrationSaver().save_figures(
                node.namespace["calibration_run_directory"],
                node.results["figures"],
            )
            node.log(f"Calibration figures saved to {figures_directory}")
        plt.show()


if __name__ == "__main__":
    parameters = Parameters()
    parameters.acquisition = (
        "single_shot"  # or "single_shot" to retain every measurement
    )

    parameters.operation = "saturation"
    parameters.reset_type = "active"
    parameters.readout_states = ["g", "e", "f"]  # GE readout; add "f" for readout_GEF.

    parameters.frequency_span_in_mhz = 200
    parameters.frequency_step_in_mhz = 0.2
    parameters.min_amp_factor = 0
    parameters.amp_factor_step = 0.01
    parameters.max_amp_factor = 1
    parameters.num_shots = 100
    parameters.use_state_discrimination = True

    options = CalibrationOptions()
    # options.ai_review = True

    calibration = PowerRabiChevron(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q6"),
    )
    calibration.run()
