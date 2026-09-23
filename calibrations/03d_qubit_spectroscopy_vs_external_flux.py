"""Spectroscopy versus external DC bias using one paused OPX job.

The source stays biased during thermal reset, drive, and readout. Voltages
are absolute source settings, centered on the selected profile's dc_bias_v.
No OPX Z line is required or driven. See calibrations/README.md for usage.
"""

from contextlib import contextmanager
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import (
    FUNCTIONS,
    align,
    declare,
    declare_stream,
    fixed,
    for_,
    pause,
    program,
    save,
    stream_processing,
)
from qualang_tools.loops import from_array
from utils.qm_session import qm_session

from calibration_utils.qubit_spectroscopy_vs_flux import (
    process_raw_dataset,
)
from calibration_utils.qubit_spectroscopy_vs_flux.external_parameters import Parameters
from calibration_utils.qubit_spectroscopy_vs_flux.external_analysis import (
    fit_external_flux,
)
from calibrations.core import BaseCalibration, CalibrationOptions
from calibrations.output_safety import assert_outputs_allowed
from quam_config import create_machine
from quam_config.instrument_limits import instrument_limits


class QubitSpectroscopyVsExternalFlux(BaseCalibration):
    def __init__(self, parameters: Parameters, machine=None, **kwargs):
        kwargs.setdefault(
            "options",
            CalibrationOptions(
                update_state=False,
                apply_profile_update=False,
            ),
        )
        super().__init__(
            name="03d_qubit_spectroscopy_vs_external_flux",
            description=__doc__,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    @contextmanager
    def automatic_dc_bias(self):
        # This calibration owns the entire voltage sweep and its cleanup.
        yield

    def create_qua_program(self):
        p = Parameters.model_validate(self.parameters.model_dump())
        self.parameters = p
        qubits = self.get_qubits()
        if len(qubits) != 1:
            raise ValueError("External DC bias requires exactly one selected qubit.")
        q = next(iter(qubits))
        bias = getattr(self.machine, "dc_bias", None)
        if bias is None:
            raise ValueError(
                "The selected profile must configure connectivity.dc_bias."
            )
        self.namespace["configured_bias_v"] = float(bias.voltage_for_qubit(q.name))
        center = p.flux_bias_center_in_v
        if center is None:
            center = bias.voltage_for_qubit(q.name)
        voltages = center + np.linspace(
            -p.flux_offset_span_in_v / 2,
            p.flux_offset_span_in_v / 2,
            p.num_flux_points,
        )
        limit = bias.max_abs_voltage_v
        if not np.all(np.isfinite(voltages)) or np.any(np.abs(voltages) > limit):
            raise ValueError(f"Every swept voltage must lie within +/-{limit:g} V.")
        dfs = np.rint(
            np.arange(
                -p.frequency_span_in_mhz * 1e6 / 2,
                p.frequency_span_in_mhz * 1e6 / 2,
                p.frequency_step_in_mhz * 1e6,
            )
        ).astype(np.int64)
        if len(dfs) < 2 or np.any(np.diff(dfs) <= 0):
            raise ValueError(
                "Frequency sweep needs at least two distinct integer-Hz points."
            )
        pulse = q.xy.operations[p.operation]
        duration = (
            p.operation_len_in_ns if p.operation_len_in_ns is not None else pulse.length
        )
        if duration < 16 or duration % 4:
            raise ValueError(
                "Operation duration must be at least 16 ns and divisible by 4."
            )
        scale = p.operation_amplitude_factor
        if not np.isfinite(scale) or not -2 <= scale < 2:
            raise ValueError(
                "Operation amplitude factor must be finite and in [-2, 2)."
            )
        amplitude_limit = instrument_limits(q.xy).max_wf_amplitude
        if abs(pulse.amplitude * scale) > amplitude_limit:
            raise ValueError(
                f"Scaled pulse amplitude must not exceed {amplitude_limit:g}."
            )
        self.namespace["sweep_axes"] = {
            "qubit": xr.DataArray([q.name], dims="qubit"),
            "detuning": xr.DataArray(
                dfs,
                dims="detuning",
                attrs={"long_name": "Qubit detuning", "units": "Hz"},
            ),
            "flux_bias": xr.DataArray(
                voltages,
                dims="flux_bias",
                attrs={"long_name": "External source voltage", "units": "V"},
            ),
        }
        self.namespace["drive_frequency_hz"] = float(q.xy.RF_frequency)
        self.namespace["bias_center_v"] = float(center)
        self.configure_acquisition(loop_order=("flux_bias", "shot", "detuning"))
        with program() as qua_program:
            i = declare(int)
            n = declare(int)
            df = declare(int)
            I, Q = declare(fixed), declare(fixed)
            I_st, Q_st = declare_stream(), declare_stream()
            with for_(i, 0, i < len(voltages), i + 1):
                if not self.simulate_requested:
                    pause()
                with for_(n, 0, n < p.num_shots, n + 1):
                    with for_(*from_array(df, dfs)):
                        q.reset_qubit_thermal()
                        q.xy.update_frequency(df + q.xy.intermediate_frequency)
                        q.xy.play(
                            p.operation, amplitude_scale=scale, duration=duration // 4
                        )
                        align()
                        self.measure_readout(q, qua_vars=(I, Q))
                        save(I, I_st)
                        save(Q, Q_st)
                        align()
            # Final handshake has no measurement: exactly one row per voltage.
            if not self.simulate_requested:
                pause()
            with stream_processing():
                for stream, name in ((I_st, "I"), (Q_st, "Q")):
                    self.save_acquisition_stream(stream, name, split_outer_axis="flux_bias")
        return qua_program

    def _wait_for(self, predicate, description):
        deadline = time.monotonic() + self.parameters.pause_timeout_s
        while True:
            assert_outputs_allowed()
            if predicate():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {description}.")
            time.sleep(0.05)

    def _settle(self):
        deadline = time.monotonic() + self.parameters.bias_settle_time_s
        while time.monotonic() < deadline:
            assert_outputs_allowed()
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    @staticmethod
    def _fetch_row(handle, index, size, timeout):
        result = handle.fetch(index, timeout=timeout)
        if isinstance(result, dict) or getattr(
            getattr(result, "dtype", None), "names", None
        ):
            result = result["value"]
        row = np.asarray(result, dtype=float).reshape(-1)
        if row.size != size:
            raise ValueError(f"Expected {size} frequency points, received {row.size}.")
        return row

    def execute_qua_program(self):
        assert_outputs_allowed()
        bias = self.machine.dc_bias
        axes = self.namespace["sweep_axes"]
        voltages = axes["flux_bias"].values
        size = len(axes["detuning"])
        if self.single_shot_acquisition:
            size *= self.parameters.num_shots
        rows = {"I": [], "Q": []}
        qmm = self.machine.connect()
        config = self.machine.generate_config()
        with qm_session(qmm, config, timeout=self.timeout) as qm:
            assert_outputs_allowed()
            job = qm.execute(self.namespace["qua_program"])
            self.namespace["job"] = job
            try:
                self._wait_for(job.is_paused, "initial OPX pause")
                # Open the source only after the OPX has stopped at its first pause.
                with bias.applied(bias.output_channel, float(voltages[0])):
                    try:
                        handles = {name: job.result_handles.get(name) for name in rows}
                        for index, voltage in enumerate(voltages):
                            assert_outputs_allowed()
                            if index:
                                bias.set_voltage(bias.output_channel, float(voltage))
                            self._settle()
                            assert_outputs_allowed()
                            job.resume()
                            # Indexed save_all results prevent stale I/Q from a previous bias.
                            self._wait_for(
                                lambda: all(
                                    h.count_so_far() >= index + 1
                                    for h in handles.values()
                                ),
                                f"I/Q results for voltage point {index + 1}",
                            )
                            self._wait_for(
                                job.is_paused, "OPX pause after spectroscopy sweep"
                            )
                            for name, handle in handles.items():
                                rows[name].append(
                                    self._fetch_row(
                                        handle,
                                        index,
                                        size,
                                        self.parameters.pause_timeout_s,
                                    )
                                )
                            self.results["ds_raw"] = self._dataset(rows, index + 1)
                            self.log(
                                f"Bias {index + 1}/{len(voltages)}: {voltage:g} V complete"
                            )
                    finally:
                        # Stop RF execution before returning the source to zero.
                        job.halt()
            except BaseException:
                # Also covers failure entering the source context or the initial pause.
                job.halt()
                raise

    def _dataset(self, rows, count):
        axes = self.namespace["sweep_axes"]
        if self.single_shot_acquisition:
            variables = {
                name: (("qubit", "shot", "detuning", "flux_bias"),
                       np.asarray(values).reshape(count, self.parameters.num_shots, -1)
                       .transpose(1, 2, 0)[None, ...])
                for name, values in rows.items()
            }
        else:
            variables = {
                name: (("qubit", "detuning", "flux_bias"), np.asarray(values).T[None, ...])
                for name, values in rows.items()
            }
        dataset = xr.Dataset(
            variables,
            coords={
                "qubit": axes["qubit"].values,
                "detuning": axes["detuning"],
                "flux_bias": axes["flux_bias"].values[:count],
                "drive_frequency_hz": ("qubit", [self.namespace["drive_frequency_hz"]]),
                "configured_bias_v": ("qubit", [self.namespace["configured_bias_v"]]),
            },
            attrs={
                "bias_source": "external_dc",
                "bias_channel": self.machine.dc_bias.output_channel,
                "bias_center_v": self.namespace["bias_center_v"],
            },
        ).assign_coords(
            flux_bias=xr.DataArray(
                axes["flux_bias"].values[:count],
                dims="flux_bias",
                attrs=axes["flux_bias"].attrs,
            )
        )

        if self.single_shot_acquisition:
            dataset = dataset.assign_coords(shot=axes["shot"])
        return self.annotate_readout_dataset(dataset)

    def analyse_data(self):
        self.prepare_acquisition_results()
        ds = process_raw_dataset(self.results["ds_raw"], self)
        # Preserve the acquisition frequency when analysing after profile changes.
        ds = ds.assign_coords(full_freq=ds.drive_frequency_hz + ds.detuning)
        ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
        self.results["ds_raw"] = ds
        self.results.pop("ds_fit", None)
        try:
            if ds.sizes["flux_bias"] < 5:
                raise ValueError(
                    "At least five bias points are needed for the flux fit."
                )
            fits, results = fit_external_flux(ds)
            self.results["ds_fit"] = fits
            self.results["fit_results"] = results
            for name, result in results.items():
                if result["success"]:
                    self.log(
                        f"{name}: selected {result['selected_quadrature']} fit "
                        f"(inlier R^2={result['r_squared']:.4f}); parabolic {result['extremum_type']} at "
                        f"{result['idle_offset']:.6g} V "
                        f"(fit uncertainty {result['extremum_voltage_std_v']:.2g} V), "
                        f"{result['qubit_frequency'] / 1e9:.6g} GHz; "
                        f"{result['num_outliers']} outlier(s) excluded."
                    )
                else:
                    self.log(f"{name}: parabola fit unavailable: {result['reason']}")
        except (
            ValueError,
            RuntimeError,
            IndexError,
            KeyError,
            ZeroDivisionError,
        ) as exc:
            self.log(f"Flux fit unavailable; retaining spectroscopy map: {exc}")
            self.results["fit_results"] = {
                str(name): {"success": False, "reason": str(exc)}
                for name in ds.qubit.values
            }
        self.outcomes = {
            name: "successful" if result["success"] else "failed"
            for name, result in self.results["fit_results"].items()
        }
        self.results["analysis"] = {
            "fit_results": self.results["fit_results"],
            "outcomes": self.outcomes,
        }

    def load_data(self, run_directory):
        # NPZ saves coordinate values without dimension names. Restore explicitly,
        # including the drive-frequency coordinate, even when axis lengths match.
        directory = Path(run_directory)
        metadata = directory / "metadata.json"
        if metadata.is_file():
            import json
            if json.loads(metadata.read_text(encoding="utf-8")).get("dataset_schema"):
                self.results["ds_raw"] = self.load_saved_run(directory)
                self.get_qubits()
                return
        with np.load(directory / "sweep.npz", allow_pickle=False) as axes, np.load(
            directory / "results.npz", allow_pickle=False
        ) as values:
            self.results["ds_raw"] = xr.Dataset(
                {
                    name: (("qubit", "detuning", "flux_bias"), values[name])
                    for name in ("I", "Q")
                },
                coords={
                    "qubit": axes["qubit"],
                    "detuning": axes["detuning"],
                    "flux_bias": axes["flux_bias"],
                    "drive_frequency_hz": ("qubit", axes["drive_frequency_hz"]),
                },
            )
            if "configured_bias_v" in axes.files:
                self.results["ds_raw"] = self.results["ds_raw"].assign_coords(
                    configured_bias_v=("qubit", axes["configured_bias_v"])
                )
        self.get_qubits()

    def plot_data(self):
        ds = self.results["ds_raw"]
        name = str(ds.qubit.values[0])
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)
        selected = ds.sel(qubit=name).assign_coords(
            freq_GHz=ds.full_freq.sel(qubit=name) / 1e9
        )
        fits = self.results.get("ds_fit")
        result = self.results.get("fit_results", {}).get(name, {})
        for ax, quadrature in zip(axes, ("I", "Q")):
            selected[quadrature].plot(
                ax=ax,
                x="flux_bias",
                y="freq_GHz",
                robust=True,
                cbar_kwargs={"label": f"{quadrature} (V)"},
            )
            panel_result = result.get("quadrature_fit_results", {}).get(quadrature, {})
            if fits is not None:
                voltage = fits.flux_bias.values
                peak = fits.quadrature_peak_freq.sel(
                    qubit=name, quadrature=quadrature
                ).values
                inlier = fits.quadrature_fit_inlier.sel(
                    qubit=name, quadrature=quadrature
                ).values
                drive = float(ds.drive_frequency_hz.sel(qubit=name))
                finite = np.isfinite(peak)
                ax.plot(
                    voltage[inlier],
                    (peak[inlier] + drive) / 1e9,
                    ".",
                    color="white",
                    label="Fit inliers",
                )
                excluded = finite & ~inlier
                ax.plot(
                    voltage[excluded],
                    (peak[excluded] + drive) / 1e9,
                    "x",
                    color="orange",
                    label="Excluded peaks",
                )
                if "coefficients_hz" in panel_result:
                    dense = np.linspace(float(voltage.min()), float(voltage.max()), 300)
                    normalized = (dense - panel_result["fit_center_v"]) / panel_result[
                        "fit_scale_v"
                    ]
                    fitted = np.polyval(panel_result["coefficients_hz"], normalized)
                    ax.plot(
                        dense,
                        (fitted + drive) / 1e9,
                        color="cyan",
                        label="Robust parabola",
                    )
            if panel_result.get("success"):
                vertex = panel_result["idle_offset"]
                ax.axvline(
                    vertex,
                    color="red",
                    linestyle="--",
                    label=f"{panel_result['extremum_type'].capitalize()}: {vertex:.6g} V",
                )
                ax.plot(
                    vertex,
                    panel_result["qubit_frequency"] / 1e9,
                    "*",
                    color="red",
                    markersize=12,
                )
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
            ax.set(xlabel="", ylabel="Qubit frequency (GHz)", title="")
            score = panel_result.get("r_squared")
            label = (
                f"{quadrature} | inlier R^2={score:.4f}"
                if score is not None
                else f"{quadrature} | fit unavailable"
            )
            if result.get("selected_quadrature") == quadrature:
                label += " | selected"
            elif not panel_result.get("success") and score is not None:
                label += " | invalid extremum"
            ax.text(
                0.015,
                0.97,
                label,
                transform=ax.transAxes,
                va="top",
                fontweight="bold",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
        axes[-1].set_xlabel("Absolute external source voltage (V)")
        fig.suptitle(f"{name}: spectroscopy versus external bias")
        if "configured_bias_v" in ds.coords:
            reference = float(ds.configured_bias_v.sel(qubit=name))
            reference_label = "configured bias"
        else:
            # Legacy runs did not store this reference; label the fallback.
            reference = float(self.machine.dc_bias.voltage_for_qubit(name))
            reference_label = "current profile bias"
        upper_axis = axes[0].secondary_xaxis(
            "top",
            functions=(
                lambda voltage: voltage - reference,
                lambda offset: offset + reference,
            ),
        )
        upper_axis.set_xlabel(f"Offset from {reference_label} ({reference:g} V) [V]")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self.results["figures"] = {"spectroscopy_vs_external_flux": fig}
        plt.show()

    def profile_updates(self):
        return self.qubit_profile_updates({
            "dc_bias_v": "idle_offset",
            "frequencies_hz.qubit_f01": "qubit_frequency",
        })


if __name__ == "__main__":
    parameters = Parameters()

    parameters.reset_type = "thermal"
    parameters.use_state_discrimination = False
    parameters.num_shots = 100
    parameters.operation_amplitude_factor = (
        0.05  # Fraction of the profile pulse amplitude; lower for weaker drive.
    )
    # parameters.flux_bias_center_in_v = 0  # Use the profile's dc_bias_v.
    parameters.flux_offset_span_in_v = 1
    parameters.num_flux_points = 10
    parameters.bias_settle_time_s = 1
    parameters.frequency_span_in_mhz = 100
    parameters.frequency_step_in_mhz = 1

    options = CalibrationOptions(apply_profile_update=True, update_state=False)

    calibration = QubitSpectroscopyVsExternalFlux(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
