"""Spectroscopy versus external DC bias using one paused OPX job.

The source stays biased during thermal reset, drive, and readout. Voltages
are absolute source settings, centered on the selected profile's dc_bias_v.
No OPX Z line is required or driven. See calibrations/README.md for usage.
"""

from contextlib import contextmanager
from dataclasses import asdict
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
from qualang_tools.multi_user import qm_session

from calibration_utils.qubit_spectroscopy_vs_flux import (
    fit_raw_data,
    log_fitted_results,
    process_raw_dataset,
)
from calibration_utils.qubit_spectroscopy_vs_flux.external_parameters import Parameters
from calibrations.core import BaseCalibration, CalibrationOptions
from calibrations.output_safety import assert_outputs_allowed
from quam_config import create_machine


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
        if abs(pulse.amplitude * scale) > 0.7:
            raise ValueError("Scaled pulse amplitude must not exceed 0.7.")
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
                        q.resonator.measure("readout", qua_vars=(I, Q))
                        save(I, I_st)
                        save(Q, Q_st)
                        align()
            # Final handshake has no measurement: exactly one row per voltage.
            if not self.simulate_requested:
                pause()
            with stream_processing():
                for stream, name in ((I_st, "I"), (Q_st, "Q")):
                    stream.buffer(len(dfs)).buffer(p.num_shots).map(
                        FUNCTIONS.average(0)
                    ).save_all(name)
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
            self.log(job.execution_report())

    def _dataset(self, rows, count):
        axes = self.namespace["sweep_axes"]
        return xr.Dataset(
            {
                name: (
                    ("qubit", "detuning", "flux_bias"),
                    np.asarray(values).T[None, ...],
                )
                for name, values in rows.items()
            },
            coords={
                "qubit": axes["qubit"].values,
                "detuning": axes["detuning"],
                "flux_bias": axes["flux_bias"].values[:count],
                "drive_frequency_hz": ("qubit", [self.namespace["drive_frequency_hz"]]),
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

    def analyse_data(self):
        ds = process_raw_dataset(self.results["ds_raw"], self)
        # Preserve the acquisition frequency when analysing after profile changes.
        ds = ds.assign_coords(full_freq=ds.drive_frequency_hz + ds.detuning)
        ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
        self.results["ds_raw"] = ds
        try:
            if ds.sizes["flux_bias"] < 5:
                raise ValueError(
                    "At least five bias points are needed for the flux fit."
                )
            # Give the FFT-initialized periodic fitter a zero-based axis, then
            # choose the equivalent sweet spot nearest the scanned bias center.
            center = float((ds.flux_bias.min() + ds.flux_bias.max()) / 2)
            origin = float(ds.flux_bias.min())
            fits, results = fit_raw_data(
                ds.assign_coords(flux_bias=ds.flux_bias - origin), self
            )
            fits = fits.assign_coords(flux_bias=fits.flux_bias + origin)
            for coordinate in ("idle_offset", "flux_min"):
                if coordinate in fits.coords:
                    fits = fits.assign_coords({coordinate: fits[coordinate] + origin})
            self.results["ds_fit"] = fits
            self.results["fit_results"] = {
                name: asdict(result) for name, result in results.items()
            }
            for name, result in self.results["fit_results"].items():
                result["idle_offset"] += origin
                period = float(result["dv_phi0"])
                if not np.isfinite(period) or period <= 0:
                    raise ValueError(
                        "Flux fit did not produce a finite positive period."
                    )
                result["idle_offset"] += (
                    round((center - result["idle_offset"]) / period) * period
                )
                result["frequency_shift"] = float(
                    fits.peak_freq.sel(
                        qubit=name,
                        flux_bias=result["idle_offset"],
                        method="nearest",
                    )
                )
                result["qubit_frequency"] = (
                    float(ds.drive_frequency_hz.sel(qubit=name))
                    + result["frequency_shift"]
                )
                # The inherited periodic fit may extrapolate a sweet spot outside the scan.
                result["success"] = bool(
                    result["success"]
                    and float(ds.flux_bias.min())
                    <= result["idle_offset"]
                    <= float(ds.flux_bias.max())
                    and np.isfinite(result["qubit_frequency"])
                )
            fits = fits.assign_coords(
                idle_offset=(
                    "qubit",
                    [
                        self.results["fit_results"][str(name)]["idle_offset"]
                        for name in fits.qubit.values
                    ],
                ),
                sweet_spot_frequency=(
                    "qubit",
                    [
                        self.results["fit_results"][str(name)]["qubit_frequency"]
                        for name in fits.qubit.values
                    ],
                ),
                success=(
                    "qubit",
                    [
                        self.results["fit_results"][str(name)]["success"]
                        for name in fits.qubit.values
                    ],
                ),
            )
            self.results["ds_fit"] = fits
            log_fitted_results(self.results["fit_results"], log_callable=self.log)
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
        self.get_qubits()

    def plot_data(self):
        ds = self.results["ds_raw"]
        name = str(ds.qubit.values[0])
        fig, ax = plt.subplots(figsize=(9, 6))
        ds.sel(qubit=name).assign_coords(
            freq_GHz=ds.full_freq.sel(qubit=name) / 1e9
        ).IQ_abs.plot(
            ax=ax,
            x="flux_bias",
            y="freq_GHz",
            robust=True,
        )
        fits = self.results.get("ds_fit")
        if fits is not None:
            ax.plot(
                fits.flux_bias,
                (fits.peak_freq.sel(qubit=name) + float(ds.drive_frequency_hz[0]))
                / 1e9,
                ".",
                color="white",
                label="Spectroscopy peaks",
            )
        result = self.results.get("fit_results", {}).get(name, {})
        if result.get("success"):
            ax.axvline(
                result["idle_offset"],
                color="red",
                linestyle="--",
                label="Fitted sweet spot",
            )
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.set(
            xlabel="External source voltage (V)",
            ylabel="Qubit frequency (GHz)",
            title=f"{name}: spectroscopy versus external bias",
        )
        fig.tight_layout()
        self.results["figures"] = {"spectroscopy_vs_external_flux": fig}
        plt.show()

    def profile_updates(self):
        updates = {}
        for name, result in self.results.get("fit_results", {}).items():
            if result["success"]:
                updates[f"qubits.json.qubits.{name}.dc_bias_v"] = float(
                    result["idle_offset"]
                )
                updates[f"qubits.json.qubits.{name}.frequencies_hz.qubit_f01"] = float(
                    result["qubit_frequency"]
                )
        return updates


if __name__ == "__main__":
    parameters = Parameters()

    parameters.reset_type = "thermal"
    parameters.use_state_discrimination = False
    parameters.num_shots = 100
    parameters.flux_bias_center_in_v = 0  # Use the profile's dc_bias_v.
    parameters.flux_offset_span_in_v = 0.5
    parameters.num_flux_points = 11
    parameters.bias_settle_time_s = 0.1

    options = CalibrationOptions(apply_profile_update=False)

    calibration = QubitSpectroscopyVsExternalFlux(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q6"),
    )
    calibration.run()
