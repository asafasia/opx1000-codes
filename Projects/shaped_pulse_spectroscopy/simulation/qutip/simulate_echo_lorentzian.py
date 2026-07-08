"""Run the echo-Lorentzian sweep locally with QuTiP.

This is a PC-only simulator. It does not connect to QOP hardware and does not
execute QUA. It mirrors the experiment sweep axes and pulse parameters, then
solves a driven two-level qubit Hamiltonian for each detuning/amplitude point.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    import qutip
except ImportError:  # pragma: no cover - exercised only when running the CLI.
    qutip = None

from parameters import SimulationParameters


def lorentzian_envelope(
    length_ns: int,
    tau_ns: float,
    peak_amplitude: float,
) -> np.ndarray:
    if length_ns < 4:
        raise ValueError("lorentzian_length_in_ns must be at least 4 ns.")
    if tau_ns <= 0:
        raise ValueError("lorentzian_tau_in_ns must be positive.")

    times = np.arange(length_ns, dtype=float) - (length_ns - 1) / 2
    return peak_amplitude / (1 + (times / tau_ns) ** 2)


def root_lorentzian_envelope(
    length_ns: int,
    cutoff: float,
    peak_amplitude: float,
) -> np.ndarray:
    if length_ns < 4:
        raise ValueError("root_lorentzian_length_in_ns must be at least 4 ns.")
    if not 0 < cutoff <= 1:
        raise ValueError("cutoff must satisfy 0 < cutoff <= 1.")
    if cutoff == 1:
        return np.full(length_ns, peak_amplitude, dtype=float)

    t_cut = length_ns / 2
    tau_ns = t_cut / np.sqrt(1 / cutoff**2 - 1)
    times = np.linspace(-t_cut, t_cut, length_ns)
    return peak_amplitude / np.sqrt(1 + (times / tau_ns) ** 2)


def gaussian_envelope(
    length_ns: int,
    cutoff: float,
    peak_amplitude: float,
) -> np.ndarray:
    if length_ns < 4:
        raise ValueError("gaussian_length_in_ns must be at least 4 ns.")
    if not 0 < cutoff <= 1:
        raise ValueError("cutoff must satisfy 0 < cutoff <= 1.")
    if cutoff == 1:
        return np.full(length_ns, peak_amplitude, dtype=float)

    t_cut = length_ns / 2
    sigma_ns = t_cut / np.sqrt(2 * np.log(1 / cutoff))
    times = np.linspace(-t_cut, t_cut, length_ns)
    return peak_amplitude * np.exp(-0.5 * (times / sigma_ns) ** 2)


def waveform_template_length(parameters: SimulationParameters) -> int:
    if parameters.waveform_template_length_in_ns is None:
        return int(parameters.lorentzian_length_in_ns)

    template_length = int(parameters.waveform_template_length_in_ns)
    pulse_length = int(parameters.lorentzian_length_in_ns)
    if template_length < 4:
        raise ValueError("waveform_template_length_in_ns must be at least 4 ns.")
    if template_length > pulse_length:
        raise ValueError(
            "waveform_template_length_in_ns cannot be longer than "
            "lorentzian_length_in_ns."
        )
    return template_length


def build_waveform(parameters: SimulationParameters) -> np.ndarray:
    waveform_length = waveform_template_length(parameters)
    if parameters.pulse_shape == "lorentzian":
        waveform = lorentzian_envelope(
            waveform_length,
            parameters.lorentzian_tau_in_ns,
            parameters.lorentzian_peak_amplitude,
        )
    elif parameters.pulse_shape == "root_lorentzian":
        waveform = root_lorentzian_envelope(
            waveform_length,
            parameters.cutoff,
            parameters.lorentzian_peak_amplitude,
        )
    elif parameters.pulse_shape == "gaussian":
        waveform = gaussian_envelope(
            waveform_length,
            parameters.cutoff,
            parameters.lorentzian_peak_amplitude,
        )
    else:
        raise ValueError(
            "pulse_shape must be 'lorentzian', 'root_lorentzian', or 'gaussian'."
        )

    if parameters.echo:
        signs = np.ones_like(waveform)
        signs[len(signs) // 2 :] = -1
        waveform = waveform * signs
    return waveform


def stretched_waveform(parameters: SimulationParameters) -> np.ndarray:
    waveform = build_waveform(parameters)
    pulse_length = int(parameters.lorentzian_length_in_ns)
    if len(waveform) == pulse_length:
        return waveform

    source_times = np.linspace(0, pulse_length - 1, len(waveform))
    target_times = np.arange(pulse_length, dtype=float)
    return np.interp(target_times, source_times, waveform)


def amplitude_to_rabi_frequency_hz(
    general_amp: Any,
    pi_amp: float,
    pi_length_ns: float,
) -> Any:
    if pi_amp == 0:
        raise ValueError("x180_amplitude must be non-zero.")
    if pi_length_ns <= 0:
        raise ValueError("x180_length_in_ns must be positive.")
    pi_amp_hz = 1 / (2 * pi_length_ns * 1e-9)
    return np.asarray(general_amp) / pi_amp * pi_amp_hz


def sweep_axes(parameters: SimulationParameters) -> tuple[np.ndarray, np.ndarray]:
    amps = np.arange(
        parameters.min_amp_factor,
        parameters.max_amp_factor,
        parameters.amp_factor_step,
    )
    if amps.size == 0:
        raise ValueError("Amplitude sweep is empty.")

    span_hz = round(parameters.frequency_span_in_mhz * 1e6)
    step_hz = round(parameters.frequency_step_in_mhz * 1e6)
    if step_hz <= 0:
        raise ValueError("frequency_step_in_mhz must be positive.")
    detunings = np.arange(-span_hz // 2, span_hz // 2 + step_hz, step_hz, dtype=float)
    return detunings, amps


def collapse_operators(parameters: SimulationParameters) -> list[Any]:
    if qutip is None:
        raise_qutip_missing()

    c_ops = []
    if parameters.t1_in_us is not None and parameters.t1_in_us > 0:
        t1_s = parameters.t1_in_us * 1e-6
        c_ops.append(np.sqrt(1 / t1_s) * qutip.sigmam())

    if parameters.t2_in_us is not None and parameters.t2_in_us > 0:
        t2_s = parameters.t2_in_us * 1e-6
        gamma_1 = 0 if parameters.t1_in_us is None else 1 / (parameters.t1_in_us * 1e-6)
        gamma_phi = max(1 / t2_s - gamma_1 / 2, 0)
        if gamma_phi > 0:
            c_ops.append(np.sqrt(gamma_phi / 2) * qutip.sigmaz())
    return c_ops


def simulate_point(
    parameters: SimulationParameters,
    waveform_v: np.ndarray,
    detuning_hz: float,
    amp_prefactor: float,
) -> float:
    if qutip is None:
        raise_qutip_missing()

    tlist_s = np.arange(len(waveform_v), dtype=float) * 1e-9
    rabi_hz = amplitude_to_rabi_frequency_hz(
        waveform_v * amp_prefactor,
        parameters.x180_amplitude,
        parameters.x180_length_in_ns,
    )
    omega_rad_s = 2 * np.pi * rabi_hz

    h0 = np.pi * detuning_hz * qutip.sigmaz()
    h_drive = 0.5 * qutip.sigmax()
    hamiltonian = [h0, [h_drive, omega_rad_s]]
    psi0 = qutip.basis(2, 0)
    result = qutip.mesolve(
        hamiltonian,
        psi0,
        tlist_s,
        c_ops=collapse_operators(parameters),
        e_ops=[qutip.basis(2, 1) * qutip.basis(2, 1).dag()],
    )
    return float(np.real(result.expect[0][-1]))


def simulate_grid(parameters: SimulationParameters) -> xr.Dataset:
    waveform = stretched_waveform(parameters)
    detunings, amps = sweep_axes(parameters)

    state = np.empty((1, len(detunings), len(amps)), dtype=float)
    for detuning_index, detuning_hz in enumerate(detunings):
        for amp_index, amp_prefactor in enumerate(amps):
            state[0, detuning_index, amp_index] = simulate_point(
                parameters,
                waveform,
                detuning_hz,
                amp_prefactor,
            )

    full_freq = parameters.rf_frequency_hz + detunings[np.newaxis, :]
    full_amp = parameters.lorentzian_peak_amplitude * amps[np.newaxis, :]
    rabi_frequency_hz = amplitude_to_rabi_frequency_hz(
        full_amp,
        parameters.x180_amplitude,
        parameters.x180_length_in_ns,
    )
    ds = xr.Dataset(
        {
            "state": (("qubit", "detuning", "amp_prefactor"), state),
            "waveform": (("time_ns",), waveform),
        },
        coords={
            "qubit": [parameters.qubit_name],
            "detuning": detunings,
            "amp_prefactor": amps,
            "time_ns": np.arange(len(waveform), dtype=float),
            "full_freq": (("qubit", "detuning"), full_freq),
            "full_amp": (("qubit", "amp_prefactor"), full_amp),
            "rabi_frequency_hz": (("qubit", "amp_prefactor"), rabi_frequency_hz),
        },
        attrs=serializable_parameters(parameters),
    )
    ds.detuning.attrs = {"long_name": "qubit detuning", "units": "Hz"}
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    ds.full_amp.attrs = {"long_name": "Lorentzian peak amplitude", "units": "V"}
    ds.rabi_frequency_hz.attrs = {"long_name": "Rabi frequency", "units": "Hz"}
    ds.state.attrs = {"long_name": "simulated excited-state population"}
    return ds


def plot_dataset(ds: xr.Dataset, output_dir: Path) -> Path:
    selected = ds.isel(qubit=0).assign_coords(
        detuning_MHz=ds.detuning / 1e6,
        rabi_frequency_MHz=ds.rabi_frequency_hz.isel(qubit=0) / 1e6,
    )
    figure, ax = plt.subplots(figsize=(9, 6))
    plotted = selected.state.transpose("amp_prefactor", "detuning").plot(
        ax=ax,
        x="detuning_MHz",
        y="rabi_frequency_MHz",
        add_colorbar=True,
        cbar_kwargs={"label": "Excited-state population"},
    )
    plotted.colorbar.set_label("Excited-state population")
    ax.set_title("QuTiP echo-Lorentzian simulation")
    ax.set_xlabel("Detuning [MHz]")
    ax.set_ylabel("Rabi frequency [MHz]")
    ax.grid(alpha=0.25)
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "echo_lorentzian_qutip.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_dataset(ds: xr.Dataset, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "echo_lorentzian_qutip.nc"
    ds.to_netcdf(path)
    return path


def save_parameters(parameters: SimulationParameters, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "parameters.json"
    path.write_text(json.dumps(serializable_parameters(parameters), indent=2))
    return path


def serializable_parameters(parameters: SimulationParameters) -> dict[str, Any]:
    values = asdict(parameters)
    values["output_dir"] = str(values["output_dir"])
    return values


def raise_qutip_missing() -> None:
    raise RuntimeError(
        "QuTiP is required for this PC-only simulation. Install it with "
        "`python -m pip install qutip` or add `qutip` to your conda environment."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulse-shape", choices=["lorentzian", "root_lorentzian", "gaussian"])
    parser.add_argument("--lorentzian-length-in-ns", type=int)
    parser.add_argument("--waveform-template-length-in-ns", type=int)
    parser.add_argument("--lorentzian-tau-in-ns", type=float)
    parser.add_argument("--lorentzian-peak-amplitude", type=float)
    parser.add_argument("--cutoff", type=float)
    parser.add_argument("--echo", action="store_true")
    parser.add_argument("--min-amp-factor", type=float)
    parser.add_argument("--max-amp-factor", type=float)
    parser.add_argument("--amp-factor-step", type=float)
    parser.add_argument("--frequency-span-in-mhz", type=float)
    parser.add_argument("--frequency-step-in-mhz", type=float)
    parser.add_argument("--qubit-name")
    parser.add_argument("--rf-frequency-hz", type=float)
    parser.add_argument("--x180-amplitude", type=float)
    parser.add_argument("--x180-length-in-ns", type=float)
    parser.add_argument("--t1-in-us", type=float)
    parser.add_argument("--t2-in-us", type=float)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def parameters_from_args(args: argparse.Namespace) -> SimulationParameters:
    parameters = SimulationParameters()
    for key, value in vars(args).items():
        if value is not None:
            setattr(parameters, key, value)
    return parameters


def main() -> None:
    parameters = parameters_from_args(parse_args())
    if qutip is None:
        raise_qutip_missing()

    ds = simulate_grid(parameters)
    parameter_path = save_parameters(parameters, parameters.output_dir)
    dataset_path = save_dataset(ds, parameters.output_dir)
    figure_path = plot_dataset(ds, parameters.output_dir)
    print(f"Saved parameters: {parameter_path}")
    print(f"Saved dataset: {dataset_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
