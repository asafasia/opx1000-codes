"""Scan resonator readout amplitude and qubit-drive frequency offset.

Runs 30 resonator spectroscopy experiments in one OPX job, while saving and
analyzing every parameter combination separately.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from qm.qua import align, declare, fixed, for_, program, save, stream_processing
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qualibration_libs.data import XarrayDataFetcher, convert_IQ_to_V

from quam_config import create_machine
from utils.plotting_settings import FIGURE_SIZE


QUBIT_NAME = "q9"
NUM_SHOTS = 150
QUBIT_DRIVE_AMPLITUDE = 0.5
READOUT_AMPLITUDE_FACTORS = np.array([0.5,1])
QUBIT_OFFSETS_HZ = np.array([-100, -60, -20, 20, 60, 100]) * 1_000_000
RESONATOR_DETUNINGS_HZ = np.arange(-20, 21, 1) * 1_000_000
SATURATION_LEAD_TIME_NS = 10_000


def build_program(machine, qubit):
    """Build one program containing the full 30-combination scan."""
    u = unit(coerce_to_integer=True)
    rr_points = len(RESONATOR_DETUNINGS_HZ)
    q_points = len(QUBIT_OFFSETS_HZ)
    amp_points = len(READOUT_AMPLITUDE_FACTORS)

    with program() as qua_program:
        Ig, Ig_st, Qg, Qg_st, n, n_st = machine.declare_qua_variables()
        Id, Id_st, Qd, Qd_st, _, _ = machine.declare_qua_variables()
        rr_df = declare(int)
        q_df = declare(int)
        readout_amp = declare(fixed)

        with for_(n, 0, n < NUM_SHOTS, n + 1):
            save(n, n_st)
            with for_(*from_array(readout_amp, READOUT_AMPLITUDE_FACTORS)):
                with for_(*from_array(q_df, QUBIT_OFFSETS_HZ)):
                    qubit.xy.update_frequency(q_df + qubit.xy.intermediate_frequency)
                    with for_(*from_array(rr_df, RESONATOR_DETUNINGS_HZ)):
                        qubit.resonator.update_frequency(
                            rr_df + qubit.resonator.intermediate_frequency
                        )

                        qubit.resonator.measure(
                            "readout",
                            amplitude_scale=readout_amp,
                            qua_vars=(Ig[0], Qg[0]),
                        )
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)

                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.xy.play(
                            "saturation",
                            amplitude_scale=QUBIT_DRIVE_AMPLITUDE,
                        )
                        qubit.resonator.wait(SATURATION_LEAD_TIME_NS * u.ns)
                        qubit.resonator.measure(
                            "readout",
                            amplitude_scale=readout_amp,
                            qua_vars=(Id[0], Qd[0]),
                        )
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        qubit.reset_qubit_thermal()
                        align(qubit.xy.name, qubit.resonator.name)

                        save(Ig[0], Ig_st[0])
                        save(Qg[0], Qg_st[0])
                        save(Id[0], Id_st[0])
                        save(Qd[0], Qd_st[0])

        with stream_processing():
            n_st.save("n")
            for stream, name in (
                (Ig_st[0], "Ig1"),
                (Qg_st[0], "Qg1"),
                (Id_st[0], "Id1"),
                (Qd_st[0], "Qd1"),
            ):
                (
                    stream.buffer(rr_points)
                    .buffer(q_points)
                    .buffer(amp_points)
                    .average()
                    .save(name)
                )

    return qua_program


def acquire(machine, qubit) -> xr.Dataset:
    """Execute the scan and return calibrated-voltage IQ data."""
    axes = {
        "qubit": xr.DataArray([QUBIT_NAME]),
        "readout_amplitude_factor": xr.DataArray(READOUT_AMPLITUDE_FACTORS),
        "qubit_offset_hz": xr.DataArray(QUBIT_OFFSETS_HZ),
        "resonator_detuning_hz": xr.DataArray(RESONATOR_DETUNINGS_HZ),
    }
    config = machine.generate_config()
    qua_program = build_program(machine, qubit)

    with qm_session(machine.connect(), config, timeout=300) as qm:
        job = qm.execute(qua_program)
        fetcher = XarrayDataFetcher(job, axes)
        for dataset in fetcher:
            progress = int(fetcher.get("n", 0)) + 1
            print(f"\rAverages: {min(progress, NUM_SHOTS)}/{NUM_SHOTS}", end="")
        print()
        print(job.execution_report())

    dataset = convert_IQ_to_V(
        dataset,
        [qubit],
        IQ_list=["Ig", "Qg", "Id", "Qd"],
    )
    dataset = dataset.assign(
        ground_abs=np.hypot(dataset.Ig, dataset.Qg),
        driven_abs=np.hypot(dataset.Id, dataset.Qd),
        complex_separation=np.hypot(dataset.Id - dataset.Ig, dataset.Qd - dataset.Qg),
    )
    dataset.attrs.update(
        qubit_name=QUBIT_NAME,
        qubit_base_frequency_hz=float(qubit.xy.RF_frequency),
        resonator_base_frequency_hz=float(qubit.resonator.RF_frequency),
        qubit_drive_amplitude_factor=QUBIT_DRIVE_AMPLITUDE,
        base_readout_pulse_amplitude=float(qubit.resonator.operations["readout"].amplitude),
        num_shots=NUM_SHOTS,
    )
    return dataset


def analyze_and_save(dataset: xr.Dataset, output_dir: Path) -> pd.DataFrame:
    """Save each experiment separately and create summary artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = output_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    base_rr = dataset.attrs["resonator_base_frequency_hz"]
    base_q = dataset.attrs["qubit_base_frequency_hz"]
    base_readout_amp = dataset.attrs["base_readout_pulse_amplitude"]
    rows = []

    for amp_factor in READOUT_AMPLITUDE_FACTORS:
        for q_offset in QUBIT_OFFSETS_HZ:
            selected = dataset.sel(
                qubit=QUBIT_NAME,
                readout_amplitude_factor=amp_factor,
                qubit_offset_hz=q_offset,
            )
            rr_freq = base_rr + selected.resonator_detuning_hz.values
            ground = selected.ground_abs.values
            driven = selected.driven_abs.values
            separation = selected.complex_separation.values
            ground_idx = int(np.argmin(ground))
            driven_idx = int(np.argmin(driven))
            max_sep_idx = int(np.argmax(separation))
            response = float(np.max(separation))

            row = {
                "resonator_pulse_amplitude": float(base_readout_amp * amp_factor),
                "resonator_amplitude_factor": float(amp_factor),
                "qubit_drive_amplitude": QUBIT_DRIVE_AMPLITUDE,
                "qubit_frequency_offset_hz": int(q_offset),
                "applied_qubit_frequency_hz": float(base_q + q_offset),
                "ground_resonator_frequency_hz": float(rr_freq[ground_idx]),
                "driven_resonator_frequency_hz": float(rr_freq[driven_idx]),
                "apparent_resonator_shift_hz": float(
                    rr_freq[driven_idx] - rr_freq[ground_idx]
                ),
                "max_complex_separation_v": response,
                "max_separation_resonator_frequency_hz": float(rr_freq[max_sep_idx]),
            }
            rows.append(row)

            table = pd.DataFrame(
                {
                    "resonator_frequency_hz": rr_freq,
                    "ground_i_v": selected.Ig.values,
                    "ground_q_v": selected.Qg.values,
                    "driven_i_v": selected.Id.values,
                    "driven_q_v": selected.Qd.values,
                    "ground_abs_v": ground,
                    "driven_abs_v": driven,
                    "complex_separation_v": separation,
                }
            )
            stem = f"amp_{amp_factor:.2f}_qoffset_{q_offset / 1e6:+.0f}MHz"
            table.to_csv(experiments_dir / f"{stem}.csv", index=False)
            with open(experiments_dir / f"{stem}.json", "w", encoding="ascii") as file:
                json.dump(row, file, indent=2)

    summary = pd.DataFrame(rows)
    strongest = summary.loc[
        summary.groupby("resonator_amplitude_factor")["max_complex_separation_v"].idxmax()
    ].copy()
    strongest["apparent_qubit_frequency_hz"] = (
        base_q + strongest["qubit_frequency_offset_hz"]
    )
    strongest.to_csv(output_dir / "apparent_qubit_frequency_by_readout_amplitude.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    dataset.to_netcdf(output_dir / "full_scan.nc")

    response_grid = summary.pivot(
        index="qubit_frequency_offset_hz",
        columns="resonator_amplitude_factor",
        values="max_complex_separation_v",
    )
    shift_grid = summary.pivot(
        index="qubit_frequency_offset_hz",
        columns="resonator_amplitude_factor",
        values="apparent_resonator_shift_hz",
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    extent = [
        READOUT_AMPLITUDE_FACTORS.min(),
        READOUT_AMPLITUDE_FACTORS.max(),
        QUBIT_OFFSETS_HZ.min() / 1e6,
        QUBIT_OFFSETS_HZ.max() / 1e6,
    ]
    image = axes[0].imshow(
        response_grid.values * 1e3,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    axes[0].set_title("Maximum complex-IQ separation")
    axes[0].set_xlabel("Readout amplitude factor")
    axes[0].set_ylabel("Qubit frequency offset [MHz]")
    fig.colorbar(image, ax=axes[0], label="Separation [mV]")

    image = axes[1].imshow(
        shift_grid.values / 1e6,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
    )
    axes[1].set_title("Driven minus ground resonator minimum")
    axes[1].set_xlabel("Readout amplitude factor")
    axes[1].set_ylabel("Qubit frequency offset [MHz]")
    fig.colorbar(image, ax=axes[1], label="Apparent resonator shift [MHz]")
    fig.tight_layout()
    fig.savefig(output_dir / "scan_heatmaps.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        strongest["resonator_pulse_amplitude"],
        strongest["apparent_qubit_frequency_hz"] / 1e9,
        "o-",
    )
    ax.set_xlabel("Actual resonator pulse amplitude")
    ax.set_ylabel("Apparent qubit frequency [GHz]")
    ax.set_title("Strongest-response qubit frequency vs readout amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "apparent_qubit_frequency.png", dpi=180)
    plt.close(fig)
    return summary


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data") / "resonator_parameter_scan" / timestamp
    machine = create_machine()
    qubit = machine.qubits[QUBIT_NAME]
    machine.connect()
    machine.qmm.close_all_qms()

    print(f"Saving results to {output_dir.resolve()}")
    print(f"Running {len(READOUT_AMPLITUDE_FACTORS) * len(QUBIT_OFFSETS_HZ)} combinations")
    dataset = acquire(machine, qubit)
    summary = analyze_and_save(dataset, output_dir)
    print(summary.to_string(index=False))
    print(f"Completed scan: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
