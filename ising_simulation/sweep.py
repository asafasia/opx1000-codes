"""Temperature sweep and result output for the Ising simulation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .model import IsingModel
from .observables import summarize_samples


def temperature_sweep(
    temperatures: np.ndarray,
    size: int = 24,
    equilibration_sweeps: int = 500,
    sample_sweeps: int = 1_000,
    sample_interval: int = 5,
    seed: int = 7,
) -> list[dict[str, float]]:
    """Simulate each temperature independently and return summary rows."""
    rows = []
    for index, temperature in enumerate(temperatures):
        model = IsingModel(size=size, seed=seed + index)
        samples = model.run(
            float(temperature),
            equilibration_sweeps=equilibration_sweeps,
            sample_sweeps=sample_sweeps,
            sample_interval=sample_interval,
        )
        rows.append(summarize_samples(samples, float(temperature), model.n_spins))
        print(
            f"T={temperature:.3f}  "
            f"E/N={rows[-1]['mean_energy_per_spin']:.3f}  "
            f"|M|/N={rows[-1]['mean_abs_magnetization_per_spin']:.3f}"
        )
    return rows


def save_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict[str, float]], path: Path) -> None:
    """Plot observables that make the finite-size phase transition visible."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Matplotlib is required for plots; install requirements.txt first."
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    temperature = np.asarray([row["temperature"] for row in rows])
    metrics = [
        ("mean_energy_per_spin", "Mean energy / spin"),
        ("mean_abs_magnetization_per_spin", "Mean |magnetization| / spin"),
        ("heat_capacity_per_spin", "Heat capacity / spin"),
        ("susceptibility_per_spin", "Susceptibility / spin"),
        ("nearest_neighbor_correlation", "Nearest-neighbor correlation"),
        ("correlation_length", "Correlation length estimate"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, (key, label) in zip(axes.flat, metrics):
        axis.plot(temperature, [row[key] for row in rows], marker="o", markersize=3)
        axis.axvline(2.269, color="tab:red", linestyle="--", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Temperature")
        axis.grid(alpha=0.25)
    figure.suptitle("2D Ising model temperature sweep")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
