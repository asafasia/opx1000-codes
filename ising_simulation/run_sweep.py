"""Run a practical temperature sweep around the 2D Ising critical point."""

from pathlib import Path
import sys
import time

import numpy as np

if __package__:
    from .sweep import save_csv, save_plot, temperature_sweep
else:
    # Support direct execution: python ising_simulation/run_sweep.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ising_simulation.sweep import save_csv, save_plot, temperature_sweep


def main() -> None:
    
    t1 = time.time()
    temperatures = np.concatenate(
        [
            np.linspace(1.5, 2.0, 5, endpoint=False),
            np.linspace(2.0, 2.6, 13, endpoint=False),
            np.linspace(2.6, 3.5, 7),
        ]
    )
    rows = temperature_sweep(
        temperatures,
        size=30,
        equilibration_sweeps=500,
        sample_sweeps=1_000,
        sample_interval=5,
        seed=7,
    )
    t2 = time.time()
    print(f"Completed temperature sweep in {t2 - t1:.2f} seconds.")
    output_dir = Path(__file__).resolve().parents[1] / "data" / "ising_simulation"
    save_csv(rows, output_dir / "temperature_sweep.csv")
    save_plot(rows, output_dir / "temperature_sweep.png")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
