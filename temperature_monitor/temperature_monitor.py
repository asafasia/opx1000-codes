import csv
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager

from old.configuration import cluster_name, qop_ip
import atexit
import signal
import sys


class TemperatureMonitor:
    def __init__(
        self,
        controller_name="con1",
        poll_interval=1.0,
        max_points=1000,
        warning_temperature=70,
        save_dir=Path("data") / "temperature_logs",
    ):
        self.controller_name = controller_name
        self.poll_interval = poll_interval
        self.max_points = max_points
        self.warning_temperature = warning_temperature

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(save_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.qmm = QuantumMachinesManager(
            host=qop_ip,
            cluster_name=cluster_name,
            log_level=0,
        )

        self.device_temperatures = (
            self.qmm.get_devices().controllers[self.controller_name].temperatures
        )

        self.temperature_keys = list(self.device_temperatures.keys())
        self._register_exit_handlers()

    def _register_exit_handlers(self):
        """Ensure saving happens on exit, crash, or termination."""

        def handler(*args):
            print("\nExit signal received.")
            self._safe_save()
            raise SystemExit

        # Handle Ctrl+C
        signal.signal(signal.SIGINT, handler)

        # Handle termination (kill, system shutdown, etc.)
        signal.signal(signal.SIGTERM, handler)

        # Last-resort Python exit hook
        atexit.register(self._safe_save)

    def read_temperatures(self):
        temperatures = (
            self.qmm.get_devices().controllers[self.controller_name].temperatures
        )

        return dict(temperatures)

    def save_csv(self, elapsed_seconds, temperature_history):
        csv_path = self.output_dir / "temperature_log.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            header = ["time_s"] + self.temperature_keys
            writer.writerow(header)

            for i, t in enumerate(elapsed_seconds):
                row = [t]

                for key in self.temperature_keys:
                    row.append(temperature_history[key][i])

                writer.writerow(row)

        print(f"CSV log saved to: {csv_path}")

    def save_report(self, elapsed_seconds, temperature_history):
        report_path = self.output_dir / "final_report.txt"

        total_runtime = elapsed_seconds[-1] if elapsed_seconds else 0

        with open(report_path, "w") as f:
            f.write("Temperature Monitoring Report\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Controller: {self.controller_name}\n")
            f.write(f"Runtime: {total_runtime:.1f} s\n")
            f.write(f"Samples: {len(elapsed_seconds)}\n")
            f.write(f"Poll interval: {self.poll_interval} s\n\n")

            for key in self.temperature_keys:
                data = np.array(temperature_history[key])

                if len(data) == 0:
                    continue

                min_temp = np.min(data)
                max_temp = np.max(data)
                avg_temp = np.mean(data)
                std_temp = np.std(data)

                if len(data) > 1:
                    slope = (data[-1] - data[0]) / (
                        elapsed_seconds[-1] - elapsed_seconds[0]
                    )
                else:
                    slope = 0

                f.write(f"{key}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Min temperature : {min_temp:.2f}\n")
                f.write(f"Max temperature : {max_temp:.2f}\n")
                f.write(f"Avg temperature : {avg_temp:.2f}\n")
                f.write(f"Std deviation   : {std_temp:.2f}\n")
                f.write(f"Drift rate      : {slope:.4f} deg/s\n")

                if max_temp > self.warning_temperature:
                    f.write("WARNING: Temperature exceeded threshold!\n")

                f.write("\n")

        print(f"Final report saved to: {report_path}")

    def save_figure(self, figure):
        fig_path = self.output_dir / "temperature_plot.png"

        figure.savefig(
            fig_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Figure saved to: {fig_path}")

    def run(self):
        start_time = time.time()

        self.elapsed_seconds = []
        self.temperature_history = {key: [] for key in self.temperature_keys}

        num_plots = len(self.temperature_keys)
        num_cols = 3
        num_rows = math.ceil(num_plots / num_cols)

        plt.ion()

        figure, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(14, 4 * num_rows),
            squeeze=False,
        )

        self.figure = figure

        figure.suptitle(
            f"Temperature Monitoring - {self.controller_name}",
            fontsize=16,
        )

        axes = axes.flatten()

        lines = {}

        for i, (axis, key) in enumerate(zip(axes, self.temperature_keys)):
            color = f"C{i}"

            (line,) = axis.plot(
                [],
                [],
                ".-",
                lw=2,
                color=color,
                markersize=4,
            )

            axis.set_title(f"{key}")
            axis.set_xlabel("Elapsed time [min]")
            axis.set_ylabel("Temperature [°C]")
            axis.grid(True, alpha=0.3)

            lines[key] = (line, axis)

        for axis in axes[num_plots:]:
            axis.set_visible(False)

        print("Starting temperature monitoring...")
        print(f"Saving outputs to: {self.output_dir}")

        try:
            while True:
                current_time = time.time() - start_time
                current_temperatures = self.read_temperatures()

                self.elapsed_seconds.append(current_time / 60)  # Convert to minutes

                for key in self.temperature_keys:
                    temp = current_temperatures[key]

                    self.temperature_history[key].append(temp)

                    if temp > self.warning_temperature:
                        print(f"WARNING: {key} temperature high " f"({temp:.2f} °C)")

                if len(self.elapsed_seconds) > self.max_points:
                    self.elapsed_seconds = self.elapsed_seconds[-self.max_points :]

                    for key in self.temperature_keys:
                        self.temperature_history[key] = self.temperature_history[key][
                            -self.max_points :
                        ]

                for key in self.temperature_keys:
                    line, axis = lines[key]

                    line.set_data(
                        self.elapsed_seconds,
                        self.temperature_history[key],
                    )

                    axis.relim()
                    axis.autoscale_view()

                figure.tight_layout()

                figure.canvas.draw_idle()

                plt.pause(self.poll_interval)

        except KeyboardInterrupt:
            print("\nStopping temperature monitoring...")

        finally:
            self.save_csv(
                self.elapsed_seconds,
                self.temperature_history,
            )

            self.save_report(
                self.elapsed_seconds,
                self.temperature_history,
            )

            self.save_figure(self.figure)

            plt.ioff()
            plt.show()

            print("Done.")

    def _safe_save(self):
        """Safely save all data collected so far."""
        try:
            if not hasattr(self, "elapsed_seconds") or not self.elapsed_seconds:
                print("No data to save.")
                return

            print("Auto-saving data...")

            self.save_csv(self.elapsed_seconds, self.temperature_history)
            self.save_report(self.elapsed_seconds, self.temperature_history)

            if hasattr(self, "figure"):
                self.save_figure(self.figure)

            print("Auto-save complete.")

        except Exception as e:
            print(f"ERROR during auto-save: {e}")


if __name__ == "__main__":
    TemperatureMonitor().run()
