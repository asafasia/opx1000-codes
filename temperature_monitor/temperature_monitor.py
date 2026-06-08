import csv
import math
import re
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
import threading


class TemperatureMonitor:
    def __init__(
        self,
        controller_name="con1",
        poll_interval=1.0,
        max_points=1000,
        warning_temperature=70,
        warning_increase=1.0,
        fem_ids=None,
        include_chassis_and_crps0=False,
        save_dir=Path("data") / "temperature_logs",
        register_exit_handlers=True,
    ):
        self.controller_name = controller_name
        self.poll_interval = poll_interval
        self.max_points = max_points
        self.warning_temperature = warning_temperature
        self.warning_increase = warning_increase

        self.started_at = datetime.now()
        timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        self.plot_date = self.started_at.strftime("%Y-%m-%d %H:%M:%S")
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

        self.temperature_keys = self._select_temperature_keys(
            self.device_temperatures,
            fem_ids=fem_ids,
            include_chassis_and_crps0=include_chassis_and_crps0,
        )
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._save_lock = threading.Lock()
        self._saved = False

        if register_exit_handlers:
            self._register_exit_handlers()

    @staticmethod
    def _select_temperature_keys(temperatures, fem_ids, include_chassis_and_crps0):
        all_keys = list(temperatures.keys())
        if fem_ids is None and not include_chassis_and_crps0:
            return all_keys

        selected = []
        fem_patterns = [
            re.compile(rf"\bfem[\s_:/-]*{re.escape(str(fem_id))}(?!\d)", re.IGNORECASE)
            for fem_id in fem_ids or []
        ]
        crps0_pattern = re.compile(r"\bcrps[\s_:/-]*0(?!\d)", re.IGNORECASE)

        for key in all_keys:
            key_text = str(key)
            is_used_fem = any(pattern.search(key_text) for pattern in fem_patterns)
            is_chassis_or_crps0 = include_chassis_and_crps0 and (
                "chassis" in key_text.lower() or crps0_pattern.search(key_text)
            )
            if is_used_fem or is_chassis_or_crps0:
                selected.append(key)

        if not selected:
            print(
                "WARNING: No temperature sensors matched the requested filter. "
                f"Available sensors: {all_keys}"
            )
            return all_keys

        print(f"Monitoring selected temperature sensors: {selected}")
        return selected

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

    def _initialize_history(self):
        self.start_time = time.time()
        self.elapsed_seconds = []
        self.temperature_history = {key: [] for key in self.temperature_keys}
        self._saved = False

    def sample_once(self):
        current_temperatures = self.read_temperatures()
        self.elapsed_seconds.append(time.time() - self.start_time)

        for key in self.temperature_keys:
            temp = current_temperatures[key]
            self.temperature_history[key].append(temp)

            if temp > self.warning_temperature:
                print(f"WARNING: {key} temperature high ({temp:.2f} °C)")

            increase = temp - self.temperature_history[key][0]
            if increase > self.warning_increase:
                print(
                    f"WARNING: {key} temperature increased by "
                    f"{increase:.2f} °C ({temp:.2f} °C)"
                )

        if self.max_points is not None and len(self.elapsed_seconds) > self.max_points:
            self.elapsed_seconds = self.elapsed_seconds[-self.max_points :]
            for key in self.temperature_keys:
                self.temperature_history[key] = self.temperature_history[key][
                    -self.max_points :
                ]

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                self.sample_once()
            except Exception as exc:
                print(f"ERROR while reading temperatures: {exc}")
            self._stop_event.wait(self.poll_interval)

    def start_background(self):
        """Start headless temperature sampling while another experiment runs."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            raise RuntimeError("Temperature monitoring is already running.")

        self._initialize_history()
        self._stop_event.clear()
        self.sample_once()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="temperature-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        print(f"Temperature monitoring started. Saving outputs to: {self.output_dir}")

    def stop_background(self):
        """Stop background sampling and save the temperature-rise results."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=max(5.0, self.poll_interval * 2))
        self._safe_save()

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
            f.write(f"Started: {self.plot_date}\n")
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
                initial_temp = data[0]
                final_temp = data[-1]
                final_increase = final_temp - initial_temp
                max_increase = max_temp - initial_temp

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
                f.write(f"Initial temp    : {initial_temp:.2f}\n")
                f.write(f"Final increase  : {final_increase:+.2f}\n")
                f.write(f"Maximum increase: {max_increase:+.2f}\n")

                if max_temp > self.warning_temperature:
                    f.write("WARNING: Temperature exceeded threshold!\n")
                if max_increase > self.warning_increase:
                    f.write("WARNING: Temperature increase exceeded threshold!\n")

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

    def create_figure(self):
        num_plots = len(self.temperature_keys)
        num_cols = 3
        num_rows = math.ceil(num_plots / num_cols)
        figure, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(14, 4 * num_rows),
            squeeze=False,
        )
        axes = axes.flatten()

        for i, (axis, key) in enumerate(zip(axes, self.temperature_keys)):
            axis.plot(
                np.asarray(self.elapsed_seconds) / 60,
                self.temperature_history[key],
                ".-",
                lw=2,
                color=f"C{i}",
                markersize=4,
            )
            axis.set_title(key)
            axis.set_xlabel("Elapsed time [min]")
            axis.set_ylabel("Temperature [°C]")
            axis.grid(True, alpha=0.3)

        for axis in axes[num_plots:]:
            axis.set_visible(False)

        figure.suptitle(
            f"Temperature Monitoring - {self.controller_name}\n{self.plot_date}",
            fontsize=16,
        )
        figure.tight_layout()
        return figure

    def _create_live_figure(self):
        num_plots = len(self.temperature_keys)
        num_cols = 2
        num_rows = math.ceil(num_plots / num_cols)
        self.figure, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(13, 4 * num_rows),
            squeeze=False,
        )
        self.figure.suptitle(
            f"Live Temperature Monitoring - {self.controller_name}\n{self.plot_date}"
        )
        axes = axes.flatten()
        self._live_lines = {}

        for i, (axis, key) in enumerate(zip(axes, self.temperature_keys)):
            (line,) = axis.plot([], [], ".-", lw=2, color=f"C{i}", markersize=4)
            axis.set_title(str(key))
            axis.set_xlabel("Elapsed time [min]")
            axis.set_ylabel("Temperature [°C]")
            axis.grid(True, alpha=0.3)
            self._live_lines[key] = (line, axis)

        for axis in axes[num_plots:]:
            axis.set_visible(False)

        self.figure.tight_layout()

    def _update_live_figure(self):
        elapsed_minutes = np.asarray(self.elapsed_seconds) / 60
        for key, (line, axis) in self._live_lines.items():
            line.set_data(elapsed_minutes, self.temperature_history[key])
            axis.relim()
            axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def run_while(self, is_running):
        """Show a live graph and monitor temperatures while work is running."""
        self._initialize_history()
        self._create_live_figure()
        plt.ion()
        plt.show(block=False)
        print(f"Temperature monitoring started. Saving outputs to: {self.output_dir}")

        try:
            while is_running():
                self.sample_once()
                self._update_live_figure()
                plt.pause(self.poll_interval)

            self.sample_once()
            self._update_live_figure()
        except KeyboardInterrupt:
            print("\nStopping temperature monitoring...")
            raise
        finally:
            self._safe_save()
            plt.ioff()

    def run(self):
        self._initialize_history()

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
            f"Temperature Monitoring - {self.controller_name}\n{self.plot_date}",
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
                self.sample_once()

                for key in self.temperature_keys:
                    line, axis = lines[key]

                    line.set_data(
                        np.asarray(self.elapsed_seconds) / 60,
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
            self._safe_save()

            plt.ioff()
            plt.show()

            print("Done.")

    def _safe_save(self):
        """Safely save all data collected so far."""
        try:
            with self._save_lock:
                if self._saved:
                    return
                if not hasattr(self, "elapsed_seconds") or not self.elapsed_seconds:
                    print("No data to save.")
                    return

                print("Auto-saving data...")

                self.save_csv(self.elapsed_seconds, self.temperature_history)
                self.save_report(self.elapsed_seconds, self.temperature_history)

                if not hasattr(self, "figure"):
                    self.figure = self.create_figure()
                self.save_figure(self.figure)

                self._saved = True
                print("Auto-save complete.")

        except Exception as e:
            print(f"ERROR during auto-save: {e}")


if __name__ == "__main__":
    TemperatureMonitor().run()
