from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimulationParameters:
    """Parameters matching the echo-Lorentzian experiment, plus simulator knobs."""

    num_shots: int = 100
    operation: str = "lorentzian"
    pulse_shape: str = "lorentzian"
    lorentzian_length_in_ns: int = 40
    waveform_template_length_in_ns: int | None = None
    lorentzian_tau_in_ns: float = 8.0
    lorentzian_peak_amplitude: float = 0.1
    cutoff: float = 0.2
    echo: bool = False
    min_amp_factor: float = 0.0
    max_amp_factor: float = 1.0
    amp_factor_step: float = 0.03
    frequency_span_in_mhz: float = 850
    frequency_step_in_mhz: float = 2

    qubit_name: str = "q_sim"
    rf_frequency_hz: float = 4.1e9
    x180_amplitude: float = 0.1
    x180_length_in_ns: float = 40.0
    t1_in_us: float | None = None
    t2_in_us: float | None = None
    output_dir: Path = Path("Projects/echo-lorentzian-qutip-simulation/output")
