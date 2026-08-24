from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimulationParameters:
    """q1 defaults for the echo-Lorentzian experiment, plus simulator knobs."""

    num_shots: int = 100
    operation: str = "lorentzian"
    pulse_shape: str = "lorentzian"
    lorentzian_length_in_ns: int = 40
    waveform_template_length_in_ns: int | None = None
    lorentzian_peak_amplitude: float = 0.15127819777954318
    cutoff: float = 0.2
    echo: bool = False
    min_amp_factor: float = 0.0
    max_amp_factor: float = 1.0
    amp_factor_step: float = 0.03
    amp_factor_points: int | None = None
    amp_factor_spacing: str = "linear"
    frequency_span_in_mhz: float = 850
    frequency_step_in_mhz: float = 2
    frequency_points: int | None = None

    qubit_name: str = "q1"
    rf_frequency_hz: float = 4267100311.915768
    x180_amplitude: float = 0.15127819777954318
    x180_length_in_ns: float = 40.0
    t1_in_us: float | None = 30.87
    t2_star_in_us: float | None = 6.855588134130561
    num_levels: int = 2
    anharmonicity_hz: float | None = None
    num_time_points: int = 1000
    output_dir: Path = Path("Projects/shaped_pulse_spectroscopy/simulation/qutip/output")
