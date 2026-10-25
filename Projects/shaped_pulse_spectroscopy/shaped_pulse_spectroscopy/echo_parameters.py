"""Shared settings for current echo experiments. Edit the defaults here.

Scan axes and intentional CLI/plan overrides remain local to each experiment.
Saved data and historical, explicit campaign specifications are not rewritten.
Importing this module never connects to hardware or changes a device profile.
"""

from dataclasses import asdict, dataclass
import math


DEFAULT_QUBIT = "q1"


@dataclass(frozen=True)
class EchoPulse:
    pulse_shape: str = "root_lorentzian"
    echo: bool = True
    lorentzian_length_in_ns: int = 20_000
    waveform_template_length_in_ns: int = 20_000
    lorentzian_peak_amplitude: float = 0.7
    cutoff: float = 0.001
    # Minimum of the RMS-squared fit to the 2026-10-22, 200-beta q1 scan.
    drag_beta: float = -1.0972249303804549
    echo_transition_time_ns: float = 16.0
    edge_smoothing: str = "polynomial"
    edge_smoothing_power: int = 5
    ac_stark_correction: bool = False
    stark_kappa_mhz_inv: float = 0.0
    stark_chirp_max_error_hz: float = 10.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()
                   if isinstance(v, (int, float))):
            raise ValueError("Echo pulse parameters must be finite.")
        if not 0 < self.lorentzian_peak_amplitude <= 1.0:
            raise ValueError("Echo nominal amplitude must be in (0, 1.0] V.")
        if not 0 < self.cutoff <= 1:
            raise ValueError("Echo cutoff must be in (0, 1].")
        for length in (self.lorentzian_length_in_ns, self.waveform_template_length_in_ns):
            if length < 16 or length % 4:
                raise ValueError("Echo lengths must be at least 16 ns and multiples of 4 ns.")
        if self.waveform_template_length_in_ns > self.lorentzian_length_in_ns:
            raise ValueError("Echo template cannot be longer than the physical pulse.")
        if not 0 <= self.echo_transition_time_ns <= self.lorentzian_length_in_ns:
            raise ValueError("Echo transition must fit within the pulse.")
        if self.pulse_shape not in ("lorentzian", "root_lorentzian", "gaussian"):
            raise ValueError("Unknown pulse shape.")
        if self.edge_smoothing not in ("none", "polynomial"):
            raise ValueError("Unknown edge smoothing option.")
        if self.edge_smoothing == "polynomial" and self.pulse_shape != "root_lorentzian":
            raise ValueError("Polynomial edge smoothing requires root_lorentzian.")
        if self.edge_smoothing_power < 1 or int(self.edge_smoothing_power) != self.edge_smoothing_power:
            raise ValueError("Polynomial edge power must be a positive integer.")


ECHO_PULSE = EchoPulse()

# Fast amplitude-detuning map, including both endpoints. Keep zero amplitude
# out of the fitted-center score: it has no spectroscopy dip to fit.
FAST_ECHO_SCAN = dict(
    min_amp_factor=0.2,
    max_amp_factor=1.0,
    amp_factor_points=13,
    amp_factor_spacing="linear",
    frequency_span_in_mhz=1.0,
    frequency_points=81,
    num_shots=200,
    fit_fwhm=False,
)


def echo_parameters(**overrides):
    """Return fresh Parameters with the current pulse and explicit overrides.

    This does not set scan axes. Use fast_echo_parameters() for the fast 2D map.
    Complete I/Q headroom still needs checking with the selected device before
    acquisition; nominal amplitude alone is not a complete voltage check.
    """
    from .parameters import Parameters

    values = dict(
        simulate=False, use_state_discrimination=True, reset_type="active",
        use_readout_mitigation=0, **asdict(ECHO_PULSE),
    )
    values.update(overrides)
    # Validate the final waveform, including caller overrides, before use.
    EchoPulse(**{name: values[name] for name in EchoPulse.__dataclass_fields__})
    return Parameters(**values)


def fast_echo_parameters(**overrides):
    """Current echo pulse on the shared, relatively small 2D grid."""
    values = {**FAST_ECHO_SCAN, **overrides}
    if values.get("frequency_points", 0) and values["frequency_points"] > 1:
        values.setdefault("frequency_step_in_mhz", values["frequency_span_in_mhz"] / (values["frequency_points"]-1))
    if values.get("amp_factor_points", 0) and values["amp_factor_points"] > 1:
        values.setdefault("amp_factor_step", (values["max_amp_factor"]-values["min_amp_factor"]) / (values["amp_factor_points"]-1))
    return echo_parameters(**values)
