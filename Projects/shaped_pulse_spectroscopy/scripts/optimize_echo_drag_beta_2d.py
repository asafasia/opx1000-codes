"""Scan echo DRAG beta using fast 2D amplitude-detuning maps.

Defaults: 40 betas from 0 to -1.5; the shared fast grid is 13 amplitudes
from 0.2 to 1, 81 detunings over 1 MHz, and 200 averages. Fit every amplitude
with the existing negative-Gaussian fitter; minimize the total center RMS
about the covariance-weighted mean. A failed slice rejects that beta.

Without --execute, only validate waveforms and generate QUA offline.
Save each 2D map, fitted line, raw data, per-amplitude fits, and beta summary.
Device profiles and the shared default beta are never updated automatically.
"""

from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import optimize_echo_drag_beta as optimizer
from shaped_pulse_spectroscopy.echo_parameters import FAST_ECHO_SCAN

DEFAULT_OUTPUT_ROOT = optimizer.REPOSITORY_ROOT / "data" / "echo_drag_beta_2d"


def default_settings():
    return optimizer.SweepSettings(
        beta_min=-1.5, beta_max=0.0, beta_points=40, beta_order="descending",
        amplitude_min=FAST_ECHO_SCAN["min_amp_factor"],
        amplitude_max=FAST_ECHO_SCAN["max_amp_factor"],
        amplitude_points=FAST_ECHO_SCAN["amp_factor_points"],
        amplitude_spacing=FAST_ECHO_SCAN["amp_factor_spacing"],
        frequency_span_mhz=FAST_ECHO_SCAN["frequency_span_in_mhz"],
        frequency_points=FAST_ECHO_SCAN["frequency_points"],
        num_shots=FAST_ECHO_SCAN["num_shots"],
    )


def main(argv=None):
    return optimizer.main(argv, defaults=default_settings(),
                          output_root=DEFAULT_OUTPUT_ROOT, plot_mode="2d", description=__doc__)


if __name__ == "__main__":
    main()
