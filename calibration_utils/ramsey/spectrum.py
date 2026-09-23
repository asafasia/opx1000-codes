"""Shared Ramsey FFT preprocessing; raw/time-domain fitting data are unchanged."""
import numpy as np
from scipy.signal import detrend


def ramsey_spectrum(time, signal):
    """Return positive frequencies (inverse input-time units) after linear detrending.

    Removing the fitted constant and slope suppresses baseline drift, not just
    the DC bin. Positive low-frequency content is retained; it is not masked.
    Irregular samples are interpolated onto a uniform grid before the FFT.
    Twofold zero-padding refines the frequency grid without increasing the
    acquisition duration or the true resolving power.
    """
    time, signal = np.asarray(time, dtype=float), np.asarray(signal, dtype=float)
    valid = np.isfinite(time) & np.isfinite(signal)
    time, signal = time[valid], signal[valid]
    order = np.argsort(time)
    time, signal = time[order], signal[order]
    if time.size < 3 or np.any(np.diff(time) <= 0):
        return np.array([]), np.array([])
    uniform_time = np.linspace(time[0], time[-1], time.size)
    uniform_signal = np.interp(uniform_time, time, signal)
    centered = detrend(uniform_signal, type="linear")
    fft_size = 2 * time.size
    frequencies = np.fft.rfftfreq(fft_size, d=uniform_time[1] - uniform_time[0])
    amplitudes = np.abs(np.fft.rfft(centered, n=fft_size))
    return frequencies[1:], amplitudes[1:]
