from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from utils.plotting_settings import FIGURE_SIZE


u = unit(coerce_to_integer=True)


def plot_raw_data(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    use_state_discrimination: bool = False,
    *,
    ds_fit: xr.Dataset | None = None,
):
    """Plot the frequency-versus-amplitude Rabi chevron."""
    variables = ("state",) if use_state_discrimination else ("I", "Q")
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Power-Rabi-chevron plot expected {missing!r} for "
            f"use_state_discrimination={use_state_discrimination}, "
            f"but dataset contains {list(ds.data_vars)}"
        )

    variables = ("state",) if use_state_discrimination else ("I", "Q")
    figure, axes = plt.subplots(
        len(variables) * len(qubits),
        1,
        squeeze=False,
        figsize=FIGURE_SIZE,
    )

    for qubit_index, qubit in enumerate(qubits):
        for variable_index, variable in enumerate(variables):
            plot_individual_data_with(
                axes[len(variables) * qubit_index + variable_index, 0],
                ds,
                {"qubit": qubit.name},
                use_state_discrimination=use_state_discrimination,
                variable=variable,
                ds_fit=ds_fit,
            )

    figure.suptitle(
        "Power Rabi chevron: measured state"
        if use_state_discrimination
        else "Power Rabi chevron: I and Q quadratures"
    )
    figure.tight_layout()
    return figure


def plot_individual_data_with(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    use_state_discrimination: bool = False,
    variable: str | None = None,
    *,
    ds_fit: xr.Dataset | None = None,
):
    """Plot one power-Rabi chevron panel with the same axes style as 04a."""
    data = variable or ("state" if use_state_discrimination else "I")
    expected_variables = ("state",) if use_state_discrimination else ("I", "Q")
    if data not in expected_variables:
        raise ValueError(
            f"Power-Rabi-chevron variable {data!r} is incompatible with "
            f"use_state_discrimination={use_state_discrimination}"
        )
    if data not in ds:
        raise RuntimeError(
            f"Power-Rabi-chevron plot expected {data!r} for "
            f"use_state_discrimination={use_state_discrimination}, but dataset contains {list(ds.data_vars)}"
        )

    selected = ds.sel(qubit=qubit["qubit"])
    normalized_amplitude = selected.full_amp.attrs.get("units", "V") == "a.u."
    amplitude_scale = 1.0 if normalized_amplitude else u.mV
    has_rabi = (
        "rabi_frequency_hz" in selected
        and np.all(np.isfinite(selected.rabi_frequency_hz.values))
    )
    y_coord = "rabi_frequency_MHz" if has_rabi else "amplitude_display"
    selected = selected.assign_coords(
        full_freq_GHz=selected.full_freq / u.GHz,
        amplitude_display=selected.full_amp / amplitude_scale,
    )
    if has_rabi:
        selected = selected.assign_coords(
            rabi_frequency_MHz=selected.rabi_frequency_hz / u.MHz,
        )
    scale = 1 if data == "state" else 1 / u.mV
    data_label = "Measured state" if data == "state" else f"{data} [mV]"

    plotted = (selected[data] * scale).plot(
        ax=ax,
        x="full_freq_GHz",
        y=y_coord,
        add_colorbar=True,
        robust=True,
    )
    plotted.colorbar.set_label(data_label)
    ax.set_title(f"{qubit['qubit']}: {data_label}")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(
        "Rabi frequency [MHz]" if has_rabi else
        ("Pulse amplitude [a.u.]" if normalized_amplitude else "Pulse amplitude [mV]")
    )
    if has_rabi:
        _add_amplitude_yaxis(ax, ds, qubit["qubit"])

    ax2 = ax.twiny()
    (selected[data] * scale).assign_coords(
        detuning_MHz=selected.detuning / u.MHz
    ).plot(
        ax=ax2,
        x="detuning_MHz",
        y=y_coord,
        add_colorbar=False,
        robust=True,
    )
    ax2.set_title("")
    ax2.set_xlabel("Detuning [MHz]")
    _overlay_peak_fit(ax2, ds_fit, qubit["qubit"], np.asarray(selected[y_coord].values))


def _add_amplitude_yaxis(ax: Axes, ds: xr.Dataset, qubit_name: str) -> None:
    full_amp = ds.sel(qubit=qubit_name).full_amp
    rabi_frequency_hz = ds.sel(qubit=qubit_name).rabi_frequency_hz
    finite = (
        np.isfinite(np.asarray(full_amp.values, dtype=float))
        & np.isfinite(np.asarray(rabi_frequency_hz.values, dtype=float))
        & (np.asarray(rabi_frequency_hz.values, dtype=float) != 0)
    )
    if not np.any(finite):
        return

    amp_per_hz = float(
        np.mean(
            np.asarray(full_amp.values, dtype=float)[finite]
            / np.asarray(rabi_frequency_hz.values, dtype=float)[finite]
        )
    )

    normalized_amplitude = full_amp.attrs.get("units", "V") == "a.u."
    amplitude_scale = 1.0 if normalized_amplitude else u.mV

    def rabi_mhz_to_amp_mv(rabi_mhz):
        return np.asarray(rabi_mhz) * u.MHz * amp_per_hz / amplitude_scale

    def amp_mv_to_rabi_mhz(amp_mv):
        return np.asarray(amp_mv) * amplitude_scale / amp_per_hz / u.MHz

    right_axis = ax.secondary_yaxis(
        "right",
        functions=(rabi_mhz_to_amp_mv, amp_mv_to_rabi_mhz),
    )
    right_axis.set_ylabel("Pulse amplitude [a.u.]" if normalized_amplitude else "Pulse amplitude [mV]")


def _overlay_peak_fit(
    ax: Axes, ds_fit: xr.Dataset | None, qubit_name: str, y_values: np.ndarray,
) -> None:
    """Draw on the top heatmap axes so its heatmap cannot hide the fit."""
    required = {"peak_detuning_hz", "peak_valid"}
    if ds_fit is None or not required.issubset(ds_fit.variables):
        return
    if qubit_name not in ds_fit.qubit.values:
        return
    selected = ds_fit.sel(qubit=qubit_name)
    peaks = np.asarray(selected.peak_detuning_hz.values, dtype=float) / u.MHz
    rabi = np.asarray(y_values, dtype=float)
    valid = np.asarray(selected.peak_valid.values, dtype=bool).copy()
    valid &= np.isfinite(peaks) & np.isfinite(rabi)
    if not np.any(valid):
        return
    used = np.asarray(selected.get("peak_fit_used", selected.peak_valid), dtype=bool)
    limits = ax.get_xlim(), ax.get_ylim()
    errors = _peak_errors_mhz(selected)
    if np.any(valid & used):
        ax.errorbar(
            peaks[valid & used], rabi[valid & used],
            xerr=errors[valid & used], fmt="o", markersize=3.5,
            markerfacecolor="white", markeredgecolor="black", ecolor="white",
            elinewidth=0.8, capsize=0, linestyle="none", zorder=5,
            label="Spectral peak used in fit",
        )
    if np.any(valid & ~used):
        ax.plot(
            peaks[valid & ~used], rabi[valid & ~used], "x",
            color="white", markersize=4, linestyle="none", zorder=5,
            label="Peak not used in trend fit",
        )
    if "quadratic_fit_detuning_hz" in selected:
        fit = np.asarray(selected.quadratic_fit_detuning_hz.values, dtype=float) / u.MHz
        finite = np.isfinite(fit) & np.isfinite(rabi)
        order = np.argsort(rabi[finite])
        if np.any(finite):
            ax.plot(
                fit[finite][order], rabi[finite][order], color="#fb923c",
                linewidth=1.8, zorder=6, label="Quadratic peak trend",
            )
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.legend(loc="best", fontsize=8, framealpha=0.85)


def _peak_errors_mhz(selected: xr.Dataset) -> np.ndarray:
    if "peak_frequency_std_hz" not in selected:
        return np.zeros(selected.amp_prefactor.size, dtype=float)
    errors = np.asarray(selected.peak_frequency_std_hz.values, dtype=float) / u.MHz
    return np.where(np.isfinite(errors) & (errors >= 0), errors, 0.0)


def _finite_number(value) -> bool:
    return value is not None and np.isscalar(value) and bool(np.isfinite(value))


def _format_estimate(value, error, *, scale: float = 1.0) -> str:
    if not _finite_number(value):
        return "unavailable"
    text = f"{float(value) / scale:.3g}"
    if _finite_number(error):
        text += f" ± {float(error) / scale:.3g}"
    return text


def plot_stark_shift(
    ds_fit: xr.Dataset,
    qubits: List[AnyTransmon],
    fit_results: dict,
):
    """Plot the apparent shift and a conditional weak-drive theory comparison."""
    figure = plt.figure(figsize=(12, 5.2 * max(1, len(qubits))))
    grid = figure.add_gridspec(
        2 * max(1, len(qubits)), 2,
        height_ratios=[4, 0.75] * max(1, len(qubits)),
    )
    figure.suptitle("Power Rabi chevron: apparent AC Stark shift", fontsize=14)

    for index, qubit in enumerate(qubits):
        left = figure.add_subplot(grid[2 * index, 0])
        right = figure.add_subplot(grid[2 * index, 1])
        note = figure.add_subplot(grid[2 * index + 1, :])
        note.set_axis_off()
        result = fit_results.get(qubit.name, {})
        left.set_title(f"{qubit.name}: spectral maximum")
        consistency = result.get("quadratic_consistency", "inconclusive")
        right.set_title(
            f"Quadratic test: {consistency}" if result.get("success", False)
            else "Quadratic test: no reliable fit"
        )
        left.set_ylabel("Peak detuning from configured frequency [MHz]")
        right.set_ylabel("Peak shift from fitted zero-amplitude intercept [MHz]")
        for ax in (left, right):
            ax.grid(alpha=0.2)
            ax.axhline(0, color="0.65", linewidth=0.8, zorder=0)

        required = {"peak_detuning_hz", "peak_valid", "full_amp"}
        if not required.issubset(ds_fit.variables) or qubit.name not in ds_fit.qubit.values:
            note.text(
                0, 0.9, result.get("reason", "Spectral-peak fitting is disabled."),
                va="top", fontsize=9,
            )
            continue

        selected = ds_fit.sel(qubit=qubit.name)
        amplitude = np.asarray(selected.full_amp.values, dtype=float)
        rabi_hz = np.asarray(
            selected.rabi_frequency_hz.values
            if "rabi_frequency_hz" in selected else np.full_like(amplitude, np.nan),
            dtype=float,
        )
        use_rabi = np.count_nonzero(np.isfinite(rabi_hz)) >= 2
        amplitude_units = selected.full_amp.attrs.get("units", "V")
        x = np.abs(rabi_hz) / u.MHz if use_rabi else np.abs(amplitude)
        left.set_xlabel(
            r"Peak Rabi frequency $f_R=\Omega/(2\pi)$ [MHz]"
            if use_rabi else f"Pulse amplitude magnitude [{amplitude_units}]"
        )
        peak = np.asarray(selected.peak_detuning_hz.values, dtype=float) / u.MHz
        errors = _peak_errors_mhz(selected)
        valid = np.asarray(selected.peak_valid.values, dtype=bool).copy()
        valid &= np.isfinite(x) & np.isfinite(peak)
        used = valid & np.asarray(selected.get("peak_fit_used", selected.peak_valid), dtype=bool)
        fit = np.asarray(
            selected.quadratic_fit_detuning_hz.values
            if "quadratic_fit_detuning_hz" in selected else np.full_like(amplitude, np.nan),
            dtype=float,
        ) / u.MHz
        _plot_peaks_and_trend(left, x, peak, errors, valid, used, fit)

        anharmonicity = result.get("signed_anharmonicity_hz")
        use_physical_scale = use_rabi and _finite_number(anharmonicity) and anharmonicity != 0
        if use_physical_scale:
            scale = rabi_hz**2 / abs(anharmonicity) / u.MHz
            right.set_xlabel(r"$f_R^2/|\Delta f|$ [MHz]")
        else:
            scale = amplitude**2
            right.set_xlabel(f"Pulse amplitude squared [{amplitude_units}²]")

        intercept = result.get("intercept_hz")
        if _finite_number(intercept):
            offset = float(intercept) / u.MHz
            _plot_peaks_and_trend(
                right, scale, peak - offset, errors, valid, used, fit - offset,
            )
            if result.get("theory_applicable", False) and use_physical_scale:
                ordered = np.sort(scale[np.isfinite(scale)])
                right.plot(
                    ordered, 0.5 * ordered, "--", color="#7c3aed", linewidth=1.7,
                    label=r"Weak-drive reference: $\frac{1}{2} f_R^2/|\Delta f|$",
                )
        else:
            right.text(
                0.5, 0.5, "No reliable quadratic fit\nZero-amplitude intercept unavailable",
                ha="center", va="center", transform=right.transAxes, color="0.35",
            )

        for ax in (left, right):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="best", fontsize=8, framealpha=0.9)

        if result.get("success", False):
            stats = [
                "Quadratic R² = " + _format_estimate(result.get("r_squared"), None),
                "Power exponent p = " + _format_estimate(
                    result.get("exponent"), result.get("exponent_std"),
                ),
                "Quadratic consistency: " + result.get("quadratic_consistency", "inconclusive"),
            ]
            if use_physical_scale:
                stats.append(
                    "Coefficient C = " + _format_estimate(
                        result.get("theory_coefficient"), result.get("theory_coefficient_std"),
                    )
                )
            else:
                stats.append(
                    "Coefficient [MHz/amplitude²] = " + _format_estimate(
                        result.get("quadratic_coefficient_hz_per_amplitude2"),
                        result.get("quadratic_coefficient_std_hz_per_amplitude2"),
                        scale=u.MHz,
                    )
                )
            line1 = "  |  ".join(stats)
            line2 = (
                "C is defined by shift = C f_R²/|Δf|. " if use_physical_scale else ""
            ) + result.get("theory_reason", "Theory comparison unavailable.")
        else:
            line1 = "No reliable trend fit: " + result.get(
                "reason", "Insufficient valid spectral peaks.",
            )
            line2 = result.get("theory_reason", "")
        note.text(0, 0.92, line1, fontsize=9, va="top", wrap=True)
        note.text(0, 0.35, line2, fontsize=8.5, color="0.35", va="top", wrap=True)

    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.4, w_pad=2.0)
    return figure


def _plot_peaks_and_trend(
    ax: Axes, x: np.ndarray, peak: np.ndarray, errors: np.ndarray,
    valid: np.ndarray, used: np.ndarray, fit: np.ndarray,
) -> None:
    finite = np.isfinite(x)
    selected = used & finite
    if np.any(selected):
        ax.errorbar(
            x[selected], peak[selected], yerr=errors[selected], fmt="o", markersize=4,
            color="#0369a1", ecolor="#7dd3fc", elinewidth=1.0, capsize=2,
            label="Spectral peak used in fit",
        )
    excluded = valid & ~used & finite
    if np.any(excluded):
        ax.errorbar(
            x[excluded], peak[excluded], yerr=errors[excluded], fmt="x", markersize=4,
            color="0.6", elinewidth=0.8, capsize=2, label="Peak not used in trend fit",
        )
    finite &= np.isfinite(fit)
    if np.any(finite):
        order = np.argsort(x[finite])
        fit_x, fit_y = x[finite][order], fit[finite][order]
        supported = np.ones(fit_x.shape, dtype=bool)
        if np.any(selected):
            supported = (fit_x >= np.min(x[selected])) & (fit_x <= np.max(x[selected]))
        if np.any(~supported):
            ax.plot(
                fit_x, fit_y, "--", color="#ea580c", linewidth=1.2,
                label="Fit extrapolation",
            )
        ax.plot(
            fit_x[supported], fit_y[supported], color="#ea580c",
            linewidth=1.8, label="Quadratic fit",
        )
