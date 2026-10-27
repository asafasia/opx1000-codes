"""Fit and display echo-spectroscopy diagnostics without acquiring new data."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .gaussian_dip import add_simple_negative_gaussian_fits
from .lorentzian import peak_rabi_frequency_mhz


def add_sweep_diagnostics(ds: xr.Dataset, qubits) -> xr.Dataset:
    """Fit the central dip; retain raw R² and a conservative acceptance mask.

    Contrast is the fitted baseline minus the fitted minimum, in measured-state
    units, not a normalized visibility. The bounded fit uses a one-bin Gaussian
    smoothing and trims up to 40 edge samples, matching the beta-scan analysis.
    Quality is evaluated against the unsmoothed samples in that same window.
    """
    if ds.sizes.get("detuning", 0) < 7:
        return ds
    fitted = add_simple_negative_gaussian_fits(ds)
    names = [name for name in fitted.data_vars if name.startswith("gaussian_negative_")]
    # Keep the optional, more expensive generic FWHM analysis intact.
    result = ds.assign({name.replace("gaussian_negative_", "sweep_"): fitted[name]
                        for name in names if name in {
                            "gaussian_negative_center_hz", "gaussian_negative_offset",
                            "gaussian_negative_fwhm_hz", "gaussian_negative_fit_r_squared",
                            "gaussian_negative_fit_contrast", "gaussian_negative_center_error_hz"}})
    frequency = ds.detuning
    sigma = result.sweep_fwhm_hz / (2 * np.sqrt(2 * np.log(2)))
    prediction = result.sweep_offset - result.sweep_fit_contrast * np.exp(
        -0.5 * ((frequency - result.sweep_center_hz) / sigma) ** 2)
    edge = min(40, max(0, (frequency.size - 7) // 4))
    window = slice(edge, -edge if edge else None)
    observed = ds.state.isel(detuning=window)
    predicted = prediction.isel(detuning=window)
    residual = ((observed - predicted) ** 2).sum("detuning", min_count=1)
    total = ((observed - observed.mean("detuning")) ** 2).sum("detuning", min_count=1)
    r_squared = 1 - residual / total.where(total > 0)
    local_frequency = frequency.isel(detuning=window)
    step = float(np.median(np.diff(np.sort(frequency))))
    accepted = (
        np.isfinite(result.sweep_center_hz)
        & np.isfinite(result.sweep_center_error_hz)
        & (result.sweep_center_error_hz > 0)
        & (result.sweep_fwhm_hz >= 2 * step)
        & (result.sweep_fit_contrast >= 0.02)
        & (r_squared >= 0.1)
        & (result.sweep_center_hz - result.sweep_fwhm_hz / 2 > local_frequency.min())
        & (result.sweep_center_hz + result.sweep_fwhm_hz / 2 < local_frequency.max())
    )
    rabi = []
    by_name = {qubit.name: qubit for qubit in qubits}
    for name in ds.qubit.values:
        rabi.append(peak_rabi_frequency_mhz(ds.sel(qubit=name), by_name[str(name)]).values)
    result = result.assign_coords(sweep_rabi_frequency_mhz=(("qubit", "amp_prefactor"), rabi))
    result = result.assign(sweep_raw_r_squared=r_squared, sweep_fit_accepted=accepted)
    result.sweep_rabi_frequency_mhz.attrs["units"] = "MHz"
    for name in ("sweep_center_hz", "sweep_fwhm_hz", "sweep_center_error_hz"):
        result[name].attrs["units"] = "Hz"
    result.sweep_fit_contrast.attrs.update(
        long_name="Fitted baseline minus dip minimum", units="measured-state units")
    result.attrs["sweep_diagnostic_model"] = "offset - contrast * exp(-0.5*((detuning-center)/sigma)**2)"
    result.attrs["sweep_diagnostic_quality"] = (
        "contrast >= 0.02; raw R2 >= 0.1; FWHM >= 2 bins; both half-height "
        "crossings inside trimmed fit window; finite positive center uncertainty")
    return result


def _representative_indices(values, count):
    values = np.asarray(values)
    targets = np.linspace(np.min(values), np.max(values), min(count, values.size))
    return np.unique([int(np.argmin(abs(values - target))) for target in targets])


def plot_sweep_diagnostics(ds: xr.Dataset) -> dict:
    """Return separately saveable summary and slice figures for every qubit."""
    if "sweep_fit_accepted" not in ds:
        return {}
    figures = {}
    for qubit in ds.qubit.values:
        selected = ds.sel(qubit=qubit)
        rabi = np.asarray(selected.sweep_rabi_frequency_mhz)
        frequency = np.asarray(selected.detuning) / 1e3
        signal = np.asarray(selected.state.transpose("detuning", "amp_prefactor"))
        good = np.asarray(selected.sweep_fit_accepted, dtype=bool)
        limit = float(selected.get("t2_star_fwhm_limit_hz", np.nan)) / 1e3
        title = f"{qubit}: echo dip fits ({good.sum()}/{good.size} accepted)"
        for suffix, variable, scale, ylabel in (
            ("resonance", "sweep_center_hz", 1e-3, "Resonance detuning [kHz]"),
            ("fwhm", "sweep_fwhm_hz", 1e-3, "FWHM [kHz]"),
            ("contrast", "sweep_fit_contrast", 1, "Fitted contrast [measured-state units]"),
        ):
            fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
            values = np.asarray(selected[variable]) * scale
            ax.plot(rabi, np.where(good, values, np.nan), ".-", color="#238a8d", label="Accepted fit")
            invalid = ~good
            if invalid.any():
                # Flag the Rabi position without letting a failed fit set the y range.
                ax.scatter(rabi[invalid], np.full(invalid.sum(), 0.025),
                           transform=ax.get_xaxis_transform(), marker="x", color="0.6",
                           label="Rejected / unavailable (Rabi position)")
            if np.isfinite(limit) and limit > 0:
                if suffix == "resonance":
                    ax.axhline(limit / 2, color="0.3", ls="--", label="T₂* reference: ±1/(2πT₂*)")
                    ax.axhline(-limit / 2, color="0.3", ls="--")
                elif suffix == "fwhm":
                    ax.axhline(limit, color="0.3", ls="--", label="T₂* FWHM: 1/(πT₂*)")
            ax.set(xlabel="Peak Rabi frequency [MHz]", ylabel=ylabel, title=title)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            figures[f"echo_lorentzian_{suffix}_{qubit}"] = fig

        indices = _representative_indices(rabi, 8)
        fig, axes = plt.subplots(int(np.ceil(len(indices) / 4)), min(4, len(indices)),
                                 squeeze=False, figsize=(14, 6), layout="constrained", sharex=True, sharey=True)
        colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(indices)))
        for ax, index, color in zip(axes.flat, indices, colors):
            ax.plot(frequency, signal[:, index], color=color, lw=1, label="Measured")
            if good[index]:
                center = float(selected.sweep_center_hz[index]) / 1e3
                sigma = float(selected.sweep_fwhm_hz[index]) / 1e3 / (2 * np.sqrt(2 * np.log(2)))
                model = float(selected.sweep_offset[index]) - float(selected.sweep_fit_contrast[index]) * np.exp(
                    -0.5 * ((frequency - center) / sigma) ** 2)
                edge = min(40, max(0, (frequency.size - 7) // 4))
                window = slice(edge, -edge if edge else None)
                ax.plot(frequency[window], model[window], color="darkorange", lw=1.6, label="Dip fit")
            else:
                ax.text(0.03, 0.06, "Fit rejected / unavailable", transform=ax.transAxes, fontsize=8)
            if np.isfinite(limit) and limit > 0:
                ax.axvspan(-limit / 2, limit / 2, color="0.5", alpha=0.12)
            ax.set(title=f"{rabi[index]:.2f} MHz", xlabel="Detuning [kHz]", ylabel="Measured state")
            ax.grid(alpha=0.15)
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(indices):]:
            ax.set_visible(False)
        fig.suptitle(f"{qubit}: constant peak-Rabi-frequency slices")
        figures[f"echo_lorentzian_detuning_slices_{qubit}"] = fig

        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        indices = _representative_indices(frequency, 5)
        # Include the closest measured zero-detuning slice even for even grids.
        indices = np.unique(np.append(indices, np.argmin(abs(frequency))))
        for index, color in zip(indices, plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(indices)))):
            ax.plot(rabi, signal[index], color=color, label=f"{frequency[index]:.1f} kHz")
        ax.set(xlabel="Peak Rabi frequency [MHz]", ylabel="Measured state",
               title=f"{qubit}: constant-detuning slices")
        ax.grid(alpha=0.2)
        ax.legend(title="Measured detuning", fontsize=8)
        figures[f"echo_lorentzian_rabi_slices_{qubit}"] = fig
    return figures
