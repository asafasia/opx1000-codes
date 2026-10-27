from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

def plot_pi_train(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    use_state_discrimination: bool,
    operation: str = "x180",
    *,
    gate_label: str | None = None,
    num_shots: int | None = None,
    t1_ns: float | None = None,
    pulse_duration_ns: float | None = None,
) -> Figure:
    """Plot pulse-count branches, optionally with a gapless driven T1 envelope.

    The reference uses exp(-3*t/(4*T1)), the resonant strong-drive limit
    with no pure dephasing. Each branch starts at its first measured point;
    it is a comparison envelope, not a fit or a shaped-pulse simulation.
    """
    show_t1 = t1_ns is not None and pulse_duration_ns is not None
    if show_t1 and (not np.isfinite(t1_ns) or not np.isfinite(pulse_duration_ns)
                    or t1_ns <= 0 or pulse_duration_ns <= 0):
        raise ValueError("T1 and pulse duration must be finite and positive.")
    variables = ("state",) if use_state_discrimination else ("I", "Q")
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Pi-train plot expected {variables}, but dataset contains {list(ds.data_vars)}"
        )

    figure, axes = plt.subplots(
        len(qubits) * len(variables),
        1,
        figsize=(11, 5.5 * len(qubits) * len(variables)),
        squeeze=False,
    )
    axis_index = 0
    for qubit in qubits:
        selected = ds.sel(qubit=qubit.name)
        for variable in variables:
            axis = axes[axis_index, 0]
            counts = np.asarray(ds.number_of_pulses.values, dtype=int)
            values = np.asarray(selected[variable].values).reshape(-1)
            is_pi = operation == "x180" or operation.startswith("x180_")
            period = 2 if is_pi else 4
            colors = ("#2563EB", "#D97706", "#0F766E", "#9333EA")
            for branch in range(period):
                mask = counts % period == branch
                if not mask.any():
                    continue
                label = (
                    ("Even counts · return to |g⟩" if branch == 0 else "Odd counts · return to |e⟩")
                    if is_pi else f"Count = {branch} mod 4"
                )
                uncertainty = None
                if variable == "state" and num_shots is not None and num_shots > 0:
                    population = values[mask]
                    if np.all((population >= 0) & (population <= 1)):
                        uncertainty = np.sqrt(population * (1 - population) / num_shots)
                axis.errorbar(
                    counts[mask], values[mask], yerr=uncertainty,
                    fmt="o-", markersize=4, linewidth=1.4,
                    elinewidth=0.8, capsize=0, color=colors[branch],
                    label=label, zorder=3,
                )
                if show_t1 and variable == "state" and is_pi:
                    start = np.flatnonzero(mask)[0]
                    reference_counts = np.linspace(counts[start], counts.max(), 300)
                    elapsed_ns = (reference_counts - counts[start]) * pulse_duration_ns
                    reference = 0.5 + (values[start] - 0.5) * np.exp(-0.75 * elapsed_ns / t1_ns)
                    axis.plot(
                        reference_counts, reference, color=colors[branch],
                        linestyle="--", linewidth=1.8,
                        label=f"T₁ reference · {'even' if branch == 0 else 'odd'}",
                    )
            if variable == "state":
                for level in ([0, 1] if is_pi else [0, 0.5, 1]):
                    axis.axhline(level, color="0.75", linestyle=":", linewidth=1, zorder=1)
                axis.set_ylim(-0.05, 1.05)
                axis.yaxis.set_major_formatter(PercentFormatter(1))
            label = "Excited-state population" if variable == "state" else f"{variable} [V]"
            axis.set_title(f"{qubit.name} · {gate_label or operation} pulse train", loc="left", fontsize=15, pad=16)
            axis.set_xlabel("Number of consecutive π pulses" if is_pi else "Number of consecutive π/2 pulses", labelpad=10)
            axis.set_ylabel(label)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True, steps=[1, 2, 5, 10], min_n_ticks=2))
            axis.set_xlim(counts.min() - 0.5, counts.max() + 0.5)
            axis.grid(axis="y", alpha=0.15)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("0.75")
            axis.tick_params(length=0, pad=8)
            axis.legend(loc="center", frameon=False, fontsize=10, ncol=2 if show_t1 else 1)
            axis_index += 1

    note = "Dotted lines: ideal populations." if use_state_discrimination else ""
    if num_shots is not None:
        note += f"  {num_shots:,} shots per point."
        if use_state_discrimination:
            note += "  Error bars: ±1 binomial standard error."
    if show_t1:
        note += (f"\nT₁ = {t1_ns / 1000:.2f} µs; reference ∝ exp[−3t/(4T₁)], "
                 f"{pulse_duration_ns:g} ns/pulse, no gaps; anchored to first point of each branch.")
    figure.text(0.08, 0.025, note.strip(), fontsize=9, color="0.4")
    figure.tight_layout(rect=(0, 0.09 if show_t1 else 0.065, 1, 1))
    return figure
