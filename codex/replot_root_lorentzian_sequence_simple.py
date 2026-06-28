from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replot a root-Lorentzian length sequence from saved npz data."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("codex/root_lorentzian_sequence/20260623_153747/sequence_manifest.json"),
        help="Sequence manifest containing run_dirs and lengths_us.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults next to the manifest.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("last_figs_preview/root_lorentzian_length_sequence_echo_true_simple.jpg"),
        help="Small JPG preview path.",
    )
    return parser


def load_panel(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    results = np.load(run_dir / "results.npz")
    sweep = np.load(run_dir / "sweep.npz")
    state = np.asarray(results["state"], dtype=float)[0]
    detuning_mhz = np.asarray(sweep["detuning"], dtype=float) / 1e6
    amp_prefactor = np.asarray(sweep["amp_prefactor"], dtype=float)
    return state, detuning_mhz, amp_prefactor


def make_simple_sequence_plot(manifest_path: Path, output_path: Path, preview_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path.cwd()
    run_dirs = [
        path if path.is_absolute() else root / path
        for path in (Path(item) for item in manifest["run_dirs"])
    ]
    lengths_us = manifest["lengths_us"]

    panels = [load_panel(run_dir) for run_dir in run_dirs]
    vmin = min(float(np.nanmin(state)) for state, _, _ in panels)
    vmax = max(float(np.nanmax(state)) for state, _, _ in panels)

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.2), sharex=True, sharey=True)
    last_image = None
    for ax, (state, detuning_mhz, amp_prefactor), length_us in zip(
        axes.ravel(), panels, lengths_us
    ):
        extent = [
            float(detuning_mhz[0]),
            float(detuning_mhz[-1]),
            float(amp_prefactor[0]),
            float(amp_prefactor[-1]),
        ]
        last_image = ax.imshow(
            state.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.axvline(0.0, color="white", lw=0.7, alpha=0.8)
        ax.set_title(f"{length_us:g} us", fontsize=11)
        ax.tick_params(labelsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Detuning [MHz]", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude prefactor", fontsize=9)

    title = (
        f"{manifest['qubit']} root-Lorentzian length sequence"
        f" echo={manifest.get('echo')}"
    )
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.91, wspace=0.12, hspace=0.26)
    colorbar_axis = fig.add_axes([0.905, 0.16, 0.018, 0.68])
    fig.colorbar(last_image, cax=colorbar_axis, label="Measured state")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path, dpi=110)
    plt.close(fig)
    print(f"Saved simple PNG to {output_path.resolve()}")
    print(f"Saved simple preview to {preview_path.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    output = args.output
    if output is None:
        output = args.manifest.parent / "root_lorentzian_length_sequence_3x3_simple.png"
    make_simple_sequence_plot(args.manifest, output, args.preview)


if __name__ == "__main__":
    main()
