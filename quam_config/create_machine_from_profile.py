"""Create a complete QuAM machine from a validated device profile."""

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from pprint import pprint

# Allow direct execution and `python -m quam_config.create_machine_from_profile`.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from profiles import load_profile
from quam_config import Quam
from quam_config.populate_quam_lf_mw_fems import apply_profile
from quam_config.wiring_lffem_mwfem import create_profile_connectivity

from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam


DEFAULT_PROFILE = "main"


@contextmanager
def _build_directory(save: bool):
    """Keep builder-generated intermediate files away from the repo if needed."""
    if save:
        yield
        return

    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="quam-profile-build-") as directory:
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(original_directory)


def create_machine_from_profile(
    profile_name: str = DEFAULT_PROFILE,
    save: bool = True,
) -> Quam:
    """Build wiring, create the base QuAM, and apply all profile parameters."""
    profile = load_profile(profile_name)
    network = profile["connectivity"]["network"]
    connectivity, _ = create_profile_connectivity(profile)

    with _build_directory(save):
        machine = Quam()
        build_quam_wiring(
            connectivity=connectivity,
            host_ip=network["host"],
            cluster_name=network["cluster_name"],
            quam_instance=machine,
            port=network["port"],
        )
        build_quam(machine)
        apply_profile(machine, profile)

        if save:
            machine.save()
    return machine


def main() -> Quam:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Profile folder name under profiles/",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Create and validate the machine without writing state.json/wiring.json.",
    )
    args = parser.parse_args()

    machine = create_machine_from_profile(args.profile, save=not args.no_save)
    machine.generate_config()
    pprint(
        f"Created machine from profile {args.profile!r}: "
        f"{len(machine.active_qubit_names)} active qubit(s)."
    )


    pprint(machine.qubits["q9"])
    return machine


if __name__ == "__main__":
    main()
