"""Close all open quantum machines for the configured QOP cluster."""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from quam_config import create_machine


def close_all_qms(profile_name: str | None = None) -> None:
    """Connect to the profile's QOP cluster and close all open QMs."""
    machine = create_machine(profile_name)
    qmm = machine.connect()
    qmm.close_all_qms()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile folder under profiles/. Defaults to QUAM_PROFILE or main.",
    )
    args = parser.parse_args()

    close_all_qms(args.profile)
    profile = args.profile or "QUAM_PROFILE/main"
    print(f"Closed all quantum machines using profile {profile!r}.")


if __name__ == "__main__":
    main()
