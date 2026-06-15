"""Select a profile and build strategy for an in-memory QuAM machine."""

import pprint
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from profiles import Profile, ProfileError, set_active_profile

if __package__ in {None, ""}:
    from quam_config.my_quam import Quam
else:
    from .my_quam import Quam


DEFAULT_PROFILE = Profile("main")
SINGLE_QUBIT_PROFILE = "single_qubit"


class CreateMachine:
    """Build a machine while keeping its source profile attached."""

    def __init__(
        self,
        profile: Profile | str = DEFAULT_PROFILE,
        *,
        qubit: str | None = None,
        mode: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        explicit_profile = (
            profile_name is not None
            or mode is not None
            or profile is not DEFAULT_PROFILE
        )
        if profile_name is not None:
            profile = profile_name
        if mode is not None:
            profile = mode

        selected_profile = profile if isinstance(profile, Profile) else Profile(str(profile))
        if (
            qubit is not None
            and selected_profile.name == DEFAULT_PROFILE.name
            and not explicit_profile
        ):
            selected_profile = Profile(SINGLE_QUBIT_PROFILE, qubit=qubit)
        elif qubit is not None:
            selected_profile = selected_profile.for_qubit(qubit)

        if selected_profile.name == SINGLE_QUBIT_PROFILE and selected_profile.qubit is None:
            raise ProfileError("Mode 'single_qubit' requires a qubit selection")
        if selected_profile.name != SINGLE_QUBIT_PROFILE and selected_profile.qubit is not None:
            raise ProfileError(
                f"Profile {selected_profile.name!r} does not support selecting a single qubit"
            )

        if __package__ in {None, ""}:
            from quam_config.create_machine_from_profile import create_machine_from_profile
        else:
            from .create_machine_from_profile import create_machine_from_profile

        self.profile = set_active_profile(selected_profile)
        self.machine = create_machine_from_profile(self.profile, save=False)

    def __getattr__(self, name: str):
        return getattr(self.machine, name)


def create_machine(
    profile: Profile | str = DEFAULT_PROFILE,
    *,
    qubit: str | None = None,
    mode: str | None = None,
    profile_name: str | None = None,
) -> Quam:
    """Create an in-memory full-chip or selected single-qubit machine."""
    return CreateMachine(
        profile,
        qubit=qubit,
        mode=mode,
        profile_name=profile_name,
    ).machine


if __name__ == "__main__":
    machine = create_machine(qubit="q2")
    config = machine.generate_config()

    from pprint import pprint
    pprint(config)
