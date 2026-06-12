import os

from .my_quam import Quam


def create_machine(profile_name: str | None = None) -> Quam:
    """Create an in-memory machine from a validated profile.

    The profile defaults to the QUAM_PROFILE environment variable, or "main".
    Importing lazily avoids circular imports while the quam_config package loads.
    """
    from .create_machine_from_profile import create_machine_from_profile

    selected_profile = profile_name or os.environ.get("QUAM_PROFILE", "main")
    return create_machine_from_profile(selected_profile, save=False)


__all__ = ["Quam", "create_machine"]



if __name__ == "__main__":
    try:
        machine = create_machine()
        print("Machine created successfully from profile.")
    except Exception as exc:
        print(f"Error creating machine: {exc}")