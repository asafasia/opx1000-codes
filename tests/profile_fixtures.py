"""Offline full-chip fixture, independent of live calibrated device profiles."""
from pathlib import Path
from profiles import Profile


def multi_qubit_profile():
    return Profile("multi_qubit", root=Path(__file__).parent / "fixtures" / "profiles")
