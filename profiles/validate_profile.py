"""Validate a profile from the command line."""

import argparse

from profiles import ProfileError, load_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", default="main", help="Profile folder name")
    parser.add_argument("--qubit", help="Qubit selection for a single-qubit profile")
    args = parser.parse_args()

    try:
        profile = load_profile(args.name, qubit=args.qubit)
    except ProfileError as exc:
        print(f"Profile {args.name!r} is invalid: {exc}")
        return 1

    qubit_count = len(profile["qubits"]["qubits"])
    pulse_count = sum(len(pulses) for pulses in profile["pulses"]["pulses"].values())
    print(f"Profile {args.name!r} is valid: {qubit_count} qubit(s), {pulse_count} pulse(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
