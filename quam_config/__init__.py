from .my_quam import Quam
from .create_machine import CreateMachine, create_machine


__all__ = ["CreateMachine", "Quam", "create_machine"]



if __name__ == "__main__":
    try:
        machine = create_machine()
        print("Machine created successfully from profile.")
    except Exception as exc:
        print(f"Error creating machine: {exc}")
