from pathlib import Path

from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam
from quam_builder.architecture.superconducting.qubit import FixedFrequencyTransmon


TEMPORARY_MW_FEM_INPUT_LO_MODE = "always_on"


def apply_temporary_mw_fem_lo_mode_bugfix(config):
    """Inject MW-FEM input lo_mode until QuAM exposes it directly."""
    for controller in config.get("controllers", {}).values():
        for fem in controller.get("fems", {}).values():
            if fem.get("type") != "MW":
                continue

            analog_inputs = fem.get("analog_inputs", {})
            for input_config in analog_inputs.values():
                if input_config is not None and "downconverter_frequency" in input_config:
                    input_config["lo_mode"] = TEMPORARY_MW_FEM_INPUT_LO_MODE
    return config


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from either FixedFrequencyQuam or FluxTunableQuam
@quam_dataclass
class Quam(FixedFrequencyQuam):
    @classmethod
    def load(cls, filepath_or_dict=None, *args, **kwargs):
        """Build a fresh machine from the selected profile by default."""
        if filepath_or_dict is not None:
            return super().load(filepath_or_dict, *args, **kwargs)

        from quam_config import create_machine

        return create_machine()

    def generate_config(self, *args, **kwargs):
        config = super().generate_config(*args, **kwargs)

        # return config
        return apply_temporary_mw_fem_lo_mode_bugfix(config)

    def save(
        self,
        path=None,
        content_mapping=None,
        include_defaults=None,
        ignore=None,
    ):
        """Save only to explicit non-root paths, such as calibration snapshots."""
        if path is None or Path(path) == Path("."):
            return None
        return super().save(path, content_mapping, include_defaults, ignore)

