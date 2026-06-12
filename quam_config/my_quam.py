from pathlib import Path

from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam
from quam_builder.architecture.superconducting.qubit import FixedFrequencyTransmon


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

