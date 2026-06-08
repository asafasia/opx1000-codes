import json
from pathlib import Path

from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from either FixedFrequencyQuam or FluxTunableQuam
@quam_dataclass
class Quam(FluxTunableQuam):
    @classmethod
    def load(cls, filepath_or_dict=None, *args, **kwargs):
        """Load only the QuAM state files by default.

        The upstream default recursively merges every JSON file below the
        working directory. This project also contains independent profile JSON,
        so the default load is constrained to state.json and wiring.json.
        Explicit paths and dictionaries keep the upstream behavior.
        """
        if filepath_or_dict is not None:
            return super().load(filepath_or_dict, *args, **kwargs)

        root = Path.cwd()
        if not (root / "state.json").exists():
            root = REPOSITORY_ROOT

        contents = {}
        for filename in ("state.json", "wiring.json"):
            path = root / filename
            if path.exists():
                with path.open(encoding="utf-8") as file:
                    contents.update(json.load(file))

        if not contents:
            raise FileNotFoundError(
                f"No state.json or wiring.json found in {root}"
            )
        return super().load(contents, *args, **kwargs)

