import json
from pathlib import Path

state_path = Path("state.json")

state = json.loads(state_path.read_text())

print(state.keys())
state.pop("controllers", None)

state_path.write_text(json.dumps(state, indent=2))
