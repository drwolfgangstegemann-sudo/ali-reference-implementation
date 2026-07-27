# ALI Plugin System

Every architectural component of ALI can be replaced by a third-party
implementation without modifying any other file in the project.

## How it works

Each component interface (`EgoInterface`, `SuperEgoInterface`,
`CausalCoreInterface`, `MemoryInterface`, `EnvironmentInterface`) defines a
`from_config(config, params)` classmethod.  The `ComponentFactory` in
`ali/plugins.py` calls this method when the Runtime starts, passing the full
`ALIConfiguration` object and any extra parameters declared in the JSON config.

## Writing a plugin

1. Create a Python file anywhere on your `PYTHONPATH`.
2. Define a class that extends the relevant interface.
3. Implement `from_config(config, params)` to receive configuration.
4. Implement the abstract methods (`generate` / `evaluate` / etc.).
5. Point to your class in `ali_config.json`.

### Minimal Ego plugin

```python
# myproject/smart_ego.py

from pathlib import Path
from typing import Any, Sequence
from ali.interfaces import EgoInterface
from ali.models import BehaviourProposal, Event, Observation, ViabilityState

class SmartEgo(EgoInterface):

    def __init__(self, workspace: Path, *, threshold: float = 0.9) -> None:
        self.workspace = workspace
        self.threshold = threshold

    @classmethod
    def from_config(cls, config, params: dict[str, Any]) -> "SmartEgo":
        return cls(config.workspace, threshold=float(params.get("threshold", 0.9)))

    def generate(self, observation, viability, history) -> list[BehaviourProposal]:
        # Your reasoning logic here.
        return []

    def learn(self, event: Event) -> None:
        pass
```

### Configuration

```json
{
  "components": {
    "ego": {
      "class": "myproject.smart_ego.SmartEgo",
      "params": {
        "threshold": 0.85
      }
    }
  }
}
```

All other components (`super_ego`, `causal_core`, `memory`, `environment`)
remain at their defaults and do not need to be listed.

## Included example

`examples/llm_ego.py` provides a complete `LLMEgo` implementation that
delegates behavioural generation to a large language model.  It ships in
**stub mode** (`"stub": true`), which constructs the prompt and generates a
safe fixed proposal without making any API call.  To enable live operation,
set `"stub": false` and uncomment the API call in `_call_llm()`.

Example configuration (`config/ali_config_llm_example.json`):

```json
{
  "components": {
    "ego": {
      "class": "examples.llm_ego.LLMEgo",
      "params": {
        "model": "claude-sonnet-4-6",
        "stub": true
      }
    }
  }
}
```

## Compliance

A plugin component must satisfy the same ALI compliance requirements as the
built-in implementation (see Appendix H of the specification):

- `EgoInterface.generate()` must not perform normative evaluation.
- `SuperEgoInterface.evaluate()` must not generate proposals.
- `MemoryInterface.store()` must preserve the complete Event without modification.
- No component may access another component's internal state directly.
