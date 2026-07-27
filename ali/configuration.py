from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ALIConfiguration:
    root: Path
    architecture_version: str
    workspace: Path
    database: Path
    runtime: dict[str, Any]
    causal_core: dict[str, Any]
    super_ego: dict[str, Any]
    # Optional plugin overrides.  Keys are component names ("ego",
    # "super_ego", "causal_core", "memory", "environment").
    # Each value is either a plain class-path string or a dict with
    # "class" and optional "params" keys.
    components: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path, config_path: Path | None = None) -> "ALIConfiguration":
        root = root.resolve()
        path = config_path or (root / "config" / "ali_config.json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"Configuration file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid configuration JSON: {exc}") from exc

        required = {
            "architecture_version", "workspace", "database",
            "runtime", "causal_core", "super_ego",
        }
        missing = required - set(data)
        if missing:
            raise RuntimeError(f"Missing configuration keys: {sorted(missing)}")

        workspace = (root / data["workspace"]).resolve()
        database  = (root / data["database"]).resolve()

        return cls(
            root=root,
            architecture_version=str(data["architecture_version"]),
            workspace=workspace,
            database=database,
            runtime=dict(data["runtime"]),
            causal_core=dict(data["causal_core"]),
            super_ego=dict(data["super_ego"]),
            components=dict(data.get("components", {})),
        )
