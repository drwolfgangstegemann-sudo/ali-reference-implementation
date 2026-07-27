from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

from .interfaces import (
    CausalCoreInterface,
    EgoInterface,
    EnvironmentInterface,
    MemoryInterface,
    SuperEgoInterface,
)

if TYPE_CHECKING:
    from .configuration import ALIConfiguration

# ── Default component class paths ────────────────────────────────────────────

DEFAULT_COMPONENTS: dict[str, str] = {
    "ego":         "ali.ego.Ego",
    "super_ego":   "ali.super_ego.SuperEgo",
    "causal_core": "ali.causal_core.CausalCore",
    "memory":      "ali.memory.SQLiteMemory",
    "environment": "ali.environment.LocalWorkspaceEnvironment",
}


# ── Class loader ─────────────────────────────────────────────────────────────

def load_class(dotted_path: str, base_class: type) -> type:
    """Resolve *dotted_path* to a Python class and verify it extends *base_class*.

    Parameters
    ----------
    dotted_path:
        A fully qualified class path such as ``"ali.ego.Ego"`` or
        ``"examples.llm_ego.LLMEgo"``.
    base_class:
        The interface the resolved class must implement.

    Raises
    ------
    ImportError
        When the module cannot be imported or the attribute does not exist.
    TypeError
        When the resolved class is not a subclass of *base_class*.
    """
    if "." not in dotted_path:
        raise ImportError(
            f"Invalid component path {dotted_path!r}. "
            "Use a fully qualified dotted path, e.g. 'mypackage.mymodule.MyClass'."
        )

    module_path, class_name = dotted_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import module '{module_path}' "
            f"for component path '{dotted_path}': {exc}"
        ) from exc

    if not hasattr(module, class_name):
        raise ImportError(
            f"Module '{module_path}' has no attribute '{class_name}'."
        )

    cls = getattr(module, class_name)

    if not (isinstance(cls, type) and issubclass(cls, base_class)):
        raise TypeError(
            f"'{dotted_path}' must be a subclass of {base_class.__name__}, "
            f"got {cls!r}."
        )

    return cls


# ── Component factory ─────────────────────────────────────────────────────────

class ComponentFactory:
    """Instantiates ALI components from configuration.

    Component resolution order
    --------------------------
    1. ``config.components[name]["class"]``  — explicit override in JSON
    2. ``config.components[name]``           — short form (plain string)
    3. ``DEFAULT_COMPONENTS[name]``          — built-in default

    Each component receives ``(config, params)`` via its ``from_config()``
    classmethod, where *params* comes from
    ``config.components[name].get("params", {})``.

    Example ali_config.json entry
    -----------------------------
    ``"components": {``
    ``    "ego": {``
    ``        "class": "examples.llm_ego.LLMEgo",``
    ``        "params": { "model": "claude-sonnet-4-6", "stub": true }``
    ``    }``
    ``}``
    """

    def __init__(self, config: "ALIConfiguration") -> None:
        self.config = config

    def _resolve(self, name: str) -> tuple[str, dict[str, Any]]:
        raw = self.config.components.get(name)
        if raw is None:
            return DEFAULT_COMPONENTS[name], {}
        if isinstance(raw, str):
            return raw, {}
        class_path = raw.get("class", DEFAULT_COMPONENTS[name])
        params = dict(raw.get("params", {}))
        return class_path, params

    def make_environment(self) -> EnvironmentInterface:
        path, params = self._resolve("environment")
        cls = load_class(path, EnvironmentInterface)
        return cls.from_config(self.config, params)

    def make_causal_core(self) -> CausalCoreInterface:
        path, params = self._resolve("causal_core")
        cls = load_class(path, CausalCoreInterface)
        return cls.from_config(self.config, params)

    def make_ego(self) -> EgoInterface:
        path, params = self._resolve("ego")
        cls = load_class(path, EgoInterface)
        return cls.from_config(self.config, params)

    def make_super_ego(self) -> SuperEgoInterface:
        path, params = self._resolve("super_ego")
        cls = load_class(path, SuperEgoInterface)
        return cls.from_config(self.config, params)

    def make_memory(self) -> MemoryInterface:
        path, params = self._resolve("memory")
        cls = load_class(path, MemoryInterface)
        return cls.from_config(self.config, params)
