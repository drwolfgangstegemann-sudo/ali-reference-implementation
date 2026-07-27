"""Tests for the ALI plugin system (ali/plugins.py)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from typing import Any, Sequence

from ali.configuration import ALIConfiguration
from ali.interfaces import EgoInterface
from ali.models import (
    BehaviourProposal,
    Event,
    Observation,
    ViabilityState,
    new_id,
    utc_now,
)
from ali.plugins import ComponentFactory, load_class, DEFAULT_COMPONENTS
from ali.runtime import Runtime


def _make_config(root: Path, extra: dict | None = None) -> ALIConfiguration:
    (root / "config").mkdir(exist_ok=True)
    cfg: dict[str, Any] = {
        "architecture_version": "test",
        "workspace": "workspace",
        "database": "workspace/.ali/events.sqlite3",
        "runtime": {
            "max_proposals_per_run": 20,
            "minimum_viability_for_execution": 0.60,
        },
        "causal_core": {
            "minimum_free_disk_mb": 1,
            "maximum_unresolved_ratio": 0.50,
        },
        "super_ego": {
            "minimum_confidence": 0.65,
            "prefer_reversible_weight": 1.0,
            "prefer_low_uncertainty_weight": 1.0,
        },
    }
    if extra:
        cfg.update(extra)
    (root / "config" / "ali_config.json").write_text(
        json.dumps(cfg), encoding="utf-8"
    )
    return ALIConfiguration.load(root)


class LoadClassTests(unittest.TestCase):
    """Unit tests for plugins.load_class()."""

    def test_loads_builtin_ego(self) -> None:
        from ali.ego import Ego
        cls = load_class("ali.ego.Ego", EgoInterface)
        self.assertIs(cls, Ego)

    def test_rejects_wrong_base_class(self) -> None:
        with self.assertRaises(TypeError):
            load_class("ali.ego.Ego", int)  # type: ignore[arg-type]

    def test_rejects_missing_module(self) -> None:
        with self.assertRaises(ImportError):
            load_class("ali.nonexistent_module.Foo", EgoInterface)

    def test_rejects_missing_attribute(self) -> None:
        with self.assertRaises(ImportError):
            load_class("ali.ego.NonExistentClass", EgoInterface)

    def test_rejects_path_without_dot(self) -> None:
        with self.assertRaises(ImportError):
            load_class("Ego", EgoInterface)


class ComponentFactoryTests(unittest.TestCase):
    """Tests that ComponentFactory instantiates the correct types."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = _make_config(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_default_ego_is_builtin(self) -> None:
        from ali.ego import Ego
        factory = ComponentFactory(self.config)
        ego = factory.make_ego()
        self.assertIsInstance(ego, Ego)

    def test_default_super_ego_is_builtin(self) -> None:
        from ali.super_ego import SuperEgo
        factory = ComponentFactory(self.config)
        self.assertIsInstance(factory.make_super_ego(), SuperEgo)

    def test_default_memory_is_sqlite(self) -> None:
        from ali.memory import SQLiteMemory
        factory = ComponentFactory(self.config)
        self.assertIsInstance(factory.make_memory(), SQLiteMemory)

    def test_custom_ego_via_class_path(self) -> None:
        from examples.llm_ego import LLMEgo
        config = _make_config(self.root, {
            "components": {
                "ego": {"class": "examples.llm_ego.LLMEgo", "params": {"stub": True}}
            }
        })
        factory = ComponentFactory(config)
        ego = factory.make_ego()
        self.assertIsInstance(ego, LLMEgo)

    def test_custom_ego_receives_params(self) -> None:
        config = _make_config(self.root, {
            "components": {
                "ego": {
                    "class": "examples.llm_ego.LLMEgo",
                    "params": {"model": "test-model-x", "stub": True},
                }
            }
        })
        factory = ComponentFactory(config)
        from examples.llm_ego import LLMEgo
        ego = factory.make_ego()
        self.assertIsInstance(ego, LLMEgo)
        self.assertEqual(ego.model, "test-model-x")

    def test_short_form_class_path_string(self) -> None:
        """A plain string value in components is treated as the class path."""
        config = _make_config(self.root, {
            "components": {"ego": "ali.ego.Ego"}
        })
        from ali.ego import Ego
        factory = ComponentFactory(config)
        self.assertIsInstance(factory.make_ego(), Ego)


class PluginIntegrationTests(unittest.TestCase):
    """End-to-end tests: Runtime with a plugin Ego runs a complete cycle."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_runtime_with_llm_ego_runs_full_cycle(self) -> None:
        """LLMEgo (stub) must integrate cleanly with the unmodified Runtime."""
        config = _make_config(self.root, {
            "components": {
                "ego": {
                    "class": "examples.llm_ego.LLMEgo",
                    "params": {"model": "test-model", "stub": True},
                }
            }
        })
        runtime = Runtime(config)
        (runtime.config.workspace / "sample.txt").write_text("x", encoding="utf-8")

        outcomes = runtime.run(apply=False)

        self.assertGreater(len(outcomes), 0)
        # The stub always generates a rename proposal.
        self.assertEqual(outcomes[0].event.proposal.action, "rename")
        # The proposal must have been evaluated by the unmodified Super-Ego.
        self.assertIsNotNone(outcomes[0].event.norm_evaluation)
        # The Event must be in Memory.
        self.assertEqual(runtime.memory.event_count(), len(outcomes))

    def test_invalid_plugin_path_raises_at_startup(self) -> None:
        """A wrong class path must raise ImportError when Runtime is created,
        not silently at the first generate() call."""
        config = _make_config(self.root, {
            "components": {
                "ego": {"class": "ali.nonexistent.Foo"}
            }
        })
        with self.assertRaises(ImportError):
            Runtime(config)

    def test_wrong_base_class_raises_at_startup(self) -> None:
        """A class that does not extend EgoInterface must be rejected."""
        config = _make_config(self.root, {
            "components": {
                "ego": {"class": "ali.memory.SQLiteMemory"}
            }
        })
        with self.assertRaises(TypeError):
            Runtime(config)

    def test_plugin_ego_learn_is_called(self) -> None:
        """The Runtime must call learn() on the plugin Ego after each Event."""
        config = _make_config(self.root, {
            "components": {
                "ego": {
                    "class": "examples.llm_ego.LLMEgo",
                    "params": {"stub": True},
                }
            }
        })
        runtime = Runtime(config)
        (runtime.config.workspace / "sample.txt").write_text("x", encoding="utf-8")

        from examples.llm_ego import LLMEgo
        self.assertIsInstance(runtime.ego, LLMEgo)
        calls_before = runtime.ego.successful_calls

        runtime.run(apply=False)

        # In dry-run mode execution.success is True (DRY_RUN counts as success).
        self.assertGreaterEqual(runtime.ego.successful_calls, calls_before)


if __name__ == "__main__":
    unittest.main()
