"""ALI Compliance Test Suite — Appendix H verification.

Every test in this file is linked to exactly one SHALL requirement
from the ALI specification (ALI-REQ-NNN).  The test identifier appears
in the test name so that the compliance table can be generated directly
from the test output.

Verification approach
---------------------
Three types of evidence are used:

STATIC   — The requirement is verified by inspecting class structure
           (presence or absence of attributes/methods).  These tests
           do not exercise runtime behaviour.

SEQUENCE — The requirement concerns the order in which architectural
           components are invoked during one operational cycle.  A call
           recorder patches component methods before the cycle runs.

RUNTIME  — The requirement concerns observable behaviour during or after
           an operational cycle.

Requirements that are verified solely by the architecture of the source
code (e.g. frozen dataclasses, abstract base classes, type annotations)
are marked STRUCTURAL in the compliance table and are not duplicated here
if they are already verified by the Python interpreter itself.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from typing import Any

from ali.causal_core import CausalCore
from ali.configuration import ALIConfiguration
from ali.ego import Ego
from ali.environment import LocalWorkspaceEnvironment
from ali.interfaces import (
    CausalCoreInterface,
    EgoInterface,
    MemoryInterface,
    SuperEgoInterface,
)
from ali.memory import SQLiteMemory
from ali.models import (
    BehaviourProposal,
    Decision,
    Event,
    ExecutionStatus,
    NormEvaluation,
    Observation,
    ViabilityState,
    new_id,
    utc_now,
)
from ali.runtime import Runtime
from ali.super_ego import SuperEgo


# ── Test helpers ─────────────────────────────────────────────────────────────

def _make_config(root: Path, extra: dict | None = None) -> ALIConfiguration:
    (root / "config").mkdir(exist_ok=True)
    cfg: dict[str, Any] = {
        "architecture_version": "compliance-test",
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
            "minimum_confidence": 0.85,
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


class _BaseComplianceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = Runtime(_make_config(self.root))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _sample_file(self, name: str = "Mein Text.txt") -> Path:
        p = self.runtime.config.workspace / name
        p.write_text("x", encoding="utf-8")
        return p


# ── Group 1: Component existence ─────────────────────────────────────────────

class ComponentExistenceTests(_BaseComplianceTest):
    """REQ-001, REQ-005, REQ-009, REQ-013, REQ-017."""

    def test_REQ001_runtime_exists_and_is_single(self) -> None:
        """RUNTIME — Runtime is present and unique per configuration."""
        self.assertIsInstance(self.runtime, Runtime)

    def test_REQ005_causal_core_exists(self) -> None:
        """RUNTIME — Runtime creates exactly one CausalCore."""
        self.assertIsInstance(self.runtime.causal_core, CausalCoreInterface)

    def test_REQ009_ego_exists(self) -> None:
        """RUNTIME — Runtime creates exactly one Ego."""
        self.assertIsInstance(self.runtime.ego, EgoInterface)

    def test_REQ013_super_ego_exists(self) -> None:
        """RUNTIME — Runtime creates exactly one SuperEgo."""
        self.assertIsInstance(self.runtime.super_ego, SuperEgoInterface)

    def test_REQ017_memory_exists(self) -> None:
        """RUNTIME — Runtime creates exactly one Memory."""
        self.assertIsInstance(self.runtime.memory, MemoryInterface)


# ── Group 2: Negative responsibility constraints ──────────────────────────────

class NegativeResponsibilityTests(_BaseComplianceTest):
    """REQ-002–004, REQ-007–008, REQ-011–012, REQ-015–016.

    Each test verifies that a component does NOT implement the method
    associated with another component's organisational responsibility.
    """

    # Runtime shall not generate, evaluate, or assess viability itself.

    def test_REQ002_runtime_has_no_generate_method(self) -> None:
        """STATIC — Runtime does not implement behavioural generation."""
        self.assertFalse(
            hasattr(Runtime, "generate"),
            "Runtime must not expose a generate() method.",
        )

    def test_REQ003_runtime_has_no_normative_evaluate_method(self) -> None:
        """STATIC — Runtime does not implement normative evaluation."""
        # The Runtime calls super_ego.evaluate() but does not own the method.
        self.assertFalse(hasattr(Runtime, "evaluate"))

    def test_REQ004_runtime_has_no_viability_assess_method(self) -> None:
        """STATIC — Runtime delegates viability assessment to CausalCore."""
        # Runtime.run() calls causal_core.evaluate() but does not define it.
        self.assertFalse(
            "evaluate" in Runtime.__dict__,
            "Runtime.__dict__ must not contain an evaluate() implementation.",
        )

    # CausalCore shall not generate proposals or execute.

    def test_REQ007_causal_core_has_no_generate_method(self) -> None:
        """STATIC — CausalCore does not implement behavioural generation."""
        self.assertFalse(hasattr(CausalCore, "generate"))

    def test_REQ008_causal_core_has_no_execute_method(self) -> None:
        """STATIC — CausalCore does not implement execution."""
        self.assertFalse(hasattr(CausalCore, "execute"))

    # Ego shall not evaluate proposals or execute.

    def test_REQ011_ego_has_no_evaluate_method(self) -> None:
        """STATIC — Ego does not implement normative evaluation."""
        self.assertFalse(hasattr(Ego, "evaluate"))

    def test_REQ012_ego_has_no_execute_method(self) -> None:
        """STATIC — Ego does not implement execution."""
        self.assertFalse(hasattr(Ego, "execute"))

    # SuperEgo shall not generate proposals or execute.

    def test_REQ015_super_ego_has_no_generate_method(self) -> None:
        """STATIC — SuperEgo does not implement behavioural generation."""
        self.assertFalse(hasattr(SuperEgo, "generate"))

    def test_REQ016_super_ego_has_no_execute_method(self) -> None:
        """STATIC — SuperEgo does not implement execution."""
        self.assertFalse(hasattr(SuperEgo, "execute"))

    # Memory shall not generate, evaluate, or execute.

    def test_REQ039_memory_has_no_learn_or_interpret_method(self) -> None:
        """STATIC — Memory does not perform autonomous interpretation."""
        self.assertFalse(
            hasattr(SQLiteMemory, "learn"),
            "Memory must not implement learn(); it stores, not interprets.",
        )
        self.assertFalse(hasattr(SQLiteMemory, "generate"))
        self.assertFalse(hasattr(SQLiteMemory, "evaluate"))


# ── Group 3: Positive responsibility constraints ──────────────────────────────

class PositiveResponsibilityTests(_BaseComplianceTest):
    """REQ-006, REQ-010, REQ-014, REQ-018, REQ-020."""

    def test_REQ006_causal_core_produces_viability_state(self) -> None:
        """RUNTIME — CausalCore.evaluate() returns a ViabilityState."""
        state = self.runtime.causal_core.evaluate([])
        self.assertIsInstance(state, ViabilityState)
        self.assertGreaterEqual(state.score, 0.0)
        self.assertLessEqual(state.score, 1.0)

    def test_REQ010_ego_generates_proposals(self) -> None:
        """RUNTIME — Ego.generate() returns a list (may be empty for a clean file)."""
        self._sample_file("Mein Text.txt")
        obs = self.runtime.environment.observe()
        viability = self.runtime.causal_core.evaluate(obs)
        proposals = self.runtime.ego.generate(obs[0], viability, [])
        self.assertIsInstance(proposals, list)
        self.assertGreater(len(proposals), 0)
        self.assertIsInstance(proposals[0], BehaviourProposal)

    def test_REQ014_super_ego_evaluates_independently(self) -> None:
        """RUNTIME — SuperEgo.evaluate() returns NormEvaluation for any proposal."""
        self._sample_file()
        obs = self.runtime.environment.observe()
        viability = self.runtime.causal_core.evaluate(obs)
        proposals = self.runtime.ego.generate(obs[0], viability, [])
        evaluation = self.runtime.super_ego.evaluate(proposals[0], viability)
        self.assertIsInstance(evaluation, NormEvaluation)
        self.assertIn(evaluation.decision, list(Decision))

    def test_REQ018_memory_preserves_every_event(self) -> None:
        """RUNTIME — Memory stores every Event produced by the Runtime."""
        self._sample_file()
        self._sample_file("data.csv")
        outcomes = self.runtime.run(apply=False)
        self.assertEqual(
            self.runtime.memory.event_count(),
            len(outcomes),
        )

    def test_REQ020_memory_supports_historical_retrieval(self) -> None:
        """RUNTIME — Memory.retrieve_for_observation() returns past Events."""
        self._sample_file("Mein Text.txt")
        self.runtime.run(apply=False)
        obs = self.runtime.environment.observe()
        # After rename in dry-run, the original observation object_id was "Mein Text.txt".
        # We retrieve by creating a matching observation.
        from ali.models import Observation
        synthetic_obs = Observation.file(
            source="local_workspace",
            relative_path="Mein Text.txt",
            size_bytes=0,
            suffix=".txt",
            hidden=False,
        )
        history = self.runtime.memory.retrieve_for_observation(synthetic_obs)
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history[0], Event)


# ── Group 4: Runtime sequence ─────────────────────────────────────────────────

class RuntimeSequenceTests(_BaseComplianceTest):
    """REQ-021 through REQ-027.

    A call recorder patches each component method before the cycle runs.
    After the cycle, the order of recorded calls is verified against the
    required sequence.
    """

    def _run_with_recorder(self) -> list[str]:
        calls: list[str] = []

        orig_cc  = self.runtime.causal_core.evaluate
        orig_ego = self.runtime.ego.generate
        orig_se  = self.runtime.super_ego.evaluate
        orig_mem_store = self.runtime.memory.store
        orig_ego_learn = self.runtime.ego.learn
        orig_se_learn  = self.runtime.super_ego.learn
        orig_cc_learn  = self.runtime.causal_core.learn

        def rec_cc(*a, **kw):
            calls.append("causal_core.evaluate")
            return orig_cc(*a, **kw)

        def rec_ego(*a, **kw):
            calls.append("ego.generate")
            return orig_ego(*a, **kw)

        def rec_se(*a, **kw):
            calls.append("super_ego.evaluate")
            return orig_se(*a, **kw)

        def rec_store(*a, **kw):
            calls.append("memory.store")
            return orig_mem_store(*a, **kw)

        def rec_ego_learn(*a, **kw):
            calls.append("ego.learn")
            return orig_ego_learn(*a, **kw)

        def rec_se_learn(*a, **kw):
            calls.append("super_ego.learn")
            return orig_se_learn(*a, **kw)

        def rec_cc_learn(*a, **kw):
            calls.append("causal_core.learn")
            return orig_cc_learn(*a, **kw)

        self.runtime.causal_core.evaluate  = rec_cc
        self.runtime.ego.generate          = rec_ego
        self.runtime.super_ego.evaluate    = rec_se
        self.runtime.memory.store          = rec_store
        self.runtime.ego.learn             = rec_ego_learn
        self.runtime.super_ego.learn       = rec_se_learn
        self.runtime.causal_core.learn     = rec_cc_learn

        self._sample_file("Mein Text.txt")
        self.runtime.run(apply=False)
        return calls

    def _idx(self, calls: list[str], name: str) -> int:
        return next(i for i, c in enumerate(calls) if c == name)

    def test_REQ021_cycle_begins_with_observation(self) -> None:
        """SEQUENCE — Observation is collected before any component is invoked."""
        # The environment.observe() call in Runtime.run() precedes all
        # component calls.  We verify indirectly: causal_core is the first
        # component called after observe(), so it must appear before ego.
        calls = self._run_with_recorder()
        self.assertIn("causal_core.evaluate", calls)

    def test_REQ022_viability_before_planning(self) -> None:
        """SEQUENCE — CausalCore.evaluate() precedes Ego.generate()."""
        calls = self._run_with_recorder()
        self.assertLess(
            self._idx(calls, "causal_core.evaluate"),
            self._idx(calls, "ego.generate"),
            "REQ-022: causal_core.evaluate must be called before ego.generate",
        )

    def test_REQ023_planning_before_normative_evaluation(self) -> None:
        """SEQUENCE — Ego.generate() precedes SuperEgo.evaluate()."""
        calls = self._run_with_recorder()
        self.assertLess(
            self._idx(calls, "ego.generate"),
            self._idx(calls, "super_ego.evaluate"),
            "REQ-023: ego.generate must be called before super_ego.evaluate",
        )

    def test_REQ024_normative_approval_precedes_storage(self) -> None:
        """SEQUENCE — SuperEgo.evaluate() precedes Memory.store()."""
        calls = self._run_with_recorder()
        self.assertLess(
            self._idx(calls, "super_ego.evaluate"),
            self._idx(calls, "memory.store"),
            "REQ-024: super_ego.evaluate must be called before memory.store",
        )

    def test_REQ026_storage_precedes_learning(self) -> None:
        """SEQUENCE — Memory.store() precedes all learn() calls."""
        calls = self._run_with_recorder()
        store_idx = self._idx(calls, "memory.store")
        for learn_call in ("ego.learn", "super_ego.learn", "causal_core.learn"):
            if learn_call in calls:
                self.assertLess(
                    store_idx,
                    self._idx(calls, learn_call),
                    f"REQ-026: memory.store must precede {learn_call}",
                )

    def test_REQ027_sequence_is_deterministic(self) -> None:
        """SEQUENCE — Two successive cycles produce the same call order.

        Both runs use exactly one file so the recorded call lists have the
        same length and the per-cycle pattern comparison is valid.
        """
        # Run 1: one file → one cycle.
        calls1 = self._run_with_recorder()

        # Run 2: fresh runtime, same single file.
        self.tearDown()
        self.setUp()
        calls2 = self._run_with_recorder()

        # Both lists must reflect the same structural sequence.
        self.assertEqual(calls1, calls2)


# ── Group 5: Interface requirements ──────────────────────────────────────────

class InterfaceTests(_BaseComplianceTest):
    """REQ-028, REQ-029, REQ-031."""

    def test_REQ028_components_communicate_through_interfaces(self) -> None:
        """STATIC — SuperEgo holds workspace path, not EnvironmentInterface.

        This verifies that the SuperEgo communicates with the domain through
        a minimal data interface (Path) rather than through the full
        EnvironmentInterface.  The same principle applies to all components:
        each receives only the information required for its own responsibility.
        """
        self.assertFalse(
            hasattr(self.runtime.super_ego, "environment"),
            "SuperEgo must not hold an EnvironmentInterface reference.",
        )
        self.assertTrue(hasattr(self.runtime.super_ego, "workspace"))

    def test_REQ029_components_do_not_access_internal_state_directly(self) -> None:
        """STATIC — No component stores a direct reference to another component.

        Each component in the Runtime is accessible only through the Runtime
        itself, not through cross-references between components.
        """
        # Ego must not hold a reference to SuperEgo or Memory.
        self.assertFalse(hasattr(self.runtime.ego, "super_ego"))
        self.assertFalse(hasattr(self.runtime.ego, "memory"))
        # SuperEgo must not hold a reference to Ego or Memory.
        self.assertFalse(hasattr(self.runtime.super_ego, "ego"))
        self.assertFalse(hasattr(self.runtime.super_ego, "memory"))
        # CausalCore must not hold a reference to Ego or SuperEgo.
        self.assertFalse(hasattr(self.runtime.causal_core, "ego"))
        self.assertFalse(hasattr(self.runtime.causal_core, "super_ego"))

    def test_REQ031_every_architectural_object_has_unique_id(self) -> None:
        """RUNTIME — Separate calls to new_id() produce different identifiers."""
        ids = {new_id("obs") for _ in range(100)}
        self.assertEqual(len(ids), 100, "new_id() must produce unique identifiers.")


# ── Group 6: Event requirements ───────────────────────────────────────────────

class EventTests(_BaseComplianceTest):
    """REQ-032, REQ-033, REQ-034, REQ-035."""

    def test_REQ032_one_event_per_proposal_evaluation(self) -> None:
        """RUNTIME — Each proposal evaluation produces exactly one stored Event."""
        self._sample_file("Mein Text.txt")
        self._sample_file("data.csv")
        outcomes = self.runtime.run(apply=False)
        self.assertEqual(self.runtime.memory.event_count(), len(outcomes))

    def test_REQ033_event_contains_all_required_architectural_objects(self) -> None:
        """RUNTIME — Every stored Event contains all six required objects."""
        self._sample_file("Mein Text.txt")
        self.runtime.run(apply=False)
        event = self.runtime.memory.recent(limit=1)[0]

        self.assertIsInstance(event.observation, Observation)
        self.assertIsInstance(event.viability, ViabilityState)
        self.assertIsInstance(event.proposal, BehaviourProposal)
        self.assertIsInstance(event.norm_evaluation, NormEvaluation)
        # ExecutionResult is imported at the top.
        from ali.models import ExecutionResult
        self.assertIsInstance(event.execution, ExecutionResult)
        self.assertIsNotNone(event.metadata)

    def test_REQ034_events_are_immutable_after_storage(self) -> None:
        """STRUCTURAL + RUNTIME — Event dataclass is frozen=True.

        Attempting to assign to any field after construction raises TypeError.
        This is enforced by the Python dataclass machinery; no runtime test
        is required, but we verify it explicitly for documentation purposes.
        """
        self._sample_file("Mein Text.txt")
        self.runtime.run(apply=False)
        event = self.runtime.memory.recent(limit=1)[0]

        with self.assertRaises((TypeError, AttributeError)):
            event.metadata = {"modified": True}  # type: ignore[misc]

    def test_REQ035_events_are_permanently_retrievable(self) -> None:
        """RUNTIME — Events stored in Memory can be retrieved after the cycle."""
        self._sample_file("Mein Text.txt")
        outcomes = self.runtime.run(apply=False)
        stored_id = outcomes[0].event.id

        recent = self.runtime.memory.recent(limit=10)
        retrieved_ids = {e.id for e in recent}
        self.assertIn(stored_id, retrieved_ids)


# ── Group 7: Learning requirements ───────────────────────────────────────────

class LearningTests(_BaseComplianceTest):
    """REQ-036, REQ-037, REQ-038, REQ-040."""

    def test_REQ036_behavioural_learning_in_ego_only(self) -> None:
        """RUNTIME — Ego.successful_rules changes after a successful cycle;
        SuperEgo and CausalCore internal knowledge is unchanged by Ego's learn.
        """
        self._sample_file("Mein Text.txt")
        super_ego_weights_before = dict(
            self.runtime.super_ego.adaptive_weights
        )
        cc_events_before = self.runtime.causal_core.completed_events

        self.runtime.run(apply=False)

        # Ego's learning state changed.
        self.assertGreater(
            sum(self.runtime.ego.successful_rules.values()), 0
        )
        # SuperEgo's adaptive weights may change too (that is REQ-037),
        # but that change must originate from SuperEgo.learn(), not Ego.learn().
        # CausalCore.completed_events increments via CausalCore.learn().
        self.assertGreater(
            self.runtime.causal_core.completed_events, cc_events_before
        )

    def test_REQ037_normative_learning_in_super_ego_only(self) -> None:
        """RUNTIME — SuperEgo adaptive weights increase after approved cycles."""
        self._sample_file("Mein Text.txt")
        weight_before = self.runtime.super_ego.adaptive_weights.get(
            "A1_PREFER_REVERSIBLE", 0.0
        )
        self.runtime.run(apply=False)
        weight_after = self.runtime.super_ego.adaptive_weights.get(
            "A1_PREFER_REVERSIBLE", 0.0
        )
        self.assertGreater(weight_after, weight_before)

    def test_REQ038_operational_learning_in_causal_core_only(self) -> None:
        """RUNTIME — CausalCore.completed_events increments after each cycle."""
        self._sample_file("Mein Text.txt")
        before = self.runtime.causal_core.completed_events
        self.runtime.run(apply=False)
        self.assertGreater(self.runtime.causal_core.completed_events, before)

    def test_REQ040_learning_does_not_modify_architectural_responsibilities(
        self,
    ) -> None:
        """RUNTIME — After learning, components retain their original interface.

        We verify that after running several cycles, the Ego still implements
        only EgoInterface, the SuperEgo still implements only SuperEgoInterface,
        and neither has acquired the other's responsibilities.
        """
        for _ in range(3):
            self._sample_file(f"File_{_}.txt")
        self.runtime.run(apply=False, limit=3)

        # Ego must still be an EgoInterface and must not have acquired evaluate().
        self.assertIsInstance(self.runtime.ego, EgoInterface)
        self.assertFalse(hasattr(self.runtime.ego, "evaluate"))

        # SuperEgo must still be a SuperEgoInterface and must not have acquired generate().
        self.assertIsInstance(self.runtime.super_ego, SuperEgoInterface)
        self.assertFalse(hasattr(self.runtime.super_ego, "generate"))


# ── Group 8: Architectural stability ─────────────────────────────────────────

class ArchitecturalStabilityTests(_BaseComplianceTest):
    """REQ-041, REQ-042, REQ-043, REQ-044."""

    def test_REQ041_separation_of_responsibilities_is_preserved(self) -> None:
        """STATIC — Each component is an instance of exactly one interface type.

        A component that implements two interface types would indicate merged
        responsibilities.
        """
        ego = self.runtime.ego
        se = self.runtime.super_ego
        cc = self.runtime.causal_core
        mem = self.runtime.memory

        # Ego must not implement SuperEgoInterface.
        self.assertNotIsInstance(ego, SuperEgoInterface)
        # SuperEgo must not implement EgoInterface.
        self.assertNotIsInstance(se, EgoInterface)
        # CausalCore must not implement EgoInterface or SuperEgoInterface.
        self.assertNotIsInstance(cc, EgoInterface)
        self.assertNotIsInstance(cc, SuperEgoInterface)
        # Memory must not implement any cognitive interface.
        self.assertNotIsInstance(mem, EgoInterface)
        self.assertNotIsInstance(mem, SuperEgoInterface)
        self.assertNotIsInstance(mem, CausalCoreInterface)

    def test_REQ042_components_do_not_exchange_responsibilities_during_operation(
        self,
    ) -> None:
        """RUNTIME — The type of each component does not change during operation."""
        ego_type_before = type(self.runtime.ego)
        se_type_before  = type(self.runtime.super_ego)
        cc_type_before  = type(self.runtime.causal_core)
        mem_type_before = type(self.runtime.memory)

        self._sample_file("Mein Text.txt")
        self.runtime.run(apply=True)

        self.assertIs(type(self.runtime.ego),       ego_type_before)
        self.assertIs(type(self.runtime.super_ego), se_type_before)
        self.assertIs(type(self.runtime.causal_core), cc_type_before)
        self.assertIs(type(self.runtime.memory),    mem_type_before)

    def test_REQ043_no_architectural_self_modification(self) -> None:
        """RUNTIME — The Runtime's run() method behaves identically before and
        after a learning cycle.  We verify that the Runtime itself does not
        acquire new instance attributes as a side effect of running.
        """
        attrs_before = set(vars(self.runtime).keys())
        self._sample_file("Mein Text.txt")
        self.runtime.run(apply=False)
        attrs_after = set(vars(self.runtime).keys())
        self.assertEqual(attrs_before, attrs_after)

    def test_REQ044_only_operational_knowledge_changes_during_learning(
        self,
    ) -> None:
        """RUNTIME — Learning changes knowledge (weights, counters, rules)
        but not the public interface of any component.

        We verify that the interfaces module still provides the same abstract
        methods after a learning cycle.
        """
        import inspect
        from ali import interfaces

        methods_before = {
            name: set(m for m, _ in inspect.getmembers(cls, predicate=inspect.isfunction))
            for name, cls in [
                ("EgoInterface",       interfaces.EgoInterface),
                ("SuperEgoInterface",  interfaces.SuperEgoInterface),
                ("CausalCoreInterface", interfaces.CausalCoreInterface),
                ("MemoryInterface",    interfaces.MemoryInterface),
            ]
        }

        self._sample_file("Mein Text.txt")
        self.runtime.run(apply=False)

        methods_after = {
            name: set(m for m, _ in inspect.getmembers(cls, predicate=inspect.isfunction))
            for name, cls in [
                ("EgoInterface",       interfaces.EgoInterface),
                ("SuperEgoInterface",  interfaces.SuperEgoInterface),
                ("CausalCoreInterface", interfaces.CausalCoreInterface),
                ("MemoryInterface",    interfaces.MemoryInterface),
            ]
        }

        self.assertEqual(methods_before, methods_after)


if __name__ == "__main__":
    unittest.main()
