from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ali.configuration import ALIConfiguration
from ali.ego import Ego
from ali.models import BehaviourProposal, Decision, ExecutionStatus, ViabilityState, new_id, utc_now
from ali.runtime import Runtime


def _make_config(root: Path) -> ALIConfiguration:
    (root / "config").mkdir(exist_ok=True)
    cfg = {
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
            "minimum_confidence": 0.85,
            "prefer_reversible_weight": 1.0,
            "prefer_low_uncertainty_weight": 1.0,
        },
    }
    (root / "config" / "ali_config.json").write_text(
        json.dumps(cfg), encoding="utf-8"
    )
    return ALIConfiguration.load(root)


class ArchitectureTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = Runtime(_make_config(self.root))

    def tearDown(self) -> None:
        self.temp.cleanup()

    # ------------------------------------------------------------------
    # Hard norm: H2 — domain containment
    # ------------------------------------------------------------------

    def test_external_target_is_rejected(self) -> None:
        source = self.runtime.config.workspace / "a.txt"
        source.write_text("x", encoding="utf-8")
        proposal = BehaviourProposal(
            id=new_id("prop"), timestamp=utc_now(),
            observation_id="obs", objective="test", action="move",
            source=str(source),
            target=str(self.root.parent / "outside.txt"),
            confidence=1.0, explanation="test", expected_result="test",
            reversible=True, metadata={},
        )
        viability = self.runtime.causal_core.evaluate(
            self.runtime.environment.observe()
        )
        evaluation = self.runtime.super_ego.evaluate(proposal, viability)
        self.assertEqual(evaluation.decision, Decision.REJECTED)
        self.assertIn("H2_LOCAL_DOMAIN", evaluation.violated_norms)

    def test_target_none_does_not_trigger_h2(self) -> None:
        """target=None must not be treated as a domain violation.

        Some future action types may not require a target path. Rejecting
        None as H2 would incorrectly block such proposals.
        """
        source = self.runtime.config.workspace / "a.txt"
        source.write_text("x", encoding="utf-8")
        proposal = BehaviourProposal(
            id=new_id("prop"), timestamp=utc_now(),
            observation_id="obs", objective="test", action="classify",
            source=str(source),
            target=None,
            confidence=1.0, explanation="test", expected_result="test",
            reversible=True, metadata={},
        )
        viability = self.runtime.causal_core.evaluate(
            self.runtime.environment.observe()
        )
        evaluation = self.runtime.super_ego.evaluate(proposal, viability)
        self.assertNotIn("H2_LOCAL_DOMAIN", evaluation.violated_norms)

    # ------------------------------------------------------------------
    # Hard norm: H3 — no overwrite
    # ------------------------------------------------------------------

    def test_overwrite_is_rejected(self) -> None:
        source = self.runtime.config.workspace / "a.txt"
        target = self.runtime.config.workspace / "b.txt"
        source.write_text("a", encoding="utf-8")
        target.write_text("b", encoding="utf-8")
        proposal = BehaviourProposal(
            id=new_id("prop"), timestamp=utc_now(),
            observation_id="obs", objective="test", action="move",
            source=str(source), target=str(target),
            confidence=1.0, explanation="test", expected_result="test",
            reversible=True, metadata={},
        )
        viability = self.runtime.causal_core.evaluate(
            self.runtime.environment.observe()
        )
        evaluation = self.runtime.super_ego.evaluate(proposal, viability)
        self.assertIn("H3_NO_OVERWRITE", evaluation.violated_norms)

    # ------------------------------------------------------------------
    # Hard norm: H6 — minimum viability
    # ------------------------------------------------------------------

    def test_low_viability_rejects_via_h6(self) -> None:
        """When viability falls below the execution threshold, H6 must fire."""
        source = self.runtime.config.workspace / "a.txt"
        source.write_text("x", encoding="utf-8")
        proposal = BehaviourProposal(
            id=new_id("prop"), timestamp=utc_now(),
            observation_id="obs", objective="test", action="rename",
            source=str(source),
            target=str(self.runtime.config.workspace / "b.txt"),
            confidence=1.0, explanation="test", expected_result="test",
            reversible=True, metadata={},
        )
        # Construct a viability state that is below the threshold (0.60).
        low_viability = ViabilityState(
            id=new_id("via"), timestamp=utc_now(),
            score=0.30, status="SAFE_MODE",
            metrics={}, risks=("LOW_FREE_DISK",),
        )
        evaluation = self.runtime.super_ego.evaluate(proposal, low_viability)
        self.assertEqual(evaluation.decision, Decision.REJECTED)
        self.assertIn("H6_MINIMUM_VIABILITY", evaluation.violated_norms)

    # ------------------------------------------------------------------
    # Causal Core
    # ------------------------------------------------------------------

    def test_causal_core_returns_viability_state(self) -> None:
        state = self.runtime.causal_core.evaluate([])
        self.assertGreaterEqual(state.score, 0.0)
        self.assertLessEqual(state.score, 1.0)
        self.assertIn(state.status, {"NORMAL", "RESTRICTED", "SAFE_MODE"})

    # ------------------------------------------------------------------
    # Ego: sanitise_name edge cases
    # ------------------------------------------------------------------

    def test_sanitise_name_empty_stem_falls_back_to_unnamed(self) -> None:
        """A filename whose stem is entirely stripped must not produce a
        bare extension like '.txt'."""
        result = Ego._sanitise_name("!!??**.txt")
        self.assertNotEqual(result, ".txt")
        self.assertTrue(result.endswith(".txt"))
        self.assertGreater(len(Path(result).stem), 0)

    def test_sanitise_name_preserves_german_umlauts(self) -> None:
        result = Ego._sanitise_name("Über mich.txt")
        self.assertIn("Über", result)

    def test_sanitise_name_collapses_whitespace(self) -> None:
        result = Ego._sanitise_name("my  file  name.md")
        self.assertNotIn("  ", result)
        self.assertIn("_", result)

    # ------------------------------------------------------------------
    # SuperEgo: no coupling to EnvironmentInterface
    # ------------------------------------------------------------------

    def test_super_ego_has_no_environment_attribute(self) -> None:
        """The SuperEgo must not hold a reference to the EnvironmentInterface.
        Domain checks are performed using the workspace path only."""
        self.assertFalse(
            hasattr(self.runtime.super_ego, "environment"),
            "SuperEgo must not store an EnvironmentInterface reference.",
        )
        self.assertTrue(hasattr(self.runtime.super_ego, "workspace"))


if __name__ == "__main__":
    unittest.main()
