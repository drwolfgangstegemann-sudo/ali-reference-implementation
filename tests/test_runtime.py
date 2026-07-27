from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ali.configuration import ALIConfiguration
from ali.models import Decision, ExecutionStatus
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


class RuntimeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = Runtime(_make_config(self.root))

    def tearDown(self) -> None:
        self.temp.cleanup()

    # ------------------------------------------------------------------
    # Dry-run correctness
    # ------------------------------------------------------------------

    def test_dry_run_stores_complete_event_without_changing_file(self) -> None:
        file_path = self.runtime.config.workspace / "Mein Text.txt"
        file_path.write_text("x", encoding="utf-8")

        outcomes = self.runtime.run(apply=False)

        self.assertEqual(len(outcomes), 1)
        self.assertTrue(file_path.exists())
        self.assertEqual(outcomes[0].event.execution.status, ExecutionStatus.DRY_RUN)
        self.assertEqual(self.runtime.memory.event_count(), 1)
        stored = self.runtime.memory.recent(limit=1)[0]
        self.assertEqual(stored.observation.object_id, "Mein Text.txt")
        self.assertEqual(stored.proposal.action, "rename")

    # ------------------------------------------------------------------
    # Apply and rollback
    # ------------------------------------------------------------------

    def test_apply_and_rollback(self) -> None:
        source = self.runtime.config.workspace / "Mein Text.txt"
        source.write_text("x", encoding="utf-8")

        outcomes = self.runtime.run(apply=True)
        self.assertEqual(len(outcomes), 1)
        target = self.runtime.config.workspace / "Mein_Text.txt"
        self.assertFalse(source.exists())
        self.assertTrue(target.exists())

        event_id = self.runtime.rollback_last()
        self.assertEqual(event_id, outcomes[0].event.id)
        self.assertTrue(source.exists())
        self.assertFalse(target.exists())

    # ------------------------------------------------------------------
    # Event count
    # ------------------------------------------------------------------

    def test_every_proposal_creates_exactly_one_event(self) -> None:
        (self.runtime.config.workspace / "a.txt").write_text("a", encoding="utf-8")
        (self.runtime.config.workspace / "b.csv").write_text("b", encoding="utf-8")

        outcomes = self.runtime.run(apply=False)
        self.assertEqual(self.runtime.memory.event_count(), len(outcomes))

    # ------------------------------------------------------------------
    # Iterative proposal evaluation
    # ------------------------------------------------------------------

    def test_rejected_proposals_are_stored_as_events(self) -> None:
        """The Runtime must record rejected proposals as Events.

        This test creates a file whose name sanitises to an already-existing
        target, forcing the rename proposal to be rejected by H3_NO_OVERWRITE.
        The rejection Event must be stored in Memory even though no action
        was executed.
        """
        source = self.runtime.config.workspace / "Mein Text.txt"
        # Pre-create the would-be target so H3 fires.
        target = self.runtime.config.workspace / "Mein_Text.txt"
        source.write_text("source", encoding="utf-8")
        target.write_text("target", encoding="utf-8")

        outcomes = self.runtime.run(apply=False)

        rejected = [
            o for o in outcomes
            if o.event.norm_evaluation.decision == Decision.REJECTED
        ]
        self.assertGreater(len(rejected), 0)
        # Rejected Events must be in Memory.
        self.assertEqual(
            self.runtime.memory.event_count(),
            len(outcomes),
        )
        for o in rejected:
            self.assertEqual(o.event.execution.status, ExecutionStatus.BLOCKED)

    # ------------------------------------------------------------------
    # History usage
    # ------------------------------------------------------------------

    def test_ego_skips_file_already_successfully_processed(self) -> None:
        """After a file is successfully handled, the Ego must not re-propose
        an action for it when the same object_id appears in history.

        After the rename is applied, the original filename disappears from
        the workspace. A second run must produce no proposal for that name.
        """
        source = self.runtime.config.workspace / "Mein Text.txt"
        source.write_text("x", encoding="utf-8")

        # First run: rename is proposed and applied.
        outcomes_first = self.runtime.run(apply=True)
        self.assertEqual(len(outcomes_first), 1)
        self.assertTrue(outcomes_first[0].applied)
        self.assertFalse(source.exists())

        # Second run: the original filename is gone from the workspace,
        # so no observation for it can be generated and no proposal can
        # be made for it.
        outcomes_second = self.runtime.run(apply=False)
        observed_names = [o.event.observation.object_id for o in outcomes_second]
        self.assertNotIn("Mein Text.txt", observed_names)

    def test_ego_history_lowers_confidence_after_rejection(self) -> None:
        """Confidence for a rule must decrease when that rule was previously
        rejected for the same file."""
        from ali.ego import Ego
        from ali.models import Observation, ViabilityState, new_id, utc_now
        from ali.models import (
            BehaviourProposal, NormEvaluation, ExecutionResult,
            Event, Decision, ExecutionStatus,
        )

        workspace = self.runtime.config.workspace
        ego = Ego(workspace)

        obs = Observation.file(
            source="local_workspace",
            relative_path="bad file.txt",
            size_bytes=0,
            suffix=".txt",
            hidden=False,
        )
        viability = ViabilityState(
            id=new_id("via"), timestamp=utc_now(),
            score=1.0, status="NORMAL", metrics={},
        )

        # First proposal: no history → base confidence.
        proposals_first = ego.generate(obs, viability, history=[])
        self.assertEqual(len(proposals_first), 1)
        confidence_first = proposals_first[0].confidence

        # Simulate a rejection event for this file and rule.
        rejected_proposal = proposals_first[0]
        fake_norm = NormEvaluation(
            id=new_id("norm"), timestamp=utc_now(),
            proposal_id=rejected_proposal.id,
            decision=Decision.REJECTED,
            activated_norms=(),
            violated_norms=("H3_NO_OVERWRITE",),
            explanation="Target exists.",
            confidence=1.0,
        )
        fake_exec = ExecutionResult(
            id=new_id("exec"), timestamp=utc_now(),
            proposal_id=rejected_proposal.id,
            status=ExecutionStatus.BLOCKED,
            success=False, duration_ms=0, outputs={}, errors=(), rollback_information=None,
        )
        fake_event = Event(
            id=new_id("event"), timestamp=utc_now(),
            architecture_version="test",
            observation=obs,
            viability=viability,
            proposal=rejected_proposal,
            norm_evaluation=fake_norm,
            execution=fake_exec,
        )

        # Second proposal: history contains one rejection → lower confidence.
        proposals_second = ego.generate(obs, viability, history=[fake_event])
        self.assertEqual(len(proposals_second), 1)
        confidence_second = proposals_second[0].confidence

        self.assertLess(
            confidence_second,
            confidence_first,
            "Confidence must decrease after a prior rejection for the same rule and file.",
        )

    # ------------------------------------------------------------------
    # Learning: Super-Ego adaptive weights
    # ------------------------------------------------------------------

    def test_super_ego_adaptive_weights_increase_after_success(self) -> None:
        """Adaptive norm weights must grow after a successful approved cycle."""
        source = self.runtime.config.workspace / "Mein Text.txt"
        source.write_text("x", encoding="utf-8")

        weight_before = self.runtime.super_ego.adaptive_weights.get(
            "A1_PREFER_REVERSIBLE", 0.0
        )
        self.runtime.run(apply=False)
        weight_after = self.runtime.super_ego.adaptive_weights.get(
            "A1_PREFER_REVERSIBLE", 0.0
        )
        self.assertGreater(
            weight_after,
            weight_before,
            "A1_PREFER_REVERSIBLE weight must increase after an approved dry-run cycle.",
        )


if __name__ == "__main__":
    unittest.main()
