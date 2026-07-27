from __future__ import annotations

from pathlib import Path

from .interfaces import SuperEgoInterface
from .models import (
    BehaviourProposal,
    Decision,
    Event,
    NormEvaluation,
    ViabilityState,
    new_id,
    utc_now,
)


class SuperEgo(SuperEgoInterface):
    """Evaluates proposals independently from behavioural generation.

    The SuperEgo receives only the workspace root path, not the full
    EnvironmentInterface. Domain containment is computed here directly,
    so the SuperEgo remains decoupled from execution concerns.
    """

    # Permanent constraints: never relaxed regardless of viability or context.
    HARD_NORMS: dict[str, str] = {
        "H1_NO_DELETE":        "Deletion is prohibited.",
        "H2_LOCAL_DOMAIN":     "All paths must remain inside the operational domain.",
        "H3_NO_OVERWRITE":     "Existing files must never be overwritten.",
        "H4_REVERSIBLE":       "Autonomous file modifications must be reversible.",
        "H5_VISIBLE_FILES":    "Hidden files must not be modified.",
        "H6_MINIMUM_VIABILITY": "Operational viability must meet the execution threshold.",
    }


    @classmethod
    def from_config(cls, config, params):
        return cls(
            workspace=config.workspace,
            minimum_confidence=float(config.super_ego["minimum_confidence"]),
            minimum_viability_for_execution=float(
                config.runtime["minimum_viability_for_execution"]
            ),
            prefer_reversible_weight=float(config.super_ego["prefer_reversible_weight"]),
            prefer_low_uncertainty_weight=float(
                config.super_ego["prefer_low_uncertainty_weight"]
            ),
        )

    def __init__(
        self,
        *,
        workspace: Path,
        minimum_confidence: float,
        minimum_viability_for_execution: float,
        prefer_reversible_weight: float,
        prefer_low_uncertainty_weight: float,
    ) -> None:
        self.workspace = workspace.resolve()
        self.minimum_confidence = minimum_confidence
        self.minimum_viability_for_execution = minimum_viability_for_execution
        # Adaptive norms: weights may increase through successful operation.
        self.adaptive_weights: dict[str, float] = {
            "A1_PREFER_REVERSIBLE":       prefer_reversible_weight,
            "A2_PREFER_LOW_UNCERTAINTY":  prefer_low_uncertainty_weight,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        proposal: BehaviourProposal,
        viability: ViabilityState,
    ) -> NormEvaluation:
        activated: list[str] = []
        violated: list[str] = []

        # H1: deletion is always forbidden.
        if proposal.action == "delete":
            violated.append("H1_NO_DELETE")

        # H2: source must lie inside the workspace.
        if not self._is_inside_domain(proposal.source):
            violated.append("H2_LOCAL_DOMAIN")

        # H2: target is only checked when present; None is valid for
        # actions that do not require a destination (e.g. future classify-
        # in-place actions). Rejecting None here would incorrectly block
        # proposals whose action semantics do not involve a target path.
        if proposal.target is not None and not self._is_inside_domain(proposal.target):
            violated.append("H2_LOCAL_DOMAIN")

        # H3: must not overwrite an existing file.
        if proposal.target is not None and Path(proposal.target).exists():
            violated.append("H3_NO_OVERWRITE")

        # H4: action must be declared reversible.
        if not proposal.reversible:
            violated.append("H4_REVERSIBLE")
        else:
            activated.append("A1_PREFER_REVERSIBLE")

        # H5: hidden files must not be touched.
        try:
            if Path(proposal.source).name.startswith("."):
                violated.append("H5_VISIBLE_FILES")
        except Exception:
            violated.append("H2_LOCAL_DOMAIN")

        # H6: system viability must meet the execution threshold.
        # This is treated as a permanent constraint because the threshold
        # is fixed in configuration and cannot be overridden at runtime.
        if viability.score < self.minimum_viability_for_execution:
            violated.append("H6_MINIMUM_VIABILITY")

        # A2: low-confidence proposals are discouraged.
        if proposal.confidence < self.minimum_confidence:
            violated.append("A2_PREFER_LOW_UNCERTAINTY")
        else:
            activated.append("A2_PREFER_LOW_UNCERTAINTY")

        unique_violations = tuple(dict.fromkeys(violated))
        if unique_violations:
            decision = Decision.REJECTED
            explanation = "Proposal rejected by: " + ", ".join(unique_violations)
            confidence = 1.0
        else:
            decision = Decision.APPROVED
            explanation = "Proposal satisfies all hard and adaptive norms."
            confidence = min(1.0, proposal.confidence * viability.score)

        return NormEvaluation(
            id=new_id("norm"),
            timestamp=utc_now(),
            proposal_id=proposal.id,
            decision=decision,
            activated_norms=tuple(activated),
            violated_norms=unique_violations,
            explanation=explanation,
            confidence=round(confidence, 4),
        )

    def learn(self, event: Event) -> None:
        """Strengthen adaptive norm weights after successful approved cycles."""
        if event.norm_evaluation.decision == Decision.APPROVED and event.execution.success:
            for norm in event.norm_evaluation.activated_norms:
                if norm in self.adaptive_weights:
                    self.adaptive_weights[norm] = min(
                        3.0, round(self.adaptive_weights[norm] * 1.01, 4)
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_inside_domain(self, path: str) -> bool:
        """Return True when *path* resolves to a location inside workspace."""
        try:
            Path(path).resolve().relative_to(self.workspace)
            return True
        except ValueError:
            return False
