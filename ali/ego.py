from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

from .interfaces import EgoInterface
from .models import (
    BehaviourProposal,
    Decision,
    Event,
    Observation,
    ViabilityState,
    new_id,
    utc_now,
)


CATEGORY_MAP: dict[str, str] = {
    ".doc":  "documents",
    ".docx": "documents",
    ".pdf":  "documents",
    ".txt":  "documents",
    ".md":   "documents",
    ".jpg":  "images",
    ".jpeg": "images",
    ".png":  "images",
    ".gif":  "images",
    ".csv":  "data",
    ".json": "data",
    ".xlsx": "data",
    ".xls":  "data",
}


class Ego(EgoInterface):
    """Generates behavioural proposals without performing normative evaluation.

    Learning state
    --------------
    successful_rules  -- counts successful executions per planning rule across
                         all files. Used to gradually increase proposal
                         confidence as the system accumulates positive evidence.

    History usage
    -------------
    generate() receives file-specific history (Events for the same object_id).
    The Ego uses this history to:
      - skip re-proposing for files that were already successfully handled,
      - lower confidence for proposals whose rule was previously rejected for
        this specific file.
    """

    @classmethod
    def from_config(cls, config, params):
        return cls(config.workspace)


    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        # Global learning state: rule -> successful execution count.
        self.successful_rules: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        observation: Observation,
        viability: ViabilityState,
        history: Sequence[Event],
    ) -> list[BehaviourProposal]:
        relative = Path(str(observation.attributes["relative_path"]))
        absolute = (self.workspace / relative).resolve()

        # History check 1: if this file was already successfully handled
        # in a previous cycle, do not propose any further action.
        if any(e.execution.success for e in history):
            return []

        # History check 2: count rule-specific rejections for this file
        # so confidence can be adjusted downward accordingly.
        prior_rejections: dict[str, int] = {}
        for e in history:
            if e.norm_evaluation.decision == Decision.REJECTED:
                rule = str(e.proposal.metadata.get("rule", ""))
                if rule:
                    prior_rejections[rule] = prior_rejections.get(rule, 0) + 1

        proposals: list[BehaviourProposal] = []

        # Rule 1: sanitise filename (highest priority — address before classifying).
        sanitised = self._sanitise_name(absolute.name)
        if sanitised != absolute.name:
            confidence = self._confidence(
                base=0.98,
                rule="sanitise_filename",
                prior_rejections=prior_rejections,
            )
            proposals.append(
                BehaviourProposal(
                    id=new_id("prop"),
                    timestamp=utc_now(),
                    observation_id=observation.id,
                    objective="Maintain a technically consistent workspace",
                    action="rename",
                    source=str(absolute),
                    target=str(absolute.with_name(sanitised)),
                    confidence=confidence,
                    explanation="Normalise whitespace and unsupported filename characters.",
                    expected_result="The same file remains available under a normalised name.",
                    reversible=True,
                    metadata={"rule": "sanitise_filename"},
                )
            )
            # Return immediately: classification should follow renaming,
            # not run in parallel with it.
            return proposals

        # Rule 2: classify file by extension into a subdirectory.
        category = CATEGORY_MAP.get(str(observation.attributes["suffix"]))
        if category and relative.parent == Path("."):
            confidence = self._confidence(
                base=0.90,
                rule="classify_by_extension",
                prior_rejections=prior_rejections,
            )
            proposals.append(
                BehaviourProposal(
                    id=new_id("prop"),
                    timestamp=utc_now(),
                    observation_id=observation.id,
                    objective="Maintain a classified document workspace",
                    action="move",
                    source=str(absolute),
                    target=str((self.workspace / category / absolute.name).resolve()),
                    confidence=confidence,
                    explanation=f"Classify the file by extension into '{category}'.",
                    expected_result=f"The file is stored in the '{category}' directory.",
                    reversible=True,
                    metadata={"rule": "classify_by_extension", "category": category},
                )
            )

        return proposals

    def learn(self, event: Event) -> None:
        """Record successful executions to inform future confidence estimates."""
        if event.execution.success:
            rule = str(event.proposal.metadata.get("rule", "unknown"))
            self.successful_rules[rule] = self.successful_rules.get(rule, 0) + 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _confidence(
        self,
        *,
        base: float,
        rule: str,
        prior_rejections: dict[str, int],
    ) -> float:
        """Compute proposal confidence from base value, global learning, and
        file-specific rejection history.

        Each successful execution of this rule globally adds a small boost.
        Each prior rejection of this rule for the current file subtracts a
        penalty, reflecting that the proposal has already failed normative
        evaluation in this context.
        """
        global_boost = self.successful_rules.get(rule, 0) * 0.001
        local_penalty = prior_rejections.get(rule, 0) * 0.05
        return max(0.0, min(0.999, base + global_boost - local_penalty))

    @staticmethod
    def _sanitise_name(name: str) -> str:
        """Return a technically safe filename.

        Collapses whitespace runs to underscores, strips characters outside
        the allowed set, and falls back to 'unnamed' if the stem becomes
        empty after sanitisation.
        """
        path = Path(name)
        stem = re.sub(r"\s+", "_", path.stem.strip())
        stem = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß._-]", "", stem)
        if not stem:
            stem = "unnamed"
        return f"{stem}{path.suffix}"
