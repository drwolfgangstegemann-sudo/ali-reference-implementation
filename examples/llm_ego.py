"""
LLMEgo — a drop-in replacement for the built-in rule-based Ego.

This module demonstrates how to implement a custom Ego that uses a large
language model as its reasoning engine.  The Causal Core, Super-Ego, Memory,
and Runtime are completely unchanged.

In this reference version the LLM call is stubbed: the prompt that would be
sent to the model is constructed and returned as part of the proposal
explanation, and a fixed proposal is generated deterministically.  Replace
_call_llm() with a real API call to make this a live agent.

Configuration
-------------
Add the following to ali_config.json::

    "components": {
        "ego": {
            "class": "examples.llm_ego.LLMEgo",
            "params": {
                "model": "claude-sonnet-4-6",
                "stub": true
            }
        }
    }

Parameters
----------
model : str
    Model identifier passed to the LLM API.  Ignored in stub mode.
stub : bool
    When true, no network call is made.  A fixed proposal is returned and
    the formatted prompt is included in the proposal explanation.  Default: true.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

from ali.interfaces import EgoInterface
from ali.models import (
    BehaviourProposal,
    Decision,
    Event,
    Observation,
    ViabilityState,
    new_id,
    utc_now,
)

if TYPE_CHECKING:
    from ali.configuration import ALIConfiguration


class LLMEgo(EgoInterface):
    """Ego component that delegates behavioural generation to a language model.

    The LLM receives a structured JSON prompt describing the current
    observation, the viability state, and the most recent operational history.
    It is expected to return a JSON object describing one proposed action.

    In stub mode the prompt is constructed but no network call is made.
    The stub always proposes a safe dry-run rename so that tests work without
    API credentials.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        model: str = "claude-sonnet-4-6",
        stub: bool = True,
    ) -> None:
        self.workspace = workspace.resolve()
        self.model = model
        self.stub = stub
        self.successful_calls: int = 0

    # ── Plugin factory ────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: "ALIConfiguration",
        params: dict[str, Any],
    ) -> "LLMEgo":
        return cls(
            config.workspace,
            model=str(params.get("model", "claude-sonnet-4-6")),
            stub=bool(params.get("stub", True)),
        )

    # ── EgoInterface ──────────────────────────────────────────────────────────

    def generate(
        self,
        observation: Observation,
        viability: ViabilityState,
        history: Sequence[Event],
    ) -> list[BehaviourProposal]:
        # Skip files already successfully handled.
        if any(e.execution.success for e in history):
            return []

        prompt = self._build_prompt(observation, viability, history)

        if self.stub:
            return self._stub_response(observation, prompt)

        # ── Real LLM call (not active in reference implementation) ────────────
        # Uncomment and adapt to enable live operation:
        #
        # import anthropic
        # client = anthropic.Anthropic()
        # message = client.messages.create(
        #     model=self.model,
        #     max_tokens=256,
        #     system=(
        #         "You are the Ego component of an ALI agent. "
        #         "Respond only with a valid JSON object matching the schema "
        #         "in the user message. No prose, no markdown."
        #     ),
        #     messages=[{"role": "user", "content": prompt}],
        # )
        # raw = message.content[0].text
        # return self._parse_response(raw, observation)

        return []

    def learn(self, event: Event) -> None:
        if event.execution.success:
            self.successful_calls += 1

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(
        self,
        observation: Observation,
        viability: ViabilityState,
        history: Sequence[Event],
    ) -> str:
        """Construct the structured JSON prompt sent to the language model."""
        recent_history = [
            {
                "action":   e.proposal.action,
                "decision": e.norm_evaluation.decision.value,
                "success":  e.execution.success,
            }
            for e in list(history)[-3:]
        ]
        payload = {
            "task": (
                "Propose one safe, reversible file action for the observed file. "
                "Return a JSON object with keys: action, source, target, "
                "confidence (0.0-1.0), explanation, expected_result."
            ),
            "observation": {
                "file":         observation.object_id,
                "size_bytes":   observation.attributes.get("size_bytes"),
                "suffix":       observation.attributes.get("suffix"),
            },
            "viability": {
                "score":  viability.score,
                "status": viability.status,
                "risks":  list(viability.risks),
            },
            "recent_history": recent_history,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _stub_response(
        self,
        observation: Observation,
        prompt: str,
    ) -> list[BehaviourProposal]:
        """Return a fixed safe proposal without making an API call.

        The prompt is embedded in the explanation so it appears in the
        stored Event and can be inspected after the cycle.
        """
        relative = Path(observation.object_id)
        absolute = (self.workspace / relative).resolve()
        target   = absolute.with_name(f"llm_reviewed_{absolute.name}")

        return [
            BehaviourProposal(
                id=new_id("prop"),
                timestamp=utc_now(),
                observation_id=observation.id,
                objective="LLM-assisted workspace review (stub mode)",
                action="rename",
                source=str(absolute),
                target=str(target),
                confidence=0.70,
                explanation=(
                    f"[LLMEgo stub — model: {self.model}] "
                    f"Would send prompt to LLM:\n{prompt}"
                ),
                expected_result="File renamed with LLM-reviewed prefix.",
                reversible=True,
                metadata={
                    "rule":     "llm_review",
                    "model":    self.model,
                    "stub":     True,
                },
            )
        ]

    def _parse_response(
        self,
        raw: str,
        observation: Observation,
    ) -> list[BehaviourProposal]:
        """Parse a live LLM JSON response into a BehaviourProposal list."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        source = str(data.get("source", ""))
        target = data.get("target")

        if not source:
            return []

        return [
            BehaviourProposal(
                id=new_id("prop"),
                timestamp=utc_now(),
                observation_id=observation.id,
                objective="LLM-assisted workspace action",
                action=str(data.get("action", "rename")),
                source=source,
                target=str(target) if target else None,
                confidence=float(data.get("confidence", 0.5)),
                explanation=str(data.get("explanation", "LLM proposal")),
                expected_result=str(data.get("expected_result", "")),
                reversible=True,
                metadata={"rule": "llm_response", "model": self.model},
            )
        ]
