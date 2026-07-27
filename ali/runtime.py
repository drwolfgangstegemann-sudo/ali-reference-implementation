from __future__ import annotations

from dataclasses import dataclass

from .configuration import ALIConfiguration
from .interfaces import (
    CausalCoreInterface,
    EgoInterface,
    EnvironmentInterface,
    MemoryInterface,
    SuperEgoInterface,
)
from .models import (
    BehaviourProposal,
    Decision,
    Event,
    ExecutionResult,
    ExecutionStatus,
    NormEvaluation,
    ViabilityState,
    new_id,
    utc_now,
)
from .plugins import ComponentFactory


@dataclass(frozen=True)
class CycleOutcome:
    event: Event
    applied: bool


class Runtime:
    """Coordinates the deterministic ALI operational cycle.

    Component instantiation
    -----------------------
    All architectural components are created via ComponentFactory, which
    resolves the class paths declared in config["components"] and calls
    each class's from_config() classmethod. No component is instantiated
    directly inside the Runtime.

    Proposal iteration
    ------------------
    For each observation the Ego may return several proposals. The Runtime
    evaluates them in sequence, records every evaluation as an immutable
    Event (including rejections), and stops at the first approved proposal.
    """

    def __init__(self, config: ALIConfiguration) -> None:
        self.config = config
        factory = ComponentFactory(config)
        self.environment: EnvironmentInterface = factory.make_environment()
        self.memory:      MemoryInterface      = factory.make_memory()
        self.causal_core: CausalCoreInterface  = factory.make_causal_core()
        self.ego:         EgoInterface         = factory.make_ego()
        self.super_ego:   SuperEgoInterface    = factory.make_super_ego()

    def run(
        self, *, apply: bool = False, limit: int | None = None
    ) -> list[CycleOutcome]:
        observations = self.environment.observe()
        max_items    = limit or int(self.config.runtime["max_proposals_per_run"])
        outcomes: list[CycleOutcome] = []

        for observation in observations:
            history    = self.memory.retrieve_for_observation(observation)
            viability  = self.causal_core.evaluate(observations)
            proposals  = self.ego.generate(observation, viability, history)

            for proposal in proposals:
                evaluation = self.super_ego.evaluate(proposal, viability)
                execution  = self._execute(
                    proposal=proposal, evaluation=evaluation, apply=apply
                )
                event = Event(
                    id=new_id("event"),
                    timestamp=utc_now(),
                    architecture_version=self.config.architecture_version,
                    observation=observation,
                    viability=viability,
                    proposal=proposal,
                    norm_evaluation=evaluation,
                    execution=execution,
                    metadata={"mode": "apply" if apply else "dry_run"},
                )
                self.memory.store(event)
                self.ego.learn(event)
                self.super_ego.learn(event)
                self.causal_core.learn(event)

                outcomes.append(
                    CycleOutcome(
                        event=event,
                        applied=execution.status == ExecutionStatus.EXECUTED,
                    )
                )

                if len(outcomes) >= max_items:
                    return outcomes

                if evaluation.decision == Decision.APPROVED:
                    break

        return outcomes

    def status(self) -> tuple[ViabilityState, list[Event]]:
        observations = self.environment.observe()
        viability    = self.causal_core.evaluate(observations)
        return viability, self.memory.recent(limit=10)

    def rollback_last(self) -> str:
        event = self.memory.last_executed_for_rollback()
        if event is None:
            raise RuntimeError("No executed reversible Event is available.")
        info = event.execution.rollback_information
        if not info:
            raise RuntimeError("The Event contains no rollback information.")
        self.environment.rollback(info)
        self.memory.mark_rolled_back(event.id)
        return event.id

    def _execute(
        self,
        *,
        proposal: BehaviourProposal,
        evaluation: NormEvaluation,
        apply: bool,
    ) -> ExecutionResult:
        if evaluation.decision != Decision.APPROVED:
            return ExecutionResult(
                id=new_id("exec"), timestamp=utc_now(),
                proposal_id=proposal.id,
                status=ExecutionStatus.BLOCKED,
                success=False, duration_ms=0,
                outputs={}, errors=(evaluation.explanation,),
                rollback_information=None,
            )

        if not apply:
            return ExecutionResult(
                id=new_id("exec"), timestamp=utc_now(),
                proposal_id=proposal.id,
                status=ExecutionStatus.DRY_RUN,
                success=True, duration_ms=0,
                outputs={
                    "source": proposal.source,
                    "target": proposal.target,
                    "message": "Approved but not executed.",
                },
                errors=(), rollback_information=None,
            )

        return self.environment.execute(proposal)
