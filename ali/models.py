from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class Decision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVISE = "REVISE"


class ExecutionStatus(str, Enum):
    EXECUTED = "EXECUTED"
    DRY_RUN = "DRY_RUN"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class Observation:
    id: str
    timestamp: str
    source: str
    object_id: str
    object_type: str
    attributes: dict[str, Any]
    confidence: float

    @classmethod
    def file(
        cls,
        *,
        source: str,
        relative_path: str,
        size_bytes: int,
        suffix: str,
        hidden: bool,
    ) -> "Observation":
        return cls(
            id=new_id("obs"),
            timestamp=utc_now(),
            source=source,
            object_id=relative_path,
            object_type="file",
            attributes={
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "suffix": suffix,
                "hidden": hidden,
            },
            confidence=1.0,
        )


@dataclass(frozen=True)
class ViabilityState:
    id: str
    timestamp: str
    score: float
    status: str
    metrics: dict[str, float | int | str | bool]
    risks: tuple[str, ...] = ()


@dataclass(frozen=True)
class BehaviourProposal:
    id: str
    timestamp: str
    observation_id: str
    objective: str
    action: str
    source: str
    target: str | None
    confidence: float
    explanation: str
    expected_result: str
    reversible: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormEvaluation:
    id: str
    timestamp: str
    proposal_id: str
    decision: Decision
    activated_norms: tuple[str, ...]
    violated_norms: tuple[str, ...]
    explanation: str
    confidence: float


@dataclass(frozen=True)
class ExecutionResult:
    id: str
    timestamp: str
    proposal_id: str
    status: ExecutionStatus
    success: bool
    duration_ms: int
    outputs: dict[str, Any]
    errors: tuple[str, ...]
    rollback_information: dict[str, Any] | None


@dataclass(frozen=True)
class Event:
    id: str
    timestamp: str
    architecture_version: str
    observation: Observation
    viability: ViabilityState
    proposal: BehaviourProposal
    norm_evaluation: NormEvaluation
    execution: ExecutionResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["norm_evaluation"]["decision"] = self.norm_evaluation.decision.value
        result["execution"]["status"] = self.execution.status.value
        return result
