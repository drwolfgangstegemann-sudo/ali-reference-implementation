from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from .interfaces import MemoryInterface
from .models import (
    BehaviourProposal,
    Decision,
    Event,
    ExecutionResult,
    ExecutionStatus,
    NormEvaluation,
    Observation,
    ViabilityState,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    architecture_version TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    action TEXT NOT NULL,
    norm_decision TEXT NOT NULL,
    execution_status TEXT NOT NULL,
    execution_success INTEGER NOT NULL,
    rolled_back INTEGER NOT NULL DEFAULT 0,
    event_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_object_id ON events(object_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""


class SQLiteMemory(MemoryInterface):
    """Persistent immutable Event store."""

    @classmethod
    def from_config(cls, config, params):
        return cls(config.database)


    def __init__(self, database: Path) -> None:
        self.database = database.resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database) as conn:
            conn.executescript(SCHEMA)

    def store(self, event: Event) -> None:
        payload = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(self.database) as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, timestamp, architecture_version, observation_id,
                    object_id, proposal_id, action, norm_decision,
                    execution_status, execution_success, event_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.timestamp,
                    event.architecture_version,
                    event.observation.id,
                    event.observation.object_id,
                    event.proposal.id,
                    event.proposal.action,
                    event.norm_evaluation.decision.value,
                    event.execution.status.value,
                    int(event.execution.success),
                    payload,
                ),
            )

    def retrieve_for_observation(
        self,
        observation: Observation,
        *,
        limit: int = 20,
    ) -> list[Event]:
        with sqlite3.connect(self.database) as conn:
            rows = conn.execute(
                """
                SELECT event_json FROM events
                WHERE object_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (observation.object_id, limit),
            ).fetchall()
        return [self._decode(row[0]) for row in rows]

    def recent(self, *, limit: int = 20) -> list[Event]:
        with sqlite3.connect(self.database) as conn:
            rows = conn.execute(
                """
                SELECT event_json FROM events
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._decode(row[0]) for row in rows]

    def last_executed_for_rollback(self) -> Event | None:
        with sqlite3.connect(self.database) as conn:
            row = conn.execute(
                """
                SELECT event_json FROM events
                WHERE execution_success = 1
                  AND execution_status = ?
                  AND rolled_back = 0
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (ExecutionStatus.EXECUTED.value,),
            ).fetchone()
        return self._decode(row[0]) if row else None

    def mark_rolled_back(self, event_id: str) -> None:
        with sqlite3.connect(self.database) as conn:
            conn.execute(
                "UPDATE events SET rolled_back = 1 WHERE event_id = ?",
                (event_id,),
            )

    def event_count(self) -> int:
        with sqlite3.connect(self.database) as conn:
            row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return int(row[0])

    @staticmethod
    def _decode(payload: str) -> Event:
        raw = json.loads(payload)
        return Event(
            id=raw["id"],
            timestamp=raw["timestamp"],
            architecture_version=raw["architecture_version"],
            observation=Observation(**raw["observation"]),
            viability=ViabilityState(
                **{
                    **raw["viability"],
                    "risks": tuple(raw["viability"].get("risks", [])),
                }
            ),
            proposal=BehaviourProposal(**raw["proposal"]),
            norm_evaluation=NormEvaluation(
                **{
                    **raw["norm_evaluation"],
                    "decision": Decision(raw["norm_evaluation"]["decision"]),
                    "activated_norms": tuple(raw["norm_evaluation"]["activated_norms"]),
                    "violated_norms": tuple(raw["norm_evaluation"]["violated_norms"]),
                }
            ),
            execution=ExecutionResult(
                **{
                    **raw["execution"],
                    "status": ExecutionStatus(raw["execution"]["status"]),
                    "errors": tuple(raw["execution"]["errors"]),
                }
            ),
            metadata=raw.get("metadata", {}),
        )
