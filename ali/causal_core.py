from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3
from typing import Sequence

from .interfaces import CausalCoreInterface
from .models import Event, Observation, ViabilityState, new_id, utc_now


class CausalCore(CausalCoreInterface):
    """Evaluates operational viability without generating behaviour."""

    @classmethod
    def from_config(cls, config, params):
        return cls(
            workspace=config.workspace,
            database=config.database,
            minimum_free_disk_mb=int(config.causal_core["minimum_free_disk_mb"]),
            maximum_unresolved_ratio=float(config.causal_core["maximum_unresolved_ratio"]),
        )


    def __init__(
        self,
        *,
        workspace: Path,
        database: Path,
        minimum_free_disk_mb: int,
        maximum_unresolved_ratio: float,
    ) -> None:
        self.workspace = workspace
        self.database = database
        self.minimum_free_disk_mb = minimum_free_disk_mb
        self.maximum_unresolved_ratio = maximum_unresolved_ratio
        self.completed_events = 0

    def evaluate(
        self,
        observations: Sequence[Observation],
        *,
        unresolved_count: int = 0,
    ) -> ViabilityState:
        risks: list[str] = []

        disk = shutil.disk_usage(self.workspace)
        free_disk_mb = disk.free / (1024 * 1024)
        disk_score = min(1.0, free_disk_mb / max(float(self.minimum_free_disk_mb), 1.0))
        if free_disk_mb < self.minimum_free_disk_mb:
            risks.append("LOW_FREE_DISK")

        db_integrity = self._database_integrity()
        if not db_integrity:
            risks.append("DATABASE_UNAVAILABLE")

        observation_count = len(observations)
        unresolved_ratio = (
            unresolved_count / observation_count if observation_count else 0.0
        )
        uncertainty_score = max(
            0.0,
            1.0 - unresolved_ratio / max(self.maximum_unresolved_ratio, 0.01),
        )
        if unresolved_ratio > self.maximum_unresolved_ratio:
            risks.append("HIGH_UNRESOLVED_RATIO")

        score = round(
            max(
                0.0,
                min(
                    1.0,
                    0.45 * disk_score
                    + 0.40 * float(db_integrity)
                    + 0.15 * uncertainty_score,
                ),
            ),
            4,
        )

        if score >= 0.80:
            status = "NORMAL"
        elif score >= 0.60:
            status = "RESTRICTED"
        else:
            status = "SAFE_MODE"

        return ViabilityState(
            id=new_id("via"),
            timestamp=utc_now(),
            score=score,
            status=status,
            metrics={
                "free_disk_mb": round(free_disk_mb, 2),
                "database_integrity": db_integrity,
                "observation_count": observation_count,
                "unresolved_count": unresolved_count,
                "unresolved_ratio": round(unresolved_ratio, 4),
                "completed_events_seen": self.completed_events,
            },
            risks=tuple(risks),
        )

    def learn(self, event: Event) -> None:
        self.completed_events += 1

    def _database_integrity(self) -> bool:
        try:
            self.database.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.database) as conn:
                result = conn.execute("PRAGMA quick_check").fetchone()
            return bool(result and result[0] == "ok")
        except sqlite3.Error:
            return False
