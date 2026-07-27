from __future__ import annotations

from pathlib import Path
import shutil
import time

from .interfaces import EnvironmentInterface
from .models import (
    BehaviourProposal,
    ExecutionResult,
    ExecutionStatus,
    Observation,
    new_id,
    utc_now,
)


class LocalWorkspaceEnvironment(EnvironmentInterface):
    """Controlled environment that acts only inside one local workspace."""

    @classmethod
    def from_config(cls, config, params):
        return cls(config.workspace)


    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / ".ali").mkdir(exist_ok=True)

    def observe(self) -> list[Observation]:
        observations: list[Observation] = []
        for path in sorted(self.workspace.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.workspace)
            if ".ali" in relative.parts:
                continue
            observations.append(
                Observation.file(
                    source="local_workspace",
                    relative_path=relative.as_posix(),
                    size_bytes=path.stat().st_size,
                    suffix=path.suffix.lower(),
                    hidden=any(part.startswith(".") for part in relative.parts),
                )
            )
        return observations

    def is_inside_domain(self, path: str) -> bool:
        try:
            Path(path).resolve().relative_to(self.workspace)
            return True
        except ValueError:
            return False

    def execute(self, proposal: BehaviourProposal) -> ExecutionResult:
        started = time.perf_counter()
        errors: list[str] = []
        outputs: dict[str, str] = {}
        rollback_information: dict[str, str] | None = None

        try:
            if proposal.action not in {"rename", "move"}:
                raise ValueError(f"Unsupported action: {proposal.action}")
            if proposal.target is None:
                raise ValueError("Target path is required")
            if not self.is_inside_domain(proposal.source) or not self.is_inside_domain(proposal.target):
                raise PermissionError("Action path lies outside the operational domain")

            source = Path(proposal.source).resolve()
            target = Path(proposal.target).resolve()

            if not source.exists():
                raise FileNotFoundError(f"Source file does not exist: {source}")
            if target.exists():
                raise FileExistsError(f"Target already exists: {target}")

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            outputs = {"source": str(source), "target": str(target)}
            rollback_information = {"source": str(source), "target": str(target)}
            success = True
            status = ExecutionStatus.EXECUTED
        except Exception as exc:
            success = False
            status = ExecutionStatus.FAILED
            errors.append(f"{type(exc).__name__}: {exc}")

        duration_ms = int((time.perf_counter() - started) * 1000)
        return ExecutionResult(
            id=new_id("exec"),
            timestamp=utc_now(),
            proposal_id=proposal.id,
            status=status,
            success=success,
            duration_ms=duration_ms,
            outputs=outputs,
            errors=tuple(errors),
            rollback_information=rollback_information,
        )

    def rollback(self, rollback_information: dict) -> None:
        source = Path(str(rollback_information["source"])).resolve()
        target = Path(str(rollback_information["target"])).resolve()

        if not self.is_inside_domain(str(source)) or not self.is_inside_domain(str(target)):
            raise PermissionError("Rollback path lies outside the operational domain")
        if not target.exists():
            raise FileNotFoundError(f"Executed target no longer exists: {target}")
        if source.exists():
            raise FileExistsError(f"Original source already exists: {source}")

        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(source))
