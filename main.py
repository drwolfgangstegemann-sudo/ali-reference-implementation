from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ali.configuration import ALIConfiguration
from ali.runtime import Runtime


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ALI Reference Implementation v0.4")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create safe demonstration files")

    run = sub.add_parser("run", help="Run the ALI operational cycle")
    run.add_argument("--apply", action="store_true", help="Execute approved reversible actions")
    run.add_argument("--limit", type=int, default=None, help="Maximum number of proposals")

    sub.add_parser("status", help="Show current viability and recent Events")

    events = sub.add_parser("events", help="Show complete stored Events")
    events.add_argument("--limit", type=int, default=10)

    sub.add_parser("rollback", help="Undo the latest executed reversible action")
    return p


def initialise_demo(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    samples = {
        "Mein erster Text.txt": "Demonstration file for the ALI reference implementation.\n",
        "Versuch  1.md": "# Demonstration\n\nThis filename contains repeated whitespace.\n",
        "daten.csv": "id,value\n1,42\n",
    }
    created = 0
    for name, content in samples.items():
        path = workspace / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            created += 1
    print(f"Workspace: {workspace}")
    print(f"Created {created} demonstration file(s).")


def print_outcomes(outcomes) -> None:
    if not outcomes:
        print("No behavioural proposal was generated.")
        return

    for item in outcomes:
        event = item.event
        print()
        print(f"Event:       {event.id}")
        print(f"Observation: {event.observation.object_id}")
        print(f"Viability:   {event.viability.score:.3f} ({event.viability.status})")
        print(f"Proposal:    {event.proposal.action}")
        print(f"Source:      {event.proposal.source}")
        print(f"Target:      {event.proposal.target}")
        print(f"Norm result: {event.norm_evaluation.decision.value}")
        print(f"Execution:   {event.execution.status.value}")
        if event.execution.errors:
            print(f"Errors:      {'; '.join(event.execution.errors)}")


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        config = ALIConfiguration.load(root)
        args = parser().parse_args()

        if args.command == "init":
            initialise_demo(config.workspace)
            return 0

        runtime = Runtime(config)

        if args.command == "run":
            outcomes = runtime.run(apply=args.apply, limit=args.limit)
            print_outcomes(outcomes)
            return 0

        if args.command == "status":
            viability, events = runtime.status()
            print(json.dumps({
                "viability": {
                    "score": viability.score,
                    "status": viability.status,
                    "metrics": viability.metrics,
                    "risks": viability.risks,
                },
                "event_count_shown": len(events),
                "recent_events": [
                    {
                        "id": event.id,
                        "object": event.observation.object_id,
                        "action": event.proposal.action,
                        "decision": event.norm_evaluation.decision.value,
                        "execution": event.execution.status.value,
                    }
                    for event in events
                ],
            }, indent=2, ensure_ascii=False))
            return 0

        if args.command == "events":
            events = runtime.memory.recent(limit=args.limit)
            print(json.dumps(
                [event.to_dict() for event in events],
                indent=2,
                ensure_ascii=False,
            ))
            return 0

        if args.command == "rollback":
            event_id = runtime.rollback_last()
            print(f"Rolled back Event: {event_id}")
            return 0

        return 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
