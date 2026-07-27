# ALI Reference Implementation v0.2

This repository is a complete, executable minimal implementation of the architecture described in *Artificial Local Intelligence (ALI): Architecture and Reference Implementation*.

It demonstrates the architectural separation of:

- Causal Core: operational viability
- Ego: behavioural proposal generation
- Super-Ego: normative evaluation
- Runtime: deterministic coordination and execution
- Memory: immutable Event storage
- Environment Interface: controlled interaction with a local workspace

The reference domain is a local document workspace. The system can propose safe filename normalisation and file classification, evaluate those proposals normatively, execute approved reversible actions, store complete Events in SQLite, and roll back previous actions.

## Safety properties

The implementation:

- acts only inside its own `workspace` directory;
- never deletes files;
- never overwrites existing files;
- never accesses the internet;
- never sends email;
- never starts external programs;
- defaults to dry-run mode;
- records every decision;
- supports rollback of every executed action.

## Requirements

- Windows, macOS, or Linux
- Python 3.11 or later
- No third-party packages

## Quick start

Open a terminal in the extracted folder.

Create sample files:

```bash
python main.py init
```

Run one complete ALI cycle without changing files:

```bash
python main.py run
```

Execute approved reversible actions:

```bash
python main.py run --apply
```

Show the Causal Core state and recent Events:

```bash
python main.py status
```

Show full Event records:

```bash
python main.py events
```

Undo the most recent executed action:

```bash
python main.py rollback
```

Run the automated test suite:

```bash
python -m unittest discover -s tests -v
```

## Windows shortcuts

The archive also contains:

- `01_initialize_demo.bat`
- `02_run_dry.bat`
- `03_run_apply.bat`
- `04_show_status.bat`
- `05_rollback.bat`
- `06_run_tests.bat`

Double-clicking these files performs the corresponding command.

## Architectural objects

Every operational cycle uses immutable data classes:

- `Observation`
- `ViabilityState`
- `BehaviourProposal`
- `NormEvaluation`
- `ExecutionResult`
- `Event`

Every completed proposal evaluation creates exactly one stored Event, including rejected and dry-run proposals.

## Project structure

```text
ALI_Reference_Implementation_v0_2/
    ali/
        models.py
        interfaces.py
        environment.py
        causal_core.py
        ego.py
        super_ego.py
        memory.py
        runtime.py
        configuration.py
    config/
        ali_config.json
    tests/
        test_architecture.py
        test_runtime.py
    workspace/
    main.py
```

## Scope

This is a minimal reference implementation, not a production autonomous agent. Its purpose is to expose architectural gaps and provide a concrete basis for engineering discussion, experimentation, and extension.
