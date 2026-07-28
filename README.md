# ALI Reference Implementation

**Artificial Local Intelligence — Reference Implementation v0.4**

This repository contains the open-source reference implementation of the ALI
architecture as described in:

>Stegemann, W. (2026). Artificial Local Intelligence: Architecture and Reference Implementation. 
Zenodo. https://doi.org/10.5281/zenodo.21631699

---

## What is ALI?

Artificial Local Intelligence (ALI) is a reference architecture for autonomous
agents that separates five organisational responsibilities into independent
components:

- **Causal Core** — evaluates operational viability before planning begins
- **Ego** — generates behavioural proposals without evaluating them
- **Super-Ego** — evaluates proposals without generating them
- **Memory** — preserves every decision as a complete, immutable Event
- **Runtime** — coordinates the cycle without participating in reasoning

The architecture is defined by a compliance specification of 44 verifiable
SHALL requirements. This implementation satisfies all 44.

---

## Requirements

- Python 3.11 or later
- No third-party packages

---

## Quick start

```bash
# Create demonstration files
python main.py init

# Run one cycle (dry run — no files changed)
python main.py run

# Execute approved actions
python main.py run --apply

# Show current viability state and recent Events
python main.py status

# Undo the last executed action
python main.py rollback

# Run the full test suite
python -m unittest discover -s tests -v
```

---

## Project structure

```
ali/                    Core architecture
    causal_core.py      Operational viability assessment
    ego.py              Behavioural proposal generation
    super_ego.py        Normative evaluation
    memory.py           Immutable Event storage (SQLite)
    runtime.py          Cycle coordination
    interfaces.py       Abstract base classes for all components
    plugins.py          ComponentFactory — plugin system
    models.py           Architectural data objects (frozen dataclasses)
    configuration.py    Configuration loading

examples/
    llm_ego.py          Plugin example: LLM-based Ego (stub mode)
    README.md           Plugin development guide

config/
    ali_config.json                Standard configuration
    ali_config_llm_example.json    Example: LLM Ego via plugin

tests/
    test_architecture.py    Hard norm and component boundary tests
    test_runtime.py         Operational cycle and learning tests
    test_plugins.py         Plugin system tests
    test_compliance.py      All 44 SHALL requirements verified

workspace/              Default operational domain (local files)
main.py                 Command-line interface
```

---

## Plugin system

Any architectural component can be replaced without touching the rest of
the code. Add a `components` entry to `ali_config.json`:

```json
{
  "components": {
    "ego": {
      "class": "examples.llm_ego.LLMEgo",
      "params": {
        "model": "claude-sonnet-4-6",
        "stub": true
      }
    }
  }
}
```

See `examples/README.md` for a complete guide to writing plugins.

---

## Test suite

```
72 tests — 0 failures

test_compliance.py    41 tests — all 44 SHALL requirements verified
test_plugins.py       15 tests — plugin loading and integration
test_architecture.py   9 tests — component boundaries and hard norms
test_runtime.py        7 tests — operational cycle and learning
```

---

## Reference

Full architectural specification (book):

Stegemann, W. (2026). Artificial Local Intelligence: Architecture and Reference Implementation. 
Zenodo. https://doi.org/10.5281/zenodo.21631699

Scientific paper:

Stegemann, W. (2026). Artificial Local Intelligence: Architecture and Reference Implementation. 
Zenodo. https://doi.org/10.5281/zenodo.21632233

---

## Author

Wolfgang Stegemann  
Independent Researcher  
ORCID: 0009-0000-1346-1170 https://orcid.org/0009-0000-1346-1170

---

## License

MIT License. See LICENSE file.
