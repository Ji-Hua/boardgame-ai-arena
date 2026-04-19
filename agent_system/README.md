# Agent System

The Agent System is the lifecycle system for Quoridor agents. It is the system in which agents are defined, materialized, evaluated, deployed, and executed.

The Agent System is a top-level system parallel to the Application System, both of which share the Engine Module as their canonical rule foundation.

## Architecture

```
Application System
        ↓
Agent Service Interface
        ↓
Agent System
        ↓
Engine Module
```

The Agent System is organized into lifecycle submodules that reflect the stages of the agent lifecycle defined in `documents/system/design/agent-system-design.md`.

## Submodule Structure

```
agent_system/
  definition/          # Canonical agent definition (SSOT)
  evaluation/          # Controlled agent assessment (Arena)
  deployment/          # Deployment selection and exposure
  runtime/             # Runtime materialization and serving (Agent Service)
  training/            # Training-oriented workflows (future)
  tests/               # Cross-cutting tests
```

### definition/

**Owns:** Canonical Agent Spec model, serialized YAML definitions, shared parsing/loading.

- `agent_spec.py` — `AgentSpec` frozen dataclass: the single source of truth for agent definition
- `agent_defs/` — 11 canonical YAML agent definitions consumed by all downstream subsystems
- `tests/` — tests for the shared Agent Spec model

Agent Spec is the semantic definition of an agent. It answers: *What is this agent?*

All downstream lifecycle stages (evaluation, deployment, runtime) depend on Agent Spec as the canonical definition source. No downstream stage may create an independent definition authority.

### evaluation/

**Owns:** Arena, evaluation-side materialization, evaluation orchestration.

- `arena/` — the complete Arena evaluation subsystem
  - `agents/` — Scorer/Policy/Agent abstractions and YAML→Agent materialization
  - `runner.py` — game execution and match orchestration
  - `experiments/` — YAML experiment definitions
  - `tests/` — Arena tests

Arena selects agents by their canonical Agent Spec identities and materializes evaluation-context instances (Scorer + Policy → Agent → AgentInstance).

### deployment/

**Owns:** Deployable subset selection, registry/exposure logic.

Currently minimal. Will hold deployment-facing views over Agent Spec as the system evolves.

### runtime/

**Owns:** Agent Service, runtime materialization, instance lifecycle management.

- `service/` — the Agent Service (FastAPI microservice)
  - `server.py` — HTTP endpoints (health, control plane, gameplay plane)
  - `service.py` — unified lifecycle/decision manager
  - `registry.py` — agent type registry (AgentMaterializer → agent instances)
  - `instance_manager.py` — live AgentInstance binding (room/seat)
  - `agents/` — concrete agent implementations (Greedy, Minimax, Random, Replay, Dummy)
  - `specs/` — materializer abstractions (AgentMaterializer, YamlAgentMaterializer)
  - `yaml_loader.py` — YAML→BaseAgent construction pipeline via AgentDefinition wrapper
- `tests/` — Agent Service tests

The Agent Service materializes runtime instances from Agent Spec for serving through the Agent Service Interface. The Application System consumes agent capability through this HTTP boundary.

### training/

**Owns:** Training-oriented workflows.

Currently a placeholder. The architectural location exists so that training code has a clear home when it is developed.

## Key Design Principles

1. **Agent Spec is SSOT** — All agent identity comes from `definition/agent_spec.py`. Both Arena and Agent Service consume the same canonical definitions from `definition/agent_defs/`.

2. **Definition is separate from materialization** — Agent Spec defines *what* an agent is. Each subsystem materializes instances appropriate to its context (evaluation instances, runtime instances).

3. **Lifecycle stages are distinct** — Define → Materialize → Train/Evaluate/Deploy/Run. Each stage has its own submodule.

4. **Engine is shared** — Both the Agent System and Application System use the same Engine Module for rule semantics.

## Cross-Package Dependencies

- Arena imports `backend.adapters.engine_adapter.EngineAdapter` for game execution
- Agent Service agents import `quoridor_engine` (Rust FFI) for search/evaluation
- The Application System communicates with Agent Service over HTTP (port 8090)
