# Quoridor RL — Target State Blueprint

Author: Ji Hua
Created Date: 2026-04-19
Last Modified: 2026-04-19
Current Version: 1

Document Type: Design
Document Subtype: Target State Blueprint
Document Status: Draft
Document Authority Scope: Global
Document Purpose:
This document defines the intended completed state of the Quoridor RL project. It describes what the project is, what capabilities it provides, and how its major modules are organized when fully realized. It is a target-state document only. It does not describe the current implementation state, migration steps, or phased rollout plans.

---

# 1. Project Definition

Quoridor RL is a complete research, training, evaluation, deployment, and gameplay platform for Quoridor agents.

When fully realized, the project provides a unified system in which Quoridor agents can be:

- defined
- instantiated
- trained
- evaluated
- compared
- promoted or rejected
- deployed
- executed in live gameplay
- inspected through transparent tooling and user interfaces

The project is not only a Quoridor game implementation.

It is not only an agent host.

It is not only an RL training environment.

It is a complete agent platform for Quoridor.

---

# 2. What the Completed Project Delivers

When complete, the project delivers all of the following:

## 2.1 A Canonical Agent Platform

The system provides one unified agent lifecycle model for all agent families, including:

- scripted agents
- heuristic agents
- search-based agents
- MCTS agents
- RL agents
- AlphaZero-style agents

All agents are managed through one coherent system instead of through separate ad hoc pipelines.

## 2.2 A Full Research and Training Loop

The system supports the full loop from agent creation to live usage:

- define agent identity and behavior
- train or derive concrete agent realizations
- evaluate against benchmarks and scenarios
- compare against other agents in Arena
- make promotion or rejection decisions
- deploy selected agents
- use them in live gameplay and visualization

## 2.3 A Transparent Evaluation Workbench

The project provides strong transparency for evaluation results.

A user or developer can understand:

- what an agent is
- how it differs from another agent
- how it performs across benchmarks
- why it wins or loses
- how stable it is across seeds and sides
- why it was promoted, rejected, or held

## 2.4 A Live Gameplay and Visualization Surface

The completed system provides an application-facing gameplay layer in which users can:

- play against agents
- watch agent vs agent games
- replay games
- inspect agent strength
- choose deployed agents for live use

## 2.5 A Foundation for Quoridor AlphaZero

The project supports advanced learning-based agents, especially:

- self-play generation
- MCTS-guided decision making
- neural policy/value integration
- checkpoint-based training
- candidate comparison
- gated promotion of stronger versions

AlphaZero is not an add-on outside the system.
It is a native agent family inside the completed platform.

---

# 3. Core Functional Capabilities

## 3.1 Agent Definition

The completed project supports a canonical definition model for agents.

It must be possible to define:

- algorithm family
- configuration
- policy
- identity
- metadata needed by downstream systems

All downstream systems consume this same definition model.

## 3.2 Agent Materialization

The system can materialize concrete agent realizations for different purposes, such as:

- training instances
- evaluation instances
- deployed instances
- live runtime instances

These realizations may differ by context, but they remain grounded in one canonical agent definition.

## 3.3 Training

The system supports training-oriented workflows for trainable agents.

This includes:

- self-play data generation
- training jobs
- checkpoint production
- model update loops
- creation of new trainable agent realizations
- promotion of newly trained candidates into evaluation

Not all agents require training, but the completed system fully supports those that do.

## 3.4 Evaluation

The system provides a structured evaluation stack including:

- static evaluation
- scenario evaluation
- arena evaluation
- gating and promotion decisions

Evaluation is reproducible and auditable.

## 3.5 Deployment

The system supports explicit deployment selection.

It can determine:

- which agents are deployable
- which versions are currently exposed
- which candidate is the active deployed representative
- which agents are available to the application-facing systems

## 3.6 Runtime Execution

The system supports clean runtime execution for deployed agents.

It can:

- create runtime instances
- bind them to game contexts
- receive state
- produce actions
- support agent vs agent and human vs agent gameplay

## 3.7 Replay and Inspection

Replay is a first-class capability.

The system can:

- replay historical games
- inspect decision sequences
- treat replay as a specialized agent-like execution path where useful
- connect replay to evaluation and debugging workflows

---

# 4. Major System Modules

The completed project is organized into several major modules.

## 4.1 Engine Module

The Engine Module is the rule authority of the platform.

Responsibilities:

- game rules
- legality validation
- path preservation
- deterministic state transitions
- canonical rule semantics

The Engine Module is shared by all higher-level systems.

## 4.2 Agent System

The Agent System is the lifecycle system for agents.

It is the core of the project.

Responsibilities:

- agent definition
- agent materialization
- training
- evaluation
- deployment
- runtime support

The Agent System is internally organized into submodules.

### 4.2.1 Definition Module

Owns:

- canonical agent definition model
- serialized agent definitions
- shared loaders/parsers
- agent identity and definitional metadata

### 4.2.2 Training Module

Owns:

- self-play
- training jobs
- model checkpoints
- RL-specific workflows
- AlphaZero-style learning loops

### 4.2.3 Evaluation Module

Owns:

- benchmark sets
- scenario tests
- arena evaluation
- evaluation runners
- result aggregation
- evaluation outputs
- gating inputs

Arena belongs here.

### 4.2.4 Deployment Module

Owns:

- promotion/rejection decisions
- deployed subset selection
- deployed version tracking
- externally exposed agent roster

### 4.2.5 Runtime Module

Owns:

- runtime materialization
- agent service
- instance lifecycle management
- serving-time execution support

## 4.3 Application System

The Application System is the user-facing layer.

Responsibilities:

- live gameplay
- configuration surfaces
- replay surfaces
- agent selection
- evaluation access
- user-facing interaction with the platform

The Application System does not define agent logic.
It consumes agent capability exposed by the Agent System.

## 4.4 Evaluation and Transparency UI

The completed project includes a user-facing workbench for agent and evaluation transparency.

Responsibilities:

- show agent definitions and metadata
- show evaluation results
- compare agents
- inspect matches
- show gating decisions
- surface benchmark and arena summaries

This is not only a cosmetic UI.
It is a core usability component of the platform.

---

# 5. Completed Agent Lifecycle

When complete, the project supports one unified lifecycle for agents.

## 5.1 Define

An agent is defined semantically.

The definition includes:

- family
- config
- policy
- identity
- metadata

## 5.2 Materialize

The system can materialize concrete realizations of the agent definition for different contexts.

## 5.3 Train

If the agent family is trainable, the system can run training workflows and produce improved realizations or semantically new agent definitions as appropriate.

## 5.4 Evaluate

The system can evaluate agents through multiple evaluation layers.

## 5.5 Decide

The system can decide whether an agent should be promoted, rejected, or kept under observation.

## 5.6 Deploy

The system can expose selected agents for live use.

## 5.7 Run

The system can execute deployed agents in live gameplay.

This lifecycle applies to both simple heuristic agents and advanced RL agents.

---

# 6. Evaluation Capabilities in the Completed System

A fully realized project includes a mature evaluation stack.

## 6.1 Benchmark Layer

The system maintains a stable benchmark set of reference agents and reference configurations.

These benchmarks are reusable and versioned.

## 6.2 Static Evaluation Layer

The system computes local or structural metrics such as:

- shortest-path related metrics
- decision latency
- wall-related efficiency
- other non-game outcome metrics

## 6.3 Scenario Evaluation Layer

The system evaluates agents on predefined states and expected behaviors.

Examples:

- best move tests
- tactical response tests
- constrained outcome scenarios

## 6.4 Arena Evaluation Layer

The system runs large-scale comparative evaluation through Arena.

Examples:

- candidate vs candidate
- benchmark vs candidate
- side-swapped matches
- multiple seeds
- batch tournaments

## 6.5 Gating Layer

The system converts evaluation results into lifecycle decisions.

Examples:

- promote
- reject
- continue evaluating
- deploy
- replace currently deployed version

---

# 7. Transparency and Usability in the Completed System

A fully realized project is not only technically capable.
It is operationally understandable.

## 7.1 Agent Transparency

The system should make it easy to answer:

- What is this agent?
- What family does it belong to?
- What are its parameters?
- What policy does it use?
- Is it deterministic?
- Is it deployed?

## 7.2 Evaluation Transparency

The system should make it easy to answer:

- How did this agent perform?
- Against whom?
- Under what seeds?
- Under what scenarios?
- Where is it weak?
- Where is it strong?

## 7.3 Decision Transparency

The system should make it easy to answer:

- Why was this candidate promoted?
- Why was it rejected?
- What metrics supported that decision?
- What threshold was applied?

## 7.4 Match Transparency

The system should make it easy to drill down into individual games and understand:

- participants
- seed
- winner
- move sequence
- turning points
- evaluation context

---

# 8. UI and Workbench Features in the Completed System

The completed system includes a practical UI/workbench rather than raw logs only.

## 8.1 Agent Browser

Users and developers can browse:

- agent definitions
- families
- parameters
- deployment status
- evaluation history

## 8.2 Candidate Comparison

Users and developers can compare:

- two or more agents
- benchmark deltas
- arena results
- scenario performance
- seed variance

## 8.3 Match Explorer

Users and developers can inspect:

- individual matches
- replays
- action logs
- summaries
- evaluation annotations

## 8.4 Gating Dashboard

Users and developers can inspect:

- promotion/rejection decisions
- thresholds
- supporting evidence
- current deployment roster

## 8.5 Live Gameplay Surface

Users can:

- play against agents
- watch agent vs agent play
- select deployed agents
- observe strength differences directly

---

# 9. AlphaZero and RL Capabilities in the Completed System

The completed project fully supports Quoridor RL and AlphaZero-style agents.

This includes:

- self-play pipelines
- replay buffer or equivalent training data flow
- policy/value model integration
- MCTS-guided search
- checkpoint production and loading
- candidate creation from trained outputs
- evaluation against fixed benchmarks
- deployment of selected trained agents
- comparison between heuristic, search-based, and learned agents inside one common platform

The completed project therefore supports both research and practical experimentation for Quoridor RL.

---

# 10. Final Repo and Module Clarity

When the project is complete, the repo should clearly communicate its architecture.

A new developer should be able to immediately understand:

- where the engine lives
- where the agent definition lives
- where Arena lives
- where training lives
- where deployment lives
- where runtime lives
- where the application-facing UI lives
- where the evaluation workbench lives

The repo should not contain multiple ambiguous active agent-definition systems.

It should clearly reflect the completed platform architecture.

---

# 11. What the Completed Project Ultimately Is

When fully realized, Quoridor RL is:

- a Quoridor rule engine
- a unified agent platform
- a training platform
- an evaluation platform
- a deployment system
- a live gameplay and replay system
- a transparency and inspection workbench
- a research foundation for AlphaZero-style agents

It is a complete end-to-end system for building, comparing, training, deploying, and understanding Quoridor agents.

---

# 12. Summary

If the Quoridor RL project is fully implemented, we get a complete agent platform for Quoridor.

We get:

- one coherent agent lifecycle
- one canonical agent definition model
- full support for evaluation and deployment
- live gameplay support
- transparent tooling and UI
- a practical foundation for MCTS + RL / AlphaZero

The final project is not only a codebase that can run Quoridor agents.

It is a complete system for creating, studying, improving, comparing, deploying, and using them.
