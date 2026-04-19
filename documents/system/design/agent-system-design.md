# Quoridor Agent System Architecture

Author: Ji Hua
Created Date: 2026-04-19
Last Modified: 2026-04-19
Current Version: 1

Document Type: Design
Document Subtype: Agent System Architecture
Document Status: In Development
Document Authority Scope: Agent System
Document Purpose:
This document defines the system-level architecture of the Quoridor Agent System. It clarifies the role of the Agent System within the overall platform, defines the canonical semantic model of Agent Spec and Agent Instance, and establishes the lifecycle model by which agents are defined, materialized, evaluated, deployed, and run. This document does not define API schemas, protocol details, file formats, or implementation-specific class structures.

---

# 1. Overview

The Quoridor Agent System is the lifecycle system responsible for agent logic.

It is the system in which agents are:

- defined
- materialized into concrete instances
- evaluated
- deployed
- executed
- optionally trained

The Agent System is not a user-facing system. It is a producer system whose outputs are consumed by other systems, especially the Application System through the Agent Service Interface.

The Engine Module remains the shared rule foundation of the entire platform. The Agent System must use the same rule semantics as the Application System.

The purpose of the Agent System is not merely to store agent code or configuration files. Its purpose is to provide a unified semantic and lifecycle model such that all agent-related workflows are grounded in one canonical definition model.

---

# 2. Position in the Overall Architecture

Within the Quoridor platform:

- The Engine Module is the rule authority.
- The Application System is the user-facing system.
- The Agent System is the lifecycle system for agents.
- The Agent Service Interface is the boundary through which the Application System accesses deployed agent capability.

At a high level:

Application System
        ↓
Agent Service Interface
        ↓
Agent System
        ↓
Engine Module

The Application System must not define agent logic.

The Agent Service Interface must not define agent logic.

The Agent System is the system of record for agent logic.

---

# 3. Design Goals

The Agent System is designed with the following goals:

- Establish one canonical semantic definition model for agents
- Ensure that agent-related workflows share the same agent identity
- Separate semantic definition from concrete runtime realization
- Support evaluation, deployment, and execution without redefining agents in each context
- Allow training-oriented workflows without making training mandatory
- Preserve rule-semantic consistency through shared Engine usage
- Support incremental evolution without requiring immediate large-scale repository refactoring

The central architectural goal is:

A Quoridor agent must have one semantic definition, and all lifecycle stages must relate back to that same definition.

---

# 4. Core Conceptual Model

## 4.1 Agent Spec

Agent Spec is the canonical semantic definition of an agent.

Agent Spec answers the question:

What is this agent, semantically?

An Agent Spec defines the semantic identity of an agent, including the aspects that make that agent meaningfully distinct from another agent.

At the conceptual level, Agent Spec may include:

- algorithm family
- configuration or parameters
- policy
- semantic identity
- comparable and serializable definition content

Agent Spec is the single source of truth for agent definition.

Agent Spec is not a runtime object.

Agent Spec is not tied to a particular game, room, match, or execution context.

Agent Spec is analogous to a class-level or blueprint-level definition.

---

## 4.2 Agent Instance

Agent Instance is a concrete realization of an Agent Spec in a specific lifecycle context.

Agent Instance answers the question:

How is this agent concretely realized in this context?

An Agent Instance is derived or materialized from an Agent Spec.

Agent Instance may carry concrete contextual state such as:

- seed or RNG ownership
- runtime-local buffers
- replay cursor
- search-local state
- training-local state
- deployment-bound realization state
- execution-context bindings

Agent Instance is analogous to an object instantiated from a class.

Agent Instance does not own definition authority.

Agent Instance must not redefine the semantic identity of the Agent Spec from which it is derived.

---

## 4.3 Relationship Between Agent Spec and Agent Instance

The relationship is:

Agent Spec defines the semantic agent.
Agent Instance is a concrete realization of that agent.

The distinction is fundamental.

Definition belongs to Agent Spec.

Concrete lifecycle work from training onward operates on Agent Instances or families of instances derived from an Agent Spec.

This means:

- Define establishes Agent Spec
- Train operates on instances derived from Agent Spec
- Evaluate operates on instances derived from Agent Spec
- Deploy operates on deployable instance views derived from Agent Spec
- Run operates on runtime instances derived from Agent Spec

The Agent System must never treat these as unrelated agent definitions.

They are different lifecycle realizations of the same semantic source.

---

# 5. Agent Lifecycle Model

## 5.1 Lifecycle Overview

The Agent System lifecycle begins with semantic definition and then proceeds through instance-side lifecycle stages.

The lifecycle model is:

Define
  ↓
Agent Spec
  ↓
Materialize / derive Agent Instances for concrete lifecycle purposes
  ↓
Train (optional)
Evaluate
Deploy
Run

The critical architectural rule is:

Definition happens at the Agent Spec level.
Concrete lifecycle operations happen at the Agent Instance level.

---

## 5.2 Define

Define is the stage in which Agent Spec is established.

Responsibilities of Define:

- establish the canonical semantic definition of the agent
- define the semantic identity of the agent
- define the algorithm/config/policy meaning of the agent
- provide the source definition from which concrete instances may be derived

Outputs of Define:

- Agent Spec

Define is the only stage with definition authority.

No downstream lifecycle stage may create an unrelated agent definition while pretending to operate on the same agent.

If a downstream process produces something semantically different, that result must be treated as a different Agent Spec.

---

## 5.3 Materialization

Materialization is the transition from Agent Spec to concrete Agent Instance.

Materialization may occur for different purposes and in different contexts.

Materialization is not itself one single business workflow. It is the architectural boundary at which semantic definition becomes concrete realization.

Examples of materialization contexts include:

- training context
- evaluation context
- deployment context
- runtime execution context

Different contexts may materialize different kinds of instances, but they must all remain grounded in the same Agent Spec identity.

---

## 5.4 Train

Train is an optional lifecycle stage.

Not all agents require training.

Train operates on Agent Instances derived from Agent Spec.

Train may:

- refine concrete parameters
- update trained state
- produce trained artifacts
- produce trained realizations associated with the originating Agent Spec
- generate semantically new candidate outputs that must be treated as new Agent Specs if their semantics change

Train is one way to concretize or derive instance-side realizations, but it is not the only way.

A non-trained heuristic agent may move directly from Agent Spec to evaluation, deployment, or runtime materialization.

Therefore:

Train is optional.
Train is not the universal path from definition to execution.

---

## 5.5 Evaluate

Evaluate operates on Agent Instances derived from Agent Spec.

Evaluation does not define agents.

Evaluation does not own a separate agent-definition model.

Evaluation consumes the same semantic source and realizes concrete instances suitable for controlled comparison.

Evaluation responsibilities include:

- instantiate evaluation-context Agent Instances
- control reproducibility inputs such as seeds
- run evaluation workflows against shared rule semantics
- compare outcomes between agents based on their Agent Spec identities

Arena belongs to this lifecycle stage.

Arena is an evaluation consumer within the Agent System.

Arena must not create an independent definition authority for agents.

Arena selects agents by their canonical Agent Spec identities and materializes evaluation-context instances from them.

---

## 5.6 Deploy

Deploy selects which agent realizations are exposed for external use.

Deployment does not create a new semantic definition of the agent.

Deployment is a lifecycle view over an existing Agent Spec and its concrete realizations.

Deployment responsibilities include:

- choose which agents are externally exposed
- determine which realizations are eligible for serving
- preserve traceability from deployed realizations back to Agent Spec identity

Deployment is therefore a selection and exposure stage, not a definition stage.

A deployed agent is not a different kind of agent.

It is a deployment-state realization of an Agent Spec.

---

## 5.7 Run

Run is the lifecycle stage in which a deployed agent is executed in a concrete serving or gameplay context.

Run operates on runtime Agent Instances derived from deployed Agent Spec realizations.

Run responsibilities include:

- bind the agent to runtime context
- own runtime-local state
- consume runtime-provided inputs
- produce agent decisions

The Application System does not run semantic agent definitions directly.

It interacts with concrete runtime instances through the Agent Service Interface.

---

# 6. Internal Structure of the Agent System

The Agent System is conceptually composed of multiple internal subsystems.

## 6.1 Definition Subsystem

The Definition Subsystem is responsible for establishing and maintaining Agent Spec.

Responsibilities:

- canonical semantic definition of agents
- identity management for agent definitions
- definition-level comparison and reference
- source-of-truth ownership for agent logic definitions

This subsystem is the architectural home of agent-definition authority.

---

## 6.2 Evaluation Subsystem

The Evaluation Subsystem is responsible for controlled agent assessment.

Responsibilities:

- materialize evaluation-context Agent Instances
- run reproducible comparisons
- collect evaluation outputs
- support benchmark and arena workflows

Arena belongs to this subsystem.

---

## 6.3 Deployment Subsystem

The Deployment Subsystem is responsible for deciding which agent realizations are exposed outside the Agent System.

Responsibilities:

- select deployable subsets
- preserve mapping from deployed realization to canonical Agent Spec
- provide deployment-facing views of agent definitions

Deployment is a lifecycle filter and exposure layer, not a new definition source.

---

## 6.4 Runtime Realization Support

The Agent System must support the realization of Agent Instances for concrete execution.

Responsibilities:

- provide the logic needed to materialize runtime instances from Agent Spec
- preserve definition-to-instance traceability
- support instance-local concerns such as seed ownership and local state

This support exists so that external serving layers can execute agents without taking over definition authority.

---

## 6.5 Training Subsystem

The Training Subsystem is an optional internal subsystem of the Agent System.

Responsibilities:

- support training-oriented workflows when applicable
- operate on Agent Instances derived from Agent Spec
- produce trained realizations or semantically new outputs as appropriate

The Training Subsystem is not the whole Agent System.

It is one internal subsystem within it.

---

# 7. Relationship to Other Systems

## 7.1 Relationship to Engine Module

The Engine Module is the shared rule foundation.

The Agent System must use the same rule semantics as the Application System.

This means:

- local simulation must preserve Engine semantics
- evaluation must preserve Engine semantics
- training-related game evolution must preserve Engine semantics
- runtime execution must preserve Engine semantics

The Agent System must not redefine rules.

The Engine Module remains the exclusive rule authority.

---

## 7.2 Relationship to Application System

The Application System is the user-facing system.

It may present, configure, select, and interact with agents from a user perspective, but it does not define them.

The Application System must not own agent-definition authority.

It consumes agent capability only through the Agent Service Interface.

---

## 7.3 Relationship to Agent Service Interface

The Agent Service Interface is a boundary adapter between the Application System and the Agent System.

Its responsibilities are to:

- expose deployed agents to the Application System
- materialize or route runtime realizations as needed
- keep internal agent logic hidden from the Application System

It does not:

- define agents
- evaluate agents
- train agents
- own semantic agent identity

The Agent Service Interface consumes deployment outputs from the Agent System.

---

## 7.4 Relationship to Arena

Arena is part of the evaluation side of the Agent System.

Arena is not a separate agent-definition authority.

Arena must consume canonical agent definitions and materialize evaluation-context Agent Instances from them.

Arena may control seeds and evaluation orchestration, but it must not redefine the semantic identity of the agents it evaluates.

---

## 7.5 Relationship to Training

Training is an internal optional workflow within the Agent System.

Training is not a top-level system parallel to the Application System.

Training must remain grounded in the same Agent Spec model as evaluation, deployment, and runtime execution.

---

# 8. Authority Boundaries

The following authority boundaries apply.

| Concern | Authority |
|---------|-----------|
| Game rules | Engine Module |
| Agent semantic definition | Agent System Definition Subsystem |
| Evaluation workflow | Agent System Evaluation Subsystem |
| Deployment selection | Agent System Deployment Subsystem |
| Runtime serving boundary | Agent Service Interface |
| User-facing interaction | Application System |

The following constraints must hold:

- The Application System cannot define agent logic
- The Agent Service Interface cannot define agent logic
- Arena cannot define agent logic
- Training cannot bypass Agent Spec authority
- Runtime instances cannot redefine Agent Spec identity
- All agent-related lifecycle stages must remain traceable to Agent Spec

---

# 9. Architectural Invariants

The following invariants must hold across the Agent System.

## 9.1 Agent Spec is the SSOT

Agent Spec is the single source of truth for agent definition.

There must not be multiple independent definition authorities for the same agent.

---

## 9.2 Definition and Realization Must Be Distinguished

Agent Spec and Agent Instance are not the same thing.

Agent Spec is semantic definition.

Agent Instance is concrete realization.

This distinction must remain explicit across all lifecycle stages.

---

## 9.3 Train is Optional

Training is an optional lifecycle stage.

An agent does not need to be trained in order to be a valid member of the Agent System.

---

## 9.4 Train, Evaluate, Deploy, and Run Operate on Instances

From the point at which Agent Spec is materialized, downstream lifecycle operations act on Agent Instances or their derived realizations.

They do not become new definition authorities merely because they operate in different contexts.

---

## 9.5 Deployment Does Not Redefine Agents

Deployment selects what is exposed.

Deployment does not redefine semantic agent identity.

---

## 9.6 Runtime Does Not Redefine Agents

Serving and runtime execution materialize concrete agent instances.

They do not define what the agent is.

---

## 9.7 Evaluation Does Not Redefine Agents

Arena and other evaluation workflows consume canonical definitions.

They must not maintain an unrelated semantic-definition model.

---

## 9.8 Shared Rule Semantics Must Hold

All agent-related workflows must share the same rule semantics through the Engine Module.

Identical rule inputs must preserve identical rule meaning across evaluation, training, deployment-related validation, and runtime execution.

---

# 10. Current Direction and Incremental Evolution

The current repository contains legacy layering and partial duplication across different agent consumers.

This document does not require immediate large-scale refactoring.

The near-term architectural priority is not full structural purity.

The near-term priority is to ensure that all agent-related consumers are grounded in one Agent Spec source of truth.

This means incremental progress may legitimately use:

- compatibility layers
- consumer-specific runtime materialization paths
- staged deprecation of duplicated legacy structures

Such transitional measures are acceptable as long as definition authority remains convergent toward one canonical Agent Spec model.

The key architectural principle is:

Definition unification must come before full runtime unification.

---

# 11. Non-Scope

This document does not define:

- API schemas
- service protocols
- file formats
- YAML schema details
- concrete implementation classes
- registry implementation details
- deployment topology
- training algorithm details
- performance requirements
- repository migration steps

Those concerns belong to other documents.

---

# 12. Summary

The Quoridor Agent System is the lifecycle system for agents.

Its central architectural principle is not merely that agents should be stored in one place. Its central principle is that one canonical Agent Spec must ground the full lifecycle of an agent.

Define establishes Agent Spec.

From that point onward, the Agent System materializes Agent Instances for concrete lifecycle purposes such as training, evaluation, deployment, and runtime execution.

These are not separate definitions of the agent.

They are different realizations of the same semantic source.

This distinction between Agent Spec and Agent Instance is the foundation on which evaluation consistency, deployment consistency, runtime consistency, and future system evolution must be built.

---

# Changelog

Version 1 (2026-04-19)
- Initial Agent System architecture document.
- Defined Agent System as the lifecycle system for agents.
- Established Agent Spec as the canonical semantic definition and Agent Instance as the concrete realization.
- Defined the lifecycle distinction between definition and instance-side workflows.
- Clarified the roles of evaluation, deployment, runtime execution, and optional training within the Agent System.
- Defined authority boundaries and architectural invariants for agent-definition ownership.
