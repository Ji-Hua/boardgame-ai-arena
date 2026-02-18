# Quoridor Live Game Refactoring Roadmap

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-02-17
Current Version: 1

Document Type: Other
Document Subtype: Refactor Roadmap
Document Status: Draft
Document Authority Scope: Global
Document Purpose:
This document defines the high-level execution plan for refactoring the Quoridor Live Game subsystem. It outlines staged restructuring priorities and sequencing. This document does not define architecture, protocols, or implementation details.

---

# 1. Scope

This roadmap applies exclusively to the Live Game subsystem.

The Training subsystem is explicitly out of scope for this phase and will be addressed in a future planning cycle.

The purpose of this roadmap is to establish a controlled restructuring sequence before performing architecture-level and module-level refactoring.

---

# 2. Refactoring Objectives

The primary objective of this phase is structural stabilization, not feature expansion.

This phase aims to:

- Establish a unified mono-repository.
- Enforce a formal documentation governance structure.
- Reorganize documentation into authoritative categories.
- Align code structure with documented system boundaries.
- Prepare for architecture-driven module refactoring.

This phase does not aim to:

- Introduce new gameplay features.
- Preserve backward compatibility.
- Maintain continuous runtime operability during restructuring.

Structural correctness and long-term stability take precedence over short-term functionality.

---

# 3. Refactoring Phases

## Phase 0 — Preparation

Objective:
- Freeze restructuring principles.
- Confirm subsystem scope.
- Establish governance alignment.

Outputs:
- Architecture Draft document.
- Governance Standard document.
- This Roadmap Draft.

---

## Phase 1 — Mono Repository Establishment

Objective:
- Create a unified mono-repository as the future single source of truth.
- Define repository-level structural boundaries.
- Migrate existing documentation into the new repository.

Principles:
- No architectural refactoring at this stage.
- No module redesign.
- Focus on structural consolidation.

This phase prioritizes repository coherence over operational continuity.

---

## Phase 2 — Documentation Governance Implementation

Objective:
- Reclassify and rewrite all migrated documents according to the Governance Standard.
- Assign Document Type, Authority Scope, and Status to each document.
- Eliminate semantic overlap between documents.
- Remove outdated or redundant materials.

Principles:
- Documentation precedes code.
- Documents define structural intent.
- Content may remain incomplete, but structure must be correct.

This phase establishes documentation as the primary structural reference.

---

## Phase 3 — Code Migration Aligned with Documentation

Objective:
- Migrate Engine, Backend, Frontend, and Agent code into the mono-repository.
- Align code placement with the documented structural model.

Principles:
- Code follows documentation.
- No backward compatibility requirements.
- No guarantee of continuous system operability.
- Functional breakage is acceptable during structural realignment.

This phase does not perform deep architectural refactoring. It aligns code organization with documented structure.

---

## Phase 4 — Architecture-Driven Module Refactoring

Objective:
- Refactor Engine and Backend responsibilities according to the Architecture document.
- Clarify authority boundaries.
- Eliminate implicit coupling.
- Stabilize module interfaces.

Principles:
- Architecture document becomes the structural authority.
- No feature expansion during refactoring.
- Focus on boundary clarity and long-term evolvability.

---

## Phase 5 — Functional Stabilization

Objective:
- Resolve residual bugs.
- Restore system operability.
- Normalize interface consistency.
- Prepare for future Training subsystem planning.

Principles:
- Stability follows structural correctness.
- Performance optimization is secondary.
- Functional completeness is restored only after structural alignment.

---

# 4. Phase Dependencies

- Phase 1 and Phase 2 are tightly coupled and may partially overlap.
- Phase 3 requires Phase 2 structural documentation alignment.
- Phase 4 requires completion of code migration.
- Phase 5 follows architectural stabilization.

No phase introduces Training subsystem design.

---

# 5. Current Status

The project is currently in Phase 0.

Architecture documentation is in Draft status.

Documentation governance has been defined but not fully enforced.

The Live Game system is operational but structurally unstable and subject to refactoring.

---

# Changelog

Version 1 (2026-02-17)
- Initial refactoring roadmap draft for the Live Game subsystem.
