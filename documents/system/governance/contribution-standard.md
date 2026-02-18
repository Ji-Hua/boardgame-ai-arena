# Quoridor Engineering Contribution Rules

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-02-17
Current Version: 1

Document Type: Governance
Document Subtype: Code Governance
Document Status: Draft
Document Authority Scope: Global
Document Purpose:
This document defines the engineering contribution rules for the Quoridor project. It establishes mandatory development discipline, branch workflow, testing requirements, architectural protection rules, and commit conventions. This document governs how code is written, reviewed, and merged. It is intended for both human contributors and AI coding agents.

---

# 1. Purpose

The purpose of this document is to:

- Protect architectural boundaries.
- Prevent structural corruption.
- Define branch and pull request workflow.
- Define testing discipline.
- Define commit and formatting standards.
- Provide enforceable rules for both humans and AI agents.

This document does not define runtime behavior.It defines how code contributions must be performed.

---

# 2. Branching Model

## 2.1 Long-Lived Branches

The repository maintains two long-lived branches:

- main — stable branch
- development — active development branch

The main branch must remain stable and recoverable at all times.

---

## 2.2 Branch Creation Rules

All new work MUST:

- Be created from the development branch.
- Be implemented in a separate feature branch.

Direct commits to main are strictly prohibited.Direct commits to development are strictly prohibited.

---

# 3. Pull Request Rules

## 3.1 PR Requirement

All changes MUST go through a Pull Request.

No code may be merged without a PR.

There are no hotfix exceptions.

---

## 3.2 Merge Strategy

All merges MUST use squash merge.

The following are NOT allowed:

- Merge commits
- Rebase merge

Each Pull Request must result in exactly one squashed commit.

---

## 3.3 PR Scope

A Pull Request SHOULD:

- Focus on a single logical purpose.
- Avoid mixing unrelated changes.
- Avoid formatting-only changes combined with logic changes.

Large structural changes SHOULD be separated when possible.

---

# 4. Commit Message Rules

## 4.1 Required Format

Commit messages MUST follow the format:

<Type>: <one-sentence summary>

Examples:

Feature: add move validation logicFix: correct boundary condition in wall placement
---

## 4.2 Commit Types

<Type> MUST be one of:

- Feature
- Fix
- Refactor
- Test
- Documents
- Infrastructure
- Reformat
- Other

Type names are case-sensitive.

---

# 5. Architecture and Authority Discipline

## 5.1 Layer Protection

Implementation MUST NOT:

- Expand Protocol contracts.
- Modify cross-module interfaces.
- Introduce new system capabilities without corresponding Architecture updates.

Protocol MUST NOT:

- Expand module responsibilities defined in Architecture.

Architecture defines structural intent.Protocol defines boundary contracts.Implementation executes within those constraints.

---

## 5.2 Cross-Module Access

Modules MUST NOT:

- Access internal structures of other modules directly.
- Import internal implementation details across boundaries.

All cross-module interaction MUST occur through defined interfaces or protocol contracts.

---

# 6. Dependency Discipline

The following rules apply:

- Circular dependencies are strictly prohibited.
- Utility modules MUST NOT become unstructured shared logic containers.
- Cross-layer imports MUST be avoided unless explicitly defined by architecture.

Any violation of structural dependency rules MUST be corrected before merge.

---

# 7. Testing Discipline

## 7.1 General Testing Rules

All existing tests MUST pass before a Pull Request can be merged.

New features MUST include appropriate test coverage.

Refactors MUST NOT break existing tests.

---

## 7.2 Protocol and Structural Changes

Any change to:

- Protocol definitions
- Public APIs
- Architectural boundaries

MUST include corresponding tests validating the change.

---

## 7.3 Test Levels

Where applicable, tests SHOULD include:

- Unit tests for internal logic.
- Integration tests for cross-module interaction.

---

# 8. Code Style and Formatting

Code formatting MUST be consistent across the repository.

Unformatted code MUST NOT be merged.

Specific formatting tools and linters are to be defined separately.

Until defined, contributors are responsible for maintaining consistent formatting and readable structure.

---

# 9. Tooling and Environment

Tooling standards (formatters, linters, dependency managers, CI tools) are not yet finalized.

When defined, they MUST be enforced before merge.

Future updates to this section may introduce mandatory tooling requirements.

---

# 10. AI Contribution Rules

AI coding agents MUST:

- Respect Architecture and Protocol boundaries.
- Not modify Governance or Protocol documents unless explicitly instructed.
- Clearly indicate scope of changes.
- Avoid introducing cross-layer shortcuts.

AI-generated code is subject to the same rules as human-written code.

---

# 11. Enforcement Model

Enforcement currently relies on:

- Manual review.
- Local validation before PR.
- Structural discipline.

Future CI enforcement may be introduced.

---

# 12. Evolution Clause

This document is currently in Draft status.

It is expected to evolve as the development workflow stabilizes.

Rules marked as MUST represent structural protection and should not be relaxed without careful consideration.

---

# Changelog

Version 1 (2026-02-17)
- Initial draft of engineering contribution rules.
