# Quoridor Documentation Governance Standard

Author: Ji Hua
Created Date: 2026-02-17
Last Modified: 2026-02-17
Current Version: 2

Document Type: Governance
Document Subtype: Document Governance
Document Status: In Development
Document Authority Scope: Global
Document Purpose:
This document defines the documentation governance framework for the Quoridor project. It establishes the official documentation classification system and the mandatory structural format that all documents must follow. This standard ensures structural clarity, prevents cross-type authority overlap, and enables consistent collaboration between human developers and AI coding agents.

---

# Part I — Documentation Classification System

## 1. Purpose of the Classification System

The Quoridor project uses a structured documentation classification system to:

- Separate concerns across documentation layers.
- Prevent semantic overlap between documents.
- Maintain structural clarity.
- Enable predictable authoring behavior for both humans and AI agents.
- Ensure that documentation remains internally consistent.

Every document must declare exactly one Document Type in its header.

Documents must operate strictly within the semantic scope of their declared type.

The system defines five top-level document categories. Subcategories may be used informally, but governance rules apply only at the top-level category.

---

## 2. Document Types

### 2.1 Governance

Governance documents define:

- Documentation rules
- Repository structure rules
- Lifecycle rules
- Change control rules
- Structural constraints
- AI and human development constraints

Governance documents operate at the meta-level and must not define runtime system behavior.

---

### 2.2 Design

Design documents define structural direction and domain-level constraints.

Design documents may define:

- System decomposition
- Module boundaries
- Module responsibilities
- Domain rules and invariants
- State machines
- Lifecycle models
- High-level behavioral constraints

Design documents describe what must exist and what constraints must hold.

Design documents must not define:

- Field-level schema
- Concrete API signatures
- Serialization details
- Internal algorithmic implementation
- Performance strategies

Design defines structural and semantic intent, not implementation details.

---

### 2.3 Interface

Interface documents define cross-module contracts and externally visible structures.

Interface documents may define:

- Cross-module data structures
- Communication formats
- Serialization rules
- Error structures
- Public APIs
- Invocation semantics

Interface documents must:

- Operate strictly within Design-defined responsibilities.
- Not expand module capability boundaries.
- Not introduce new system-level responsibilities.

Interface documents define boundaries, not structure or implementation.

---

### 2.4 Implementation

Implementation documents define internal module realization.

Implementation documents may define:

- Internal module design
- Internal data structures
- Internal algorithms
- Performance strategies
- Class and function organization

Implementation documents must not:

- Modify cross-module contracts.
- Expand module responsibilities.
- Override Design constraints.
- Redefine Interface contracts.

Implementation concretizes Design within the limits of Interface.

---

### 2.5 Other

Other documents include:

- Notes
- Draft ideas
- Informal discussions
- Rationale
- Examples
- Runtime explanations
- Exploratory materials

Other documents have no structural authority and must not be treated as normative references.

---

# Part II — Document Structure Standard

All official Quoridor documents must follow the structure defined in this section.

Human-facing documents must be written in Markdown format. Protocol schemas or API specifications may use other formats where appropriate.

Each document must contain exactly four structural components in the following order:

1. Title
2. Header
3. Content
4. Changelog

---

## 1. Title Requirements

- The title must appear before the header.
- The title must use a single top-level Markdown heading (#).
- Only one top-level heading is allowed per document.
- The title must clearly describe the document subject.
- The title must not contain version numbers.

Example:

# Game Engine Architecture

---

## 2. Header Requirements

The header must appear immediately after the title.

The header must contain the following fields in the exact order listed below:

- Author
- Created Date
- Last Modified
- Current Version
- Document Type
- Document Subtype
- Document Status
- Document Authority Scope
- Document Purpose

No additional header fields are allowed.

Field definitions:

Author
- Required.
- Name of the primary document owner.

Created Date
- Required.
- Format: YYYY-MM-DD.

Last Modified
- Required.
- Format: YYYY-MM-DD.
- Must be updated whenever content changes.

Current Version
- Required.
- Integer only (1, 2, 3, ...).
- Represents document revision count.
- Must increment by 1 for each content modification.

Document Type
- Required.
- Must be one of: Governance, Design, Interface, Implementation, Other.

Document Subtype
- Optional
- May be specified to further clarify the document’s specific purpose or usage within its top-level category.
- Subtypes are informational only and do not introduce additional governance rules or authority levels.
- Governance constraints apply exclusively at the top-level Document Type.

Document Status
- Required.
- Must be one of: Draft, In Development, Final, Deprecated.

Document Authority Scope
- Required.
- Must clearly define scope.
- Examples:
  - Global
  - Engine module
  - Backend module
  - Interface layer

Document Purpose
- Required.
- One concise paragraph explaining what the document defines.
- Must not describe implementation details.

Documents missing any required header field are considered invalid.

---

## 3. Content Rules

The Content section must:

- Operate strictly within the semantic scope of the declared Document Type.
- Avoid defining responsibilities outside its classification.
- Avoid overlapping authority with other document types.
- Be written in structured Markdown format.
- Use consistent heading hierarchy.

Content must not redefine structural authority beyond its type definition.

---

## 4. Changelog Requirements

Every document must contain a Changelog section at the end.

The Changelog must:

- Appear after all content.
- Be ordered in reverse chronological order (latest version first).
- Contain an entry for every version increment.
- Describe changes clearly and concisely.

Changelog format:

Version X (YYYY-MM-DD)
- Description of modification.

Version X-1 (YYYY-MM-DD)
- Description of modification.

Version 1 (YYYY-MM-DD)
- Initial creation.

Rules:

- Every semantic or textual modification requires a version increment.
- Version numbers must increase by exactly 1.
- Final documents may receive textual clarifications but must not change semantics.
- Semantic changes to a Final document require:
  - Marking it as Deprecated.
  - Creating a new document.

---

## 5. Document Lifecycle Summary

Document Status meanings:

Draft
- Conceptual stage.
- No code may depend on it.

In Development
- Code may reference it.
- Content may change freely.

Final
- Semantic behavior defined by this document is frozen.
- Textual clarifications are allowed.
- Semantic modifications are not allowed.

Deprecated
- Document is replaced.
- No new code may depend on it.
- Maintained only for historical reference.

---

## Note on Design Philosophy

This documentation governance structure reflects a layered authority model:

- Governance defines meta-level constraints.
- Design defines structural and semantic intent.
- Interface defines cross-module contracts.
- Implementation defines internal realization.

Other documents provide context but do not carry authority.

---

# Changelog

Version 2 (2026-02-17)
- Replaced Architecture/Protocol/API classification with five top-level categories: Governance, Design, Interface, Implementation, Other.
- Introduced layered authority model.
- Clarified scope boundaries for each document type.

Version 1 (2026-02-17)
- Initial governance document structure defined.
