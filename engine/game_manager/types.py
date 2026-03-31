"""Types for the GameManager module.

ActionResult encapsulates the outcome of submit_action.
State and Action are opaque types from the Rust Rule Engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# State and Action are opaque types provided by the Rust Rule Engine
# via the FFI layer. They are not constructed or mutated in Python.
State = Any
Action = Any
Player = Any


@dataclass(frozen=True)
class ActionResult:
    """Result of a submit_action call."""

    success: bool
    state: State | None = None
    error: str | None = None
