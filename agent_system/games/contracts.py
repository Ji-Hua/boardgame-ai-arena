"""Game-neutral contracts. Full states never cross the agent decision boundary."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Protocol

def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class GameRef:
    game_id: str
    ruleset: str
    version: int

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != {"game_id", "ruleset", "version"}:
            raise ValueError("An explicit game_id/ruleset/version is required")
        if not all(isinstance(value[k], str) and value[k] for k in ("game_id", "ruleset")):
            raise ValueError("Invalid game identity")
        if type(value["version"]) is not int or value["version"] < 1:
            raise ValueError("Invalid rules version")
        return cls(**value)

    def wire(self):
        return asdict(self)

QUORIDOR = GameRef("quoridor", "standard-9x9", 1)
MR_JACK = GameRef("mr_jack", "london-2016-base", 1)

@dataclass(frozen=True)
class Decision:
    role: str
    observation: Any
    legal_actions: tuple[int, ...]
    decision_index: int

class Game(Protocol):
    ref: GameRef
    roles: tuple[str, ...]
    def initial_state(self): ...
    def actor(self, state) -> str | None: ...
    def observe(self, state, role: str) -> Any: ...
    def legal_actions(self, state) -> tuple[int, ...]: ...
    def apply(self, state, action: int): ...
    def chance_outcomes(self, state) -> tuple[tuple[int, float], ...]: ...
    def returns(self, state) -> dict[str, float]: ...
    def serialize(self, state) -> Any: ...
    def restore(self, value): ...
