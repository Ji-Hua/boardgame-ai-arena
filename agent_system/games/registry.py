"""Explicit Quoridor registry. Metadata imports do not load engines."""
from .contracts import GameRef, QUORIDOR
DEFINITIONS = ({"ref": QUORIDOR.wire(), "title": "Quoridor", "roles": ["p1", "p2"],
    "perfect_information": True, "chance": False, "training_backend": "quoridor_legacy"},)
def validate_ref(value):
    ref = GameRef.parse(value)
    if ref != QUORIDOR:
        raise ValueError("Unsupported game or rules version")
    return ref
def create_game(value):
    validate_ref(value)
    from .quoridor import Quoridor
    return Quoridor()
