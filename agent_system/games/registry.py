"""Explicit, lazy game registry; importing metadata does not load Torch or engines."""
from .contracts import GameRef, QUORIDOR, KUHN_POKER

DEFINITIONS = (
    {"ref": QUORIDOR.wire(), "title": "Quoridor", "roles": ["p1", "p2"],
     "perfect_information": True, "chance": False, "training_backend": "quoridor_legacy"},
    {"ref": KUHN_POKER.wire(), "title": "Kuhn Poker", "roles": ["p1", "p2"],
     "perfect_information": False, "chance": True, "training_backend": None},
)

def validate_ref(value):
    ref = GameRef.parse(value)
    if ref not in (QUORIDOR, KUHN_POKER):
        raise ValueError("Unsupported game or rules version")
    return ref

def create_game(value):
    ref = validate_ref(value)
    if ref == QUORIDOR:
        from .quoridor import Quoridor
        return Quoridor()
    from .openspiel import KuhnPoker
    return KuhnPoker()
