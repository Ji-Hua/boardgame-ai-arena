"""Explicit, lazy game registry; importing metadata does not load Torch or engines."""
from .contracts import GameRef, QUORIDOR, KUHN_POKER, MR_JACK

DEFINITIONS = (
    {"ref": MR_JACK.wire(), "title": "Mr. Jack · London 2016", "roles": ["detective","jack"],
     "perfect_information": False, "chance": True, "training_backend": "research_neural_cpu"},
    {"ref": QUORIDOR.wire(), "title": "Quoridor", "roles": ["p1", "p2"],
     "perfect_information": True, "chance": False, "training_backend": "quoridor_legacy"},
    {"ref": KUHN_POKER.wire(), "title": "Kuhn Poker", "roles": ["p1", "p2"],
     "perfect_information": False, "chance": True, "training_backend": "openspiel_cpu"},
)

def validate_ref(value):
    ref = GameRef.parse(value)
    if ref not in (QUORIDOR, KUHN_POKER, MR_JACK):
        raise ValueError("Unsupported game or rules version")
    return ref

def create_game(value):
    ref = validate_ref(value)
    if ref == QUORIDOR:
        from .quoridor import Quoridor
        return Quoridor()
    if ref == MR_JACK:
        from .mrjack import MrJack
        return MrJack()
    from .openspiel import KuhnPoker
    return KuhnPoker()
