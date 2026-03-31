"""GameManager package — stateful game orchestration over the Rust Rule Engine."""

from engine.game_manager.types import ActionResult


def __getattr__(name):
    if name == "GameManager":
        from engine.game_manager.game_manager import GameManager
        return GameManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GameManager", "ActionResult"]
