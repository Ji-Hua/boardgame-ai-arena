"""ParamSchema — lightweight parameter validation for agent specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParamDef:
    """Definition of a single parameter."""

    type: type
    default: Any
    min_val: Any = None
    max_val: Any = None

    def validate(self, value: Any) -> Any:
        """Validate and coerce a single value. Returns the validated value."""
        coerced = self.type(value)
        if self.min_val is not None and coerced < self.min_val:
            raise ValueError(
                f"Value {coerced} below minimum {self.min_val}"
            )
        if self.max_val is not None and coerced > self.max_val:
            raise ValueError(
                f"Value {coerced} above maximum {self.max_val}"
            )
        return coerced


class ParamSchema:
    """Validates and fills defaults for a set of named parameters."""

    def __init__(self, params: dict[str, ParamDef] | None = None) -> None:
        self._params: dict[str, ParamDef] = dict(params or {})

    @property
    def definitions(self) -> dict[str, ParamDef]:
        return dict(self._params)

    def defaults(self) -> dict[str, Any]:
        """Return a dict of all parameters set to their defaults."""
        return {name: p.default for name, p in self._params.items()}

    def validate(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate params, fill missing with defaults, reject unknowns."""
        params = dict(params or {})
        result: dict[str, Any] = {}

        for name, pdef in self._params.items():
            if name in params:
                result[name] = pdef.validate(params[name])
            else:
                result[name] = pdef.default

        unknown = set(params) - set(self._params)
        if unknown:
            raise ValueError(f"Unknown parameters: {unknown}")

        return result

    def is_empty(self) -> bool:
        return len(self._params) == 0
