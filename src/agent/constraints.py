"""Numeric design constraints, enforced by code rather than trusted to the prompt.

A natural-language goal like "keep the roof at least as high as the baseline"
becomes `A_Car_Roof_Height >= 0`. The agent sees the rule in its prompt, but a
proposal that breaks it is refused before scoring either way.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass

from ..contracts import PARAM_BOUNDS

_OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}
_RULE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class Constraint:
    name: str
    op: str
    value: float

    def __str__(self) -> str:
        return f"{self.name} {self.op} {self.value:g}"

    def violated_by(self, params: dict[str, float]) -> bool:
        return not _OPS[self.op](params[self.name], self.value)


def parse_constraint(text: str) -> Constraint:
    match = _RULE.match(text)
    if not match:
        raise ValueError(f"constraint '{text}' must look like 'NAME >= NUMBER' (<=, >=, <, >)")
    name, op, value = match.group(1), match.group(2), float(match.group(3))
    if name not in PARAM_BOUNDS:
        raise ValueError(f"constraint names unknown parameter '{name}'")
    return Constraint(name, op, value)


def violations(params: dict[str, float], constraints: list[Constraint]) -> list[str]:
    return [f"violates constraint {rule} (value {params[rule.name]:g})"
            for rule in constraints if rule.violated_by(params)]
