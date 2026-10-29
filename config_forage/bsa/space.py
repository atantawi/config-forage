"""Decision space: user-provided design axes, each with a finite set of options.

Extracted from the author's earlier BSA research code (commit c4339d2)."""

from __future__ import annotations

import math
from dataclasses import dataclass

Decision = tuple[int | None, ...]  # one option index per axis; None where the axis does not apply


@dataclass(frozen=True)
class Axis:
    name: str
    description: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class DecisionSpace:
    axes: tuple[Axis, ...]

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(len(a.options) for a in self.axes)

    def n_combinations(self) -> int:
        return math.prod(self.sizes)

    def render_hint(self) -> str:
        lines = ["Consider design decisions along these axes:"]
        for a in self.axes:
            lines.append(f"- {a.name} ({a.description}) options: " + "; ".join(a.options))
        return "\n".join(lines)

    def render_brief(self, d: Decision) -> str:
        return "\n".join(f"- {a.name}: {a.options[i]}" for a, i in zip(self.axes, d) if i is not None)

    def labels(self, d: Decision) -> dict[str, str]:
        return {a.name: a.options[i] for a, i in zip(self.axes, d) if i is not None}

    def to_json(self) -> dict:
        return {"axes": [{"name": a.name, "description": a.description, "options": list(a.options)}
                         for a in self.axes]}

    @classmethod
    def from_json(cls, obj: dict) -> "DecisionSpace":
        return cls(tuple(Axis(a["name"], a.get("description", ""), tuple(a["options"])) for a in obj["axes"]))
