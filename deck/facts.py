"""Read measured deck facts without executing demos or cloud calls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from awsai_demo.lineage import load_yaml

if TYPE_CHECKING:
    from pathlib import Path

type FactLane = Literal["offline", "live"]
LANES: tuple[FactLane, ...] = ("offline", "live")


class FactError(ValueError):
    """A reference has no current, contract-approved measurement."""


@dataclass(frozen=True)
class FactDefinition:
    """The editorial contract for one independently measured value."""

    source: str
    how: str
    type: str


def load_contract(root: Path) -> dict[str, dict[str, FactDefinition]]:
    """Load and validate the exact keys the slides may reference."""
    raw = load_yaml(
        (root / "deck/facts_contract.yaml").read_text(encoding="utf-8")
    )
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema_version", *LANES}
        or raw["schema_version"] != 1
    ):
        message = "Invalid facts contract envelope"
        raise FactError(message)
    result: dict[str, dict[str, FactDefinition]] = {}
    for lane in LANES:
        rows = raw[lane]
        if not isinstance(rows, dict):
            message = f"Facts contract {lane} must be a mapping"
            raise FactError(message)
        result[lane] = {}
        for key, definition in rows.items():
            if (
                not isinstance(key, str)
                or not isinstance(definition, dict)
                or set(definition) != {"source", "how", "type"}
                or not all(isinstance(v, str) for v in definition.values())
                or definition["type"]
                not in {
                    "integer",
                    "boolean",
                    "string",
                    "table",
                    "identifier",
                }
            ):
                message = f"Invalid facts contract definition: {lane}.{key}"
                raise FactError(message)
            result[lane][key] = FactDefinition(**definition)
    return result


def measurement_fingerprint(root: Path) -> str:
    """Hash measurement inputs; editorial slide edits are irrelevant."""
    paths = [
        root / name
        for name in (
            "uv.lock",
            "data/aws_ai_stack_notes.md",
            "data/lineage.yaml",
            "data/pricing_snapshot.json",
            "deck/collect_facts.py",
        )
    ]
    paths.extend(
        path
        for path in (root / "src").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes() if path.exists() else b"<missing>"
        digest.update(len(relative).to_bytes(8) + relative)
        digest.update(len(content).to_bytes(8) + content)
    return digest.hexdigest()


def empty_document(
    contract: dict[str, dict[str, FactDefinition]],
) -> dict[str, Any]:
    """Represent uncollected values without inventing facts."""
    return {
        "schema_version": 1,
        **{lane: dict.fromkeys(contract[lane]) for lane in LANES},
        "metadata": {
            lane: {
                "fingerprint": None,
                "timestamp": None,
                "sources": {},
                "missing": dict.fromkeys(
                    contract[lane], "This lane has not been collected"
                ),
            }
            for lane in LANES
        },
    }


def valid_value(value: object, kind: str) -> bool:
    """Accept identifiers as ints or strings; reject bools and nulls."""
    if kind == "integer":
        return type(value) is int
    if kind == "identifier":
        return type(value) is int or type(value) is str
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    return isinstance(value, list) and all(
        isinstance(row, list)
        and all(type(cell) in {str, int, float, bool} for cell in row)
        for row in value
    )


@dataclass
class FactStore:
    """Resolve values and retain readable draft problems."""

    document: dict[str, Any]
    contract: dict[str, dict[str, FactDefinition]]
    fingerprint: str
    problems: list[str] = field(default_factory=list)

    def resolve(self, path: str, *, draft: bool = False) -> object:
        """Return a value or a draft placeholder with its reason."""
        path = path.removeprefix("facts.")
        lane, _, key = path.partition(".")
        reason = self._problem(lane, key)
        if reason is None:
            return self.document[lane][key]
        message = f"{path}: {reason}"
        if not draft:
            raise FactError(message)
        if message not in self.problems:
            self.problems.append(message)
        return f"[MISSING: {message}]"

    def _problem(self, lane: str, key: str) -> str | None:
        if lane not in self.contract or key not in self.contract[lane]:
            return "Reference is outside facts_contract.yaml"
        meta = self.document["metadata"][lane]
        missing = meta["missing"].get(key)
        if missing is not None:
            return str(missing)
        if meta["fingerprint"] != self.fingerprint:
            return "Stale measurement fingerprint; re-record this lane"
        if not valid_value(
            self.document[lane][key], self.contract[lane][key].type
        ):
            return "Recorded value has the wrong type or contains null"
        return None


def load_facts(path: Path, *, root: Path) -> FactStore:
    """Load exact contract keys; preserve explicit uncollected lanes."""
    contract = load_contract(root)
    document = empty_document(contract)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or set(raw) != {"schema_version", "metadata", *LANES}
            or raw["schema_version"] != 1
        ):
            message = "Invalid facts.json envelope"
            raise FactError(message)
        for lane in LANES:
            if (
                not isinstance(raw[lane], dict)
                or set(raw[lane]) != set(contract[lane])
                or not isinstance(raw["metadata"], dict)
                or set(raw["metadata"]) != set(LANES)
                or not isinstance(raw["metadata"][lane], dict)
                or set(raw["metadata"][lane])
                != {"fingerprint", "timestamp", "sources", "missing"}
                or not isinstance(raw["metadata"][lane]["missing"], dict)
                or not isinstance(raw["metadata"][lane]["sources"], dict)
            ):
                message = f"Invalid facts.json keys or metadata for {lane}"
                raise FactError(message)
        document = cast("dict[str, Any]", raw)
    return FactStore(document, contract, measurement_fingerprint(root))
