"""Load and validate the shared lifecycle and component datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, Set

STATUSES = frozenset(
    {
        "active",
        "preview",
        "maintenance",
        "sunset",
        "legacy",
        "end_of_life",
        "renamed",
        "moved",
    }
)
_ENTITIES = frozenset(
    {"service", "feature", "model", "api", "package", "endpoint", "product"}
)
_RELATIONS = frozenset(
    {"rename", "successor", "migration", "merged", "moved", "built_on", "none"}
)
_RECORD_FIELDS = frozenset(
    {
        "id",
        "name",
        "former_names",
        "entity_type",
        "scope",
        "official_status",
        "normalized_status",
        "launched",
        "announced",
        "effective",
        "end_of_support",
        "successors",
        "relation",
        "sources",
        "note",
    }
)
_COMPONENT_FIELDS = frozenset(
    {"id", "name", "layer", "lineage", "demos", "slides", "covered"}
)


@dataclass(frozen=True)
class LineageRecord:
    """A sourced lifecycle record with precise status scope."""

    id: str
    name: str
    former_names: tuple[str, ...]
    entity_type: str
    scope: str
    official_status: str
    normalized_status: str
    launched: str | None
    announced: str | None
    effective: str | None
    end_of_support: str | None
    successors: tuple[str, ...]
    relation: str
    sources: tuple[str, ...]
    note: str | None


@dataclass(frozen=True)
class Component:
    """Links from one component to facts, demonstrations and slides."""

    id: str
    name: str
    layer: str
    lineage: tuple[str, ...]
    demos: tuple[str, ...]
    slides: tuple[str, ...]
    covered: bool
    reason: str | None


def load_yaml(text: str) -> object:
    """Parse safe YAML, rejecting duplicate keys."""
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        def construct_mapping(
            self, node: yaml.nodes.MappingNode, deep: bool = False
        ) -> dict[Any, Any]:
            self.flatten_mapping(node)
            keys = [
                self.construct_object(key, deep=deep) for key, _ in node.value
            ]
            if len(keys) != len(set(keys)):
                message = "Duplicate YAML key"
                raise ValueError(message)
            return cast(
                "dict[Any, Any]", super().construct_mapping(node, deep=deep)
            )

    loader = UniqueLoader(text)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        message = "Expected a YAML mapping"
        raise TypeError(message)
    if not all(isinstance(key, str) for key in value):
        message = "Mapping keys must be strings"
        raise ValueError(message)
    return cast("dict[str, object]", value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        message = "Expected a non-empty string"
        raise ValueError(message)
    return value


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        message = "Expected a list of strings"
        raise TypeError(message)
    return tuple(_text(item) for item in value)


def _date(value: object) -> str | None:
    if value is None:
        return None
    text = _text(value)
    if re.fullmatch(r"\d{4}", text):
        date.fromisoformat(text + "-01-01")
    elif re.fullmatch(r"\d{4}-\d{2}", text):
        date.fromisoformat(text + "-01")
    else:
        date.fromisoformat(text)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            message = "Dates must use YYYY, YYYY-MM or YYYY-MM-DD"
            raise ValueError(message)
    return text


def _fields(
    row: Mapping[str, object], required: Set[str], extra: Set[str]
) -> None:
    if not required <= row.keys() or row.keys() - required - extra:
        message = "Missing or unknown fields: " + str(
            row.get("id", "document")
        )
        raise ValueError(message)


def _document(text: str, key: str) -> list[object]:
    document = _mapping(load_yaml(text))
    required = {"schema_version", key}
    if key == "records":
        required.add("verified")
        _date(document.get("verified"))
    _fields(document, required, set())
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
    ):
        message = "Unsupported schema_version"
        raise ValueError(message)
    items = document[key]
    if not isinstance(items, list):
        message = f"{key} must be a list"
        raise TypeError(message)
    return cast("list[object]", items)


def _unique(records: Sequence[LineageRecord | Component]) -> None:
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        message = "Duplicate record id"
        raise ValueError(message)


def _record(value: object) -> LineageRecord:
    row = _mapping(value)
    _fields(row, _RECORD_FIELDS, set())
    for field, allowed in (
        ("entity_type", _ENTITIES),
        ("normalized_status", STATUSES),
        ("relation", _RELATIONS),
    ):
        if _text(row[field]) not in allowed:
            message = f"Invalid {field} in {row['id']}"
            raise ValueError(message)
    sources = _strings(row["sources"])
    if not sources or any(
        urlsplit(url).scheme != "https" or not urlsplit(url).netloc
        for url in sources
    ):
        message = f"HTTPS sources required for {row['id']}"
        raise ValueError(message)
    return LineageRecord(
        id=_text(row["id"]),
        name=_text(row["name"]),
        former_names=_strings(row["former_names"]),
        entity_type=_text(row["entity_type"]),
        scope=_text(row["scope"]),
        official_status=_text(row["official_status"]),
        normalized_status=_text(row["normalized_status"]),
        launched=_date(row["launched"]),
        announced=_date(row["announced"]),
        effective=_date(row["effective"]),
        end_of_support=_date(row["end_of_support"]),
        successors=_strings(row["successors"]),
        relation=_text(row["relation"]),
        sources=sources,
        note=None if row["note"] is None else _text(row["note"]),
    )


def parse_lineage(text: str) -> tuple[LineageRecord, ...]:
    """Validate schema, dates, sources and successor references."""
    records = tuple(_record(row) for row in _document(text, "records"))
    _unique(records)
    ids = {record.id for record in records}
    for record in records:
        if not set(record.successors) <= ids:
            message = f"Unknown successor in {record.id}"
            raise ValueError(message)
    return records


def _component(value: object) -> Component:
    row = _mapping(value)
    _fields(row, _COMPONENT_FIELDS, {"reason"})
    if not isinstance(row["covered"], bool):
        message = "covered must be boolean"
        raise TypeError(message)
    demos, slides = _strings(row["demos"]), _strings(row["slides"])
    reason = None if "reason" not in row else _text(row["reason"])
    if row["covered"]:
        if not demos or not slides:
            message = f"Covered component needs demos and slides: {row['id']}"
            raise ValueError(message)
    elif reason is None:
        message = f"Excluded component needs a reason: {row['id']}"
        raise ValueError(message)
    return Component(
        _text(row["id"]),
        _text(row["name"]),
        _text(row["layer"]),
        _strings(row["lineage"]),
        demos,
        slides,
        row["covered"],
        reason,
    )


def parse_components(
    text: str,
    *,
    lineage_ids: Set[str],
    demo_ids: Set[str],
    slide_ids: Set[str] | None = None,
) -> tuple[Component, ...]:
    """Validate references, including slides when available."""
    records = tuple(_component(row) for row in _document(text, "components"))
    _unique(records)
    covered_demos: set[str] = set()
    for record in records:
        if not set(record.lineage) <= lineage_ids:
            message = f"Unknown lineage reference in {record.id}"
            raise ValueError(message)
        if not set(record.demos) <= demo_ids:
            message = f"Unknown demo reference in {record.id}"
            raise ValueError(message)
        if slide_ids is not None and not set(record.slides) <= slide_ids:
            message = f"Unknown slide reference in {record.id}"
            raise ValueError(message)
        covered_demos.update(record.demos)
    if not demo_ids <= covered_demos:
        message = "Demos missing from components: " + ", ".join(
            sorted(demo_ids - covered_demos)
        )
        raise ValueError(message)
    return records


def data_text(name: str, directory: Path | None = None) -> str:
    """Read curated data from a checkout or from the installed wheel."""
    if name not in {"lineage.yaml", "components.yaml"}:
        message = "Unknown curated data file"
        raise ValueError(message)
    if directory is not None:
        return (directory / name).read_text(encoding="utf-8")
    checkout = Path(__file__).resolve().parents[2] / "data" / name
    if checkout.is_file():
        return checkout.read_text(encoding="utf-8")
    return (
        resources.files("awsai_demo")
        .joinpath("data", name)
        .read_text(encoding="utf-8")
    )


def load_lineage(directory: Path | None = None) -> tuple[LineageRecord, ...]:
    """Load the workshop's lifecycle records without network access."""
    return parse_lineage(data_text("lineage.yaml", directory))
