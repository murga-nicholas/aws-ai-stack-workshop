"""Strict slides resolved from local facts and lineage data."""

from __future__ import annotations

import re
import shlex
from dataclasses import asdict, dataclass, fields
from datetime import date
from typing import TYPE_CHECKING, Any, Never, cast

from awsai_demo.cli import build_parser
from awsai_demo.lineage import STATUSES, load_lineage, load_yaml
from deck.facts import load_contract
from deck.notes import load_notes
from deck.snippets import extract_snippet

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from awsai_demo.lineage import LineageRecord

type Resolver = Callable[[str], object]
type Row = dict[str, Any]
_REFERENCE = re.compile(r"\{((?:facts|lineage)\.[\w.-]+)\}")
_CONTENT_W = (18288000 - 2 * 1439997) / 914400
_MAIN_W = 11338560 / 914400
_RAIL_W = 3721608 / 914400


@dataclass(frozen=True, kw_only=True)
class Block:
    """Common content block position metadata."""

    type: str
    h: float | None = None


@dataclass(frozen=True, kw_only=True)
class TextBlock(Block):
    """One styled paragraph."""

    text: str
    size: float = 16
    bold: bool = False
    tone: str = "default"


@dataclass(frozen=True, kw_only=True)
class BulletsBlock(Block):
    """A list with explicitly validated indentation levels."""

    items: tuple[Row, ...]
    size: float = 16


@dataclass(frozen=True, kw_only=True)
class TableBlock(Block):
    """Resolved rectangular data with relative column widths."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    widths: tuple[float, ...] = ()


@dataclass(frozen=True, kw_only=True)
class LineageBlock(Block):
    """Lineage records expanded into visible migration rows."""

    ids: tuple[str, ...]
    rows: tuple[Row, ...]


@dataclass(frozen=True, kw_only=True)
class TimelineBlock(Block):
    """Dated events ordered chronologically."""

    events: tuple[Row, ...]


@dataclass(frozen=True, kw_only=True)
class LayerMapBlock(Block):
    """Named architectural layers and their labelled items."""

    layers: tuple[Row, ...]


@dataclass(frozen=True, kw_only=True)
class DiagramBlock(Block):
    """Nodes and edges in a bounded local coordinate system."""

    nodes: tuple[Row, ...]
    edges: tuple[Row, ...]
    boundaries: tuple[Row, ...] = ()


@dataclass(frozen=True, kw_only=True)
class CodeBlock(Block):
    """A viewport into tested Python source, never inline code."""

    ref: str
    code: str
    caption: str = ""


@dataclass(frozen=True, kw_only=True)
class MetricsBlock(Block):
    """Formatted scalar values and descriptive labels."""

    items: tuple[Row, ...]


@dataclass(frozen=True, kw_only=True)
class CalloutBlock(Block):
    """An emphasized statement with a semantic tone."""

    text: str
    tone: str


@dataclass(frozen=True, kw_only=True)
class CompareBlock(Block):
    """Two parallel columns of labelled statements."""

    left: Row
    right: Row


@dataclass(frozen=True, kw_only=True)
class MatrixBlock(TableBlock):
    """A fact-derived responsibility matrix."""


@dataclass(frozen=True, kw_only=True)
class StepsBlock(Block):
    """Ordered short instructions."""

    items: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ImageBlock(Block):
    """An image restricted to the repository's presentation assets."""

    path: str
    caption: str = ""


@dataclass(frozen=True, kw_only=True)
class LinksBlock(Block):
    """Titled cards of clickable documentation links."""

    groups: tuple[Row, ...]


@dataclass(frozen=True, kw_only=True)
class SourceNoteBlock(Block):
    """A visible source attribution."""

    text: str


@dataclass(frozen=True)
class Section:
    """Agenda segment shared by slides in one YAML file."""

    id: str
    index: str
    title: str
    minutes: tuple[float, float] | None


@dataclass(frozen=True, kw_only=True)
class Slide:
    """A validated slide and its presenter script."""

    id: str
    kind: str
    eyebrow: str
    title: str
    deck: str
    section: Section
    notes: str
    audience: tuple[str, ...] = ()
    layout: str = "full"
    main: tuple[Block, ...] = ()
    rail: tuple[Block, ...] = ()
    run: tuple[str, ...] = ()
    source_note: str = ""
    bullets: tuple[str, ...] = ()


@dataclass(frozen=True)
class TitleSlide(Slide):
    """Opening title and presenter attribution."""


@dataclass(frozen=True)
class SectionSlide(Slide):
    """A numbered section divider."""


@dataclass(frozen=True)
class ContentSlide(Slide):
    """A presentation slide with stacked content blocks."""


@dataclass(frozen=True)
class ClosingSlide(Slide):
    """Closing title and next step."""


@dataclass(frozen=True)
class AppendixSlide(Slide):
    """A reference slide outside the presented timeline."""


@dataclass(frozen=True)
class Deck:
    """An ordered set of presentation sections and slides."""

    sections: tuple[Section, ...]
    slides: tuple[Slide, ...]


def _fail(message: str) -> Never:
    raise ValueError(message)


def _mapping(value: object) -> Row:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in value
    ):
        _fail("Expected a mapping with string keys")
    return cast("Row", value)


def _fields(row: Row, required: str, optional: str = "") -> None:
    missing = set(required.split()) - row.keys()
    unknown = row.keys() - set((required + " " + optional).split())
    if missing or unknown:
        _fail(f"Missing fields {sorted(missing)}; unknown {sorted(unknown)}")


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("Expected nonempty text")
    return value


def _list(value: object) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        _fail("Expected a list")
    return list(value)


def _strings(value: object) -> tuple[str, ...]:
    return tuple(_text(item) for item in _list(value))


def _number(value: object, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        _fail("Expected a number")
    number = float(cast("float", value))
    if number < 0 or (positive and number == 0):
        _fail("Expected a positive size or nonnegative coordinate")
    return number


def _choice(value: object, choices: set[str] | frozenset[str]) -> str:
    text = _text(value)
    if text not in choices:
        _fail(f"Unknown value {text}; expected {sorted(choices)}")
    return text


def format_value(value: object, *, kind: str | None = None) -> str:
    """Format visible values, rejecting nulls and structured objects.

    An identifier is rendered verbatim. Other integers use grouping.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        if kind == "identifier":
            return str(value)
        return f"{value:,}"
    if isinstance(value, (str, float)):
        return str(value)
    return _fail("Visible values must be non-null scalars")


def parse_run(command: str) -> tuple[str, ...]:
    """Validate a command with the real application parser."""
    args = shlex.split(command)
    if args[:2] == ["uv", "run"]:
        args = args[2:]
    if not args or args[0] != "awsai-demo" or "--help" in args:
        _fail(f"Expected an awsai-demo command: {command}")
    build_parser().parse_args(args[1:])
    return tuple(args[1:])


def _fact_kinds(root: Path) -> dict[str, str]:
    """Read contract types for fact references."""
    return {
        f"facts.{lane}.{key}": item.type
        for lane, items in load_contract(root).items()
        for key, item in items.items()
    }


class _Context:
    def __init__(
        self,
        root: Path,
        resolve: Resolver | None,
        *,
        draft: bool = False,
        problems: list[str] | None = None,
    ) -> None:
        self.root = root
        self.resolve = resolve
        self.draft = draft
        self.problems = [] if problems is None else problems
        self.records = {
            record.id: record for record in load_lineage(root / "data")
        }
        self.fact_kinds = _fact_kinds(root)

    def content_problem(self, message: str) -> None:
        if not self.draft:
            _fail(message)
        self.problems.append(message)

    def record(self, identity: str) -> LineageRecord:
        if identity not in self.records:
            _fail(f"Unknown lineage id: {identity}")
        return self.records[identity]

    def field(self, record: LineageRecord, name: str) -> object:
        if name not in {field.name for field in fields(record)}:
            _fail(f"Unknown lineage field: {name}")
        value: object = getattr(record, name)
        if name == "successors":
            return ", ".join(
                self.record(item).name for item in record.successors
            )
        if isinstance(value, tuple):
            return ", ".join(value)
        return value

    def lookup(self, path: str) -> object:
        if path.startswith("lineage."):
            _, identity, name = path.split(".", 2)
            return self.field(self.record(identity), name)
        if self.resolve is None:
            _fail(f"Facts resolver required for {path}")
        return self.resolve(path)

    def interpolate(self, value: object) -> object:
        if isinstance(value, str):
            remainder = _REFERENCE.sub("", value)
            if "{" in remainder or "}" in remainder:
                self.content_problem(f"Unsupported interpolation: {value}")
            return _REFERENCE.sub(
                lambda match: format_value(
                    self.lookup(match.group(1)),
                    kind=self.fact_kinds.get(match.group(1)),
                ),
                value,
            )
        if isinstance(value, list):
            return [self.interpolate(item) for item in value]
        if isinstance(value, dict):
            return {key: self.interpolate(item) for key, item in value.items()}
        return value


def _rows(
    value: object, columns: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, str) and value.startswith("[MISSING:"):
        return ((value, *("" for _ in columns[1:])),)
    rows = tuple(
        tuple(format_value(cell) for cell in _list(row))
        for row in _list(value)
    )
    if not columns or any(len(row) != len(columns) for row in rows):
        _fail("Table rows must match nonempty columns")
    return rows


def _source(
    source: object, context: _Context
) -> tuple[object, tuple[str, ...]]:
    row = _mapping(source)
    if set(row) == {"facts"}:
        return context.lookup("facts." + _text(row["facts"])), ()
    _fields(row, "lineage")
    selection = _mapping(row["lineage"])
    _fields(selection, "filter fields")
    filters = _mapping(selection["filter"])
    names = _strings(selection["fields"])
    values = []
    for record in context.records.values():
        matched = all(
            context.field(record, key)
            in (value if isinstance(value, list) else [value])
            for key, value in filters.items()
        )
        if matched:
            values.append(
                [
                    "-"
                    if (value := context.field(record, name)) is None
                    else value
                    for name in names
                ]
            )
    return values, tuple(name.replace("_", " ").title() for name in names)


def _table(row: Row, context: _Context) -> TableBlock:
    kind = row["type"]
    _fields(row, "type", "h columns rows source widths")
    if ("source" in row) == ("rows" in row):
        _fail("Table requires exactly one of rows or source")
    values, default_columns = (
        _source(row["source"], context)
        if "source" in row
        else (row["rows"], ())
    )
    columns = _strings(row.get("columns", default_columns))
    widths = tuple(_number(item) for item in _list(row.get("widths", [])))
    if widths and len(widths) != len(columns):
        _fail("Table widths must match its columns")
    cls = MatrixBlock if kind == "matrix" else TableBlock
    return cls(
        type=kind,
        h=row.get("h"),
        columns=columns,
        rows=_rows(values, columns),
        widths=widths,
    )


def _timeline(row: Row, context: _Context) -> TimelineBlock:
    _fields(row, "type", "h events ids date_field")
    if ("events" in row) == ("ids" in row):
        _fail("Timeline requires exactly one of events or ids")
    events = []
    if "ids" in row:
        for identity in _strings(row["ids"]):
            record = context.record(identity)
            value = context.field(record, row.get("date_field", "launched"))
            events.append({"date": _text(value), "label": record.name})
    else:
        for item in _list(row["events"]):
            event = _mapping(item)
            _fields(event, "date label", "lane")
            events.append({key: _text(value) for key, value in event.items()})

    def sort_key(event: Row) -> date:
        value = event["date"]
        return date.fromisoformat((value + "-01-01")[:10])

    return TimelineBlock(
        type="timeline",
        h=row.get("h"),
        events=tuple(sorted(events, key=sort_key)),
    )


def _diagram(row: Row, width: float, context: _Context) -> DiagramBlock:
    _fields(row, "type nodes edges", "h boundaries")
    nodes = []
    height = row.get("h", 4.0)
    for value in _list(row["nodes"]):
        node = _mapping(value)
        _fields(node, "id label x y w h", "kind lane")
        normalized = dict(node)
        for key in ("id", "label", "lane"):
            if key in node:
                normalized[key] = _text(node[key])
        for key in ("x", "y", "w", "h"):
            normalized[key] = _number(node[key], positive=key in {"w", "h"})
        normalized["kind"] = _choice(
            node.get("kind", "default"),
            {"default", "guard", "store", "external", "note"},
        )
        if node["x"] + node["w"] > width or node["y"] + node["h"] > height:
            context.content_problem(
                f"Diagram node outside block: {node['id']}"
            )
        nodes.append(normalized)
    identities = [node["id"] for node in nodes]
    if len(identities) != len(set(identities)):
        _fail("Duplicate diagram node id")
    edges = []
    for value in _list(row["edges"]):
        edge = _mapping(value)
        _fields(edge, "from to", "label style")
        for key in ("from", "to"):
            _choice(edge[key], set(identities))
        if "label" in edge:
            _text(edge["label"])
        _choice(edge.get("style", "solid"), {"solid", "dashed"})
        edges.append(edge)
    boundaries = []
    for value in _list(row.get("boundaries", [])):
        boundary = _mapping(value)
        _fields(boundary, "label nodes")
        _text(boundary["label"])
        for identity in _strings(boundary["nodes"]):
            _choice(identity, set(identities))
        boundaries.append(boundary)
    return DiagramBlock(
        type="diagram",
        h=row.get("h"),
        nodes=tuple(nodes),
        edges=tuple(edges),
        boundaries=tuple(boundaries),
    )


def _layer_map(row: Row, context: _Context) -> LayerMapBlock:
    _fields(row, "type layers", "h")
    layers = []
    for value in _list(row["layers"]):
        layer = _mapping(value)
        _fields(layer, "name items")
        _text(layer["name"])
        for item in _list(layer["items"]):
            entry = _mapping(item)
            _fields(entry, "label", "status ref")
            _text(entry["label"])
            if "status" in entry:
                _choice(entry["status"], STATUSES)
            if "ref" in entry:
                context.record(_text(entry["ref"]))
        layers.append(layer)
    return LayerMapBlock(
        type="layer_map", h=row.get("h"), layers=tuple(layers)
    )


def _links(row: Row) -> LinksBlock:
    _fields(row, "type groups", "h")
    groups = []
    for value in _list(row["groups"]):
        group = _mapping(value)
        _fields(group, "title items")
        links = []
        for item in _list(group["items"]):
            link = _mapping(item)
            _fields(link, "label url")
            url = _text(link["url"])
            if not url.startswith("https://"):
                _fail(f"Link url must use https: {url}")
            links.append({"label": _text(link["label"]), "url": url})
        if not links:
            _fail("Link group needs at least one link")
        groups.append({"title": _text(group["title"]), "items": links})
    if not groups:
        _fail("Links block needs at least one group")
    return LinksBlock(type="links", h=row.get("h"), groups=tuple(groups))


def _block(value: object, context: _Context, width: float) -> Block:
    row = _mapping(value)
    kind = _text(row.get("type"))
    if "h" in row:
        _number(row["h"])
    base: Row = {"type": kind, "h": row.get("h")}
    if kind in {"table", "matrix"}:
        return _table(row, context)
    if kind == "timeline":
        return _timeline(row, context)
    if kind == "diagram":
        return _diagram(row, width, context)
    if kind == "layer_map":
        return _layer_map(row, context)
    if kind == "links":
        return _links(row)
    if kind == "lineage":
        _fields(row, "type ids", "h")
        identities = _strings(row["ids"])
        rows = []
        for identity in identities:
            record = context.record(identity)
            dates = [
                getattr(record, name)
                for name in ("launched", "effective", "end_of_support")
            ]
            rows.append(
                {
                    "before": ", ".join(record.former_names) or record.name,
                    "relation": record.relation,
                    "after": context.field(record, "successors") or "-",
                    "status": record.normalized_status,
                    "dates": " | ".join(value for value in dates if value),
                }
            )
        return LineageBlock(**base, ids=identities, rows=tuple(rows))
    if kind in {"text", "callout", "source_note"}:
        _fields(
            row,
            "type text",
            "h size bold tone"
            if kind == "text"
            else "h tone"
            if kind == "callout"
            else "h",
        )
        text = _text(row["text"])
        if kind == "source_note":
            return SourceNoteBlock(**base, text=text)
        if kind == "callout":
            tone = _choice(row.get("tone"), {"info", "warning", "success"})
            return CalloutBlock(**base, text=text, tone=tone)
        if not isinstance(row.get("bold", False), bool):
            _fail("Text bold must be boolean")
        return TextBlock(
            **base,
            text=text,
            size=_number(row.get("size", 16)),
            bold=row.get("bold", False),
            tone=_text(row.get("tone", "default")),
        )
    if kind in {"bullets", "steps", "metrics"}:
        _fields(row, "type items", "h size" if kind == "bullets" else "h")
        if kind == "steps":
            return StepsBlock(**base, items=_strings(row["items"]))
        items: list[Row] = []
        for item in _list(row["items"]):
            if kind == "metrics":
                metric = _mapping(item)
                _fields(metric, "value label")
                items.append(
                    {
                        "value": format_value(metric["value"]),
                        "label": _text(metric["label"]),
                    }
                )
            else:
                bullet = (
                    {"text": item} if isinstance(item, str) else _mapping(item)
                )
                _fields(bullet, "text", "level")
                level = bullet.get("level", 0)
                if type(level) is not int or level < 0:
                    _fail("Bullet level must be a nonnegative integer")
                items.append({"text": _text(bullet["text"]), "level": level})
        if kind == "metrics":
            return MetricsBlock(**base, items=tuple(items))
        return BulletsBlock(
            **base, items=tuple(items), size=_number(row.get("size", 16))
        )
    if kind == "compare":
        _fields(row, "type left right", "h")
        columns = []
        for name in ("left", "right"):
            column = _mapping(row[name])
            _fields(column, "title items")
            columns.append(
                {
                    "title": _text(column["title"]),
                    "items": _strings(column["items"]),
                }
            )
        return CompareBlock(**base, left=columns[0], right=columns[1])
    if kind == "code":
        _fields(row, "type ref", "h caption")
        ref = _text(row["ref"])
        return CodeBlock(
            **base,
            ref=ref,
            code=extract_snippet(ref, context.root),
            caption=_text(row["caption"]) if "caption" in row else "",
        )
    if kind == "image":
        _fields(row, "type path", "h caption")
        path = _text(row["path"])
        resolved = (context.root / path).resolve()
        assets = (context.root / "deck" / "assets").resolve()
        if not resolved.is_relative_to(assets) or not resolved.is_file():
            _fail(f"Image must exist under deck/assets: {path}")
        return ImageBlock(
            **base,
            path=path,
            caption=_text(row["caption"]) if "caption" in row else "",
        )
    return _fail(f"Unknown block type: {kind}")


def _section(value: object) -> Section:
    row = _mapping(value)
    _fields(row, "id index title minutes")
    minutes = row["minutes"]
    parsed: tuple[float, float] | None = None
    if minutes is not None:
        values = _list(minutes)
        if len(values) != 2:
            _fail("Section minutes must be [start, end]")
        parsed = (_number(values[0], positive=False), _number(values[1]))
        if parsed[0] >= parsed[1]:
            _fail("Section minutes must increase")
    return Section(
        _text(row["id"]), _text(row["index"]), _text(row["title"]), parsed
    )


def _slide(
    value: object,
    section: Section,
    context: _Context,
    notes: Mapping[str, str],
) -> Slide:
    row = _mapping(context.interpolate(value))
    kind = row.get("kind")
    classes = {
        "title": TitleSlide,
        "section": SectionSlide,
        "content": ContentSlide,
        "closing": ClosingSlide,
        "appendix": AppendixSlide,
    }
    selected = _choice(kind, set(classes))
    extra = (
        "layout main rail"
        if selected in {"content", "appendix"}
        else ("bullets" if selected == "section" else "")
    )
    _fields(
        row, "id kind eyebrow title deck", "audience run source_note " + extra
    )
    identity = _text(row["id"])
    if not identity.startswith(section.id + "."):
        _fail(f"Slide id must start with {section.id}.")
    if identity not in notes:
        _fail(f"Missing notes section: {identity}")
    layout = _choice(row.get("layout", "full"), {"full", "main_rail"})
    if layout == "full" and row.get("rail"):
        _fail("Full layout cannot have a rail")
    commands = _strings(row.get("run", []))
    for command in commands:
        parse_run(command)
    main = tuple(
        _block(block, context, _CONTENT_W if layout == "full" else _MAIN_W)
        for block in _list(row.get("main", []))
    )
    rail = tuple(
        _block(block, context, _RAIL_W) for block in _list(row.get("rail", []))
    )
    return classes[selected](
        id=identity,
        kind=selected,
        eyebrow=_text(row["eyebrow"]),
        title=_text(row["title"]),
        deck=_text(row["deck"]),
        section=section,
        notes=notes[identity],
        audience=_strings(row.get("audience", [])),
        layout=layout,
        main=main,
        rail=rail,
        run=commands,
        source_note=_text(row["source_note"]) if "source_note" in row else "",
        bullets=_strings(row.get("bullets", [])),
    )


def load_deck(
    content_dir: Path,
    notes_path: Path,
    *,
    root: Path,
    resolve: Resolver | None = None,
    draft: bool = False,
    problems: list[str] | None = None,
) -> Deck:
    """Load slide files in filename order, validating all references."""
    context = _Context(root, resolve, draft=draft, problems=problems)
    notes = load_notes(notes_path)
    sections = []
    slides: list[Slide] = []
    for path in sorted(content_dir.glob("*.yaml")):
        row = _mapping(load_yaml(path.read_text(encoding="utf-8")))
        _fields(row, "schema_version section slides")
        if (
            type(row["schema_version"]) is not int
            or row["schema_version"] != 1
        ):
            _fail("Unsupported slide schema_version")
        section = _section(context.interpolate(row["section"]))
        sections.append(section)
        slides.extend(
            _slide(value, section, context, notes)
            for value in _list(row["slides"])
        )
    identities = [slide.id for slide in slides]
    if not slides or len(identities) != len(set(identities)):
        _fail("Slides must exist and their ids must be unique")
    section_ids = [section.id for section in sections]
    if len(section_ids) != len(set(section_ids)):
        _fail("Section ids must be unique")
    if set(notes) - set(identities):
        _fail(
            "Notes sections without slides: "
            + ", ".join(sorted(set(notes) - set(identities)))
        )
    return Deck(tuple(sections), tuple(slides))


def validate_components(deck: Deck, path: Path) -> None:
    """Check component slide references against the deck."""
    row = _mapping(load_yaml(path.read_text(encoding="utf-8")))
    identities = {slide.id for slide in deck.slides}
    for component in _list(row["components"]):
        missing = set(_strings(component["slides"])) - identities
        if missing:
            _fail(
                f"Component {component['id']} references missing slides: "
                f"{sorted(missing)}"
            )


def block_data(block: Block) -> Row:
    """Expose typed block fields to the drawing layer."""
    return asdict(block)


def visible_text(slide: Slide) -> tuple[str, ...]:
    """Collect visible text, including source snippets and tables."""
    ignored = {
        "type",
        "h",
        "size",
        "bold",
        "tone",
        "widths",
        "ids",
        "ref",
        "path",
        "x",
        "y",
        "w",
        "id",
        "kind",
        "lane",
        "level",
        "style",
        "from",
        "to",
        "nodes",
    }

    def texts(value: object, key: str = "") -> list[str]:
        if key in ignored and key != "nodes":
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, dict):
            return [
                text
                for child_key, item in value.items()
                for text in texts(item, child_key)
            ]
        if isinstance(value, (tuple, list)):
            return [text for item in value for text in texts(item)]
        return []

    return (
        slide.eyebrow,
        slide.title,
        slide.deck,
        *slide.bullets,
        *(
            text
            for block in (*slide.main, *slide.rail)
            for text in texts(block_data(block))
        ),
        *slide.run,
        *([slide.source_note] if slide.source_note else []),
    )
