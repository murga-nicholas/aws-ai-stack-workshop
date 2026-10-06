"""Reusable blocks adapted from the sibling's DataArt components.

All text keeps a readable fixed size. Author-provided heights allocate
space; the build reports overflow instead of silently shrinking type.
Colour follows the siblings: white cards with accent bars, navy panels
in the rail, highlighted code, and a schedule bar for the agenda.
"""

from __future__ import annotations

import keyword
import re
from typing import TYPE_CHECKING, Any, TypedDict

from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

from deck import theme as T

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pptx.slide import Slide

DEFAULT_HEIGHTS = {
    "text": 0.8,
    "bullets": 2.2,
    "table": 3.6,
    "lineage": 3.6,
    "timeline": 4.0,
    "layer_map": 4.5,
    "diagram": 3.0,
    "code": 4.2,
    "metrics": 1.6,
    "callout": 1.2,
    "compare": 2.8,
    "matrix": 4.5,
    "steps": 3.2,
    "image": 3.0,
    "links": 5.4,
    "source_note": 0.45,
}
# The speaker notes name these colours ("the amber callout").
TONES = {"info": T.P.blue, "warning": T.P.amber, "success": T.P.teal}
# Reference cards follow the sibling's teal, sky, amber order.
LINK_ACCENTS = (T.P.teal, T.P.sky, T.P.amber, T.P.violet)
LINK_MARK = "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}"
# Prose drawn on the navy panel when it sits in the right-hand rail.
RAIL_PROSE = frozenset({"text", "callout", "bullets", "steps"})
COMMAND = "awsai-demo"
SCHEDULE = re.compile(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?")
CODE_TOKEN = re.compile(
    r"(?P<comment>#.*)"
    r"|(?P<string>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
    r"|(?P<decorator>@[\w.]+)"
    r"|(?P<word>\b[A-Za-z_]\w*\b)"
)
CODE_COLOURS = {
    "comment": T.P.code_dim,
    "string": T.P.yellow,
    "decorator": T.P.teal,
    "word": T.P.sky,
}
TRIPLE = '"""'


class Area(TypedDict):
    """Named drawing coordinates, shared by all component primitives."""

    x: int
    y: int
    w: int
    h: int


def item_runs(
    items: Sequence[Any], *, numbered: bool, marker: str
) -> list[list[T.Run]]:
    """Render bullet levels or numbered steps with coloured markers."""
    lines: list[list[T.Run]] = []
    for index, item in enumerate(items, 1):
        level = int(item.get("level", 0)) if isinstance(item, dict) else 0
        text = str(item["text"]) if isinstance(item, dict) else str(item)
        sign = f"{'  ' * level}{index}. " if numbered else f"{'  ' * level}• "
        lines.append([(sign, {"colour": marker, "bold": True}), (text, {})])
    return lines


def wrapped(text: str, width: int, size: float) -> int:
    """Count the lines bold text needs, erring long like the checks."""
    per_line = max(int(width / (size * 0.6 * 12700)), 1)
    return sum(-(-max(len(line), 1) // per_line) for line in text.split("\n"))


def _tokens(text: str) -> list[T.Run]:
    """Split one line of code into coloured runs, keeping every char."""
    runs: list[T.Run] = []
    last = 0
    for match in CODE_TOKEN.finditer(text):
        kind = str(match.lastgroup)
        if kind == "word" and match.group() not in keyword.kwlist:
            continue
        runs.append((text[last : match.start()], {}))
        runs.append((match.group(), {"colour": CODE_COLOURS[kind]}))
        last = match.end()
    runs.append((text[last:], {}))
    return [run for run in runs if run[0]]


def code_lines(code: str) -> list[list[T.Run]]:
    """Highlight Python the way the sibling's code panels do.

    Keywords are sky, strings yellow, comments dim and decorators teal;
    a triple-quoted string keeps its colour across lines.
    """
    lines: list[list[T.Run]] = []
    quoted = False
    for line in code.split("\n"):
        runs: list[T.Run] = []
        rest = line
        if quoted:
            end = rest.find(TRIPLE)
            if end < 0:
                lines.append([(rest, {"colour": T.P.yellow})])
                continue
            runs.append((rest[: end + 3], {"colour": T.P.yellow}))
            rest, quoted = rest[end + 3 :], False
        if rest.count(TRIPLE) % 2:
            start = rest.rfind(TRIPLE)
            runs.extend(_tokens(rest[:start]))
            runs.append((rest[start:], {"colour": T.P.yellow}))
            quoted = True
        else:
            runs.extend(_tokens(rest))
        lines.append(runs or [("", {})])
    return lines


def table(
    slide: Slide,
    columns: list[str],
    rows: list[list[str]],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    widths: list[float] | None = None,
    chips: bool = False,
) -> None:
    """Draw a 14-point ruled table, preserving overflow for review.

    ``chips`` draws a matrix: each value after the first column sits
    in a pill whose colour is shared by every cell with that value.
    """
    shares = widths or [1.0] * len(columns)
    row_h = (h - T.inches(0.45)) // max(len(rows), 1)
    mono = {
        index
        for index, cells in enumerate(zip(*rows, strict=True))
        if all(cell == "-" or cell.startswith(COMMAND) for cell in cells)
    }
    owners: dict[str, str] = {}
    for row in rows:
        for cell in row[1:]:
            owners.setdefault(cell, T.ACCENTS[len(owners) % len(T.ACCENTS)])
    offset = x
    for index, (heading, share) in enumerate(
        zip(columns, shares, strict=True)
    ):
        width = round(w * share / sum(shares))
        T.box(
            slide,
            heading.upper(),
            x=offset,
            y=y,
            w=width - T.inches(0.12),
            h=T.inches(0.38),
            size=12.5,
            colour=T.P.faint,
            bold=True,
        )
        for row_index, row in enumerate(rows):
            top = y + T.inches(0.45) + row_index * row_h
            if chips and index:
                fill, chip_h = owners[row[index]], T.inches(0.36)
                chip_y = top + (row_h - T.inches(0.05) - chip_h) // 2
                chip_w = min(
                    width - T.inches(0.12),
                    round(len(row[index]) * 13 * 0.6 * 12700) + T.inches(0.3),
                )
                T.panel(
                    slide,
                    x=offset,
                    y=chip_y,
                    w=chip_w,
                    h=chip_h,
                    fill=fill,
                    line=None,
                    radius=T.PILL,
                )
                T.box(
                    slide,
                    row[index],
                    x=offset,
                    y=chip_y,
                    w=chip_w,
                    h=chip_h,
                    size=13,
                    colour=T.on_fill(fill),
                    bold=True,
                    align=PP_ALIGN.CENTER,
                    anchor=MSO_ANCHOR.MIDDLE,
                )
                continue
            T.box(
                slide,
                row[index],
                x=offset,
                y=top,
                w=width - T.inches(0.12),
                h=row_h - T.inches(0.05),
                size=14,
                colour=T.P.ink_soft if index == 0 else T.P.body,
                bold=index == 0,
                font=T.FONT_CODE if index in mono else T.FONT_UI,
                anchor=MSO_ANCHOR.MIDDLE,
            )
        offset += width
    T.bar(slide, x=x, y=y + T.inches(0.4), w=w, h=12700, colour=T.P.rule)
    for row_index in range(1, len(rows)):
        T.bar(
            slide,
            x=x,
            y=y + T.inches(0.42) + row_index * row_h,
            w=w,
            h=12700,
            colour=T.P.rule,
        )


def is_schedule(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> bool:
    """Tell whether a table is a minute-by-minute agenda."""
    return (
        bool(rows)
        and columns[0].lower() == "when"
        and all(SCHEDULE.fullmatch(row[0]) for row in rows)
    )


def agenda(
    slide: Slide,
    columns: list[str],
    rows: list[list[str]],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    widths: list[float] | None = None,
    schedule: dict[str, str] | None = None,
) -> None:
    """Draw the siblings' agenda: a minute bar, then one card per row.

    ``schedule`` maps a row's minute range to its section accent, so
    the bar, the row markers and the section eyebrows share a colour.
    """
    colours = [
        (schedule or {}).get(row[0], T.ACCENTS[index % len(T.ACCENTS)])
        for index, row in enumerate(rows)
    ]
    spans = [[float(part) for part in row[0].split("-")] for row in rows]
    total = max(end for _, end in spans)
    bar_h = T.inches(0.36)
    for (start, end), row, colour in zip(spans, rows, colours, strict=True):
        left = x + round(w * start / total)
        width = round(w * (end - start) / total) - T.inches(0.04)
        T.bar(slide, x=left, y=y, w=width, h=bar_h, colour=colour)
        fit = (width - T.inches(0.1)) / (len(row[1]) * 0.6 * 12700)
        T.box(
            slide,
            row[1],
            x=left,
            y=y,
            w=width,
            h=bar_h,
            size=min(14.0, fit),
            colour=T.on_fill(colour),
            bold=True,
            align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE,
        )
    scale = y + bar_h + T.inches(0.05)
    T.box(
        slide,
        "0 min",
        x=x,
        y=scale,
        w=T.inches(2),
        h=T.inches(0.25),
        size=11,
        colour=T.P.faint,
    )
    T.box(
        slide,
        f"{total:g} min",
        x=x + w - T.inches(2),
        y=scale,
        w=T.inches(2),
        h=T.inches(0.25),
        size=11,
        colour=T.P.faint,
        align=PP_ALIGN.RIGHT,
    )
    gutter, pad = T.inches(0.75), T.inches(0.25)
    shares = widths or [1.0] * len(columns)
    span = w - gutter - pad
    lefts = [
        x + gutter + round(span * sum(shares[:index]) / sum(shares))
        for index in range(len(columns) + 1)
    ]
    last = len(columns) - 1
    command = all(
        row[last] == "-" or row[last].startswith(COMMAND) for row in rows
    )
    header = y + T.inches(0.85)
    for index, heading in enumerate(columns):
        T.box(
            slide,
            heading.upper(),
            x=lefts[index],
            y=header,
            w=lefts[index + 1] - lefts[index] - T.inches(0.12),
            h=T.inches(0.3),
            size=12.5,
            colour=T.P.faint,
            bold=True,
            align=PP_ALIGN.RIGHT
            if command and index == last
            else PP_ALIGN.LEFT,
        )
    first = header + T.inches(0.4)
    pitch = (y + h - first) // len(rows)
    for row_index, (row, colour) in enumerate(zip(rows, colours, strict=True)):
        top = first + row_index * pitch
        card_h = pitch - T.inches(0.07)
        T.panel(slide, x=x, y=top, w=w, h=card_h, fill=T.P.card)
        T.bar(
            slide,
            x=x + T.inches(0.25),
            y=top + (card_h - T.inches(0.28)) // 2,
            w=T.inches(0.28),
            h=T.inches(0.28),
            colour=colour,
        )
        for index, cell in enumerate(row):
            mono = command and index == last
            T.box(
                slide,
                cell,
                x=lefts[index],
                y=top,
                w=lefts[index + 1] - lefts[index] - T.inches(0.12),
                h=card_h,
                size=14,
                colour=colour if index == 1 else T.P.body,
                bold=index == 1,
                font=T.FONT_CODE if mono else T.FONT_UI,
                align=PP_ALIGN.RIGHT if mono else PP_ALIGN.LEFT,
                anchor=MSO_ANCHOR.MIDDLE,
            )


def compare(
    slide: Slide, block: dict[str, Any], *, x: int, y: int, w: int, h: int
) -> None:
    """Draw the sibling's two comparison cards with accent top bars."""
    gap = T.inches(0.22)
    width = (w - gap) // 2
    for index, side in enumerate((block["left"], block["right"])):
        left = x + index * (width + gap)
        accent = (T.P.blue, T.P.teal)[index]
        title = str(side["title"])
        # A long title in a narrow rail card shrinks rather than breaks.
        fit = (width - 2 * gap) / (len(title) * 0.6 * 12700)
        T.panel(slide, x=left, y=y, w=width, h=h, fill=T.P.card)
        T.bar(slide, x=left, y=y, w=width, h=T.inches(0.075), colour=accent)
        T.box(
            slide,
            title,
            x=left + gap,
            y=y + gap,
            w=width - 2 * gap,
            h=T.inches(0.48),
            size=min(18.0, fit),
            bold=True,
            colour=T.P.ink_soft,
        )
        T.rich(
            slide,
            item_runs(side["items"], numbered=False, marker=accent),
            x=left + gap,
            y=y + T.inches(0.85),
            w=width - 2 * gap,
            h=h - T.inches(1.0),
            size=14,
        )


def links(
    slide: Slide,
    groups: list[dict[str, Any]],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw one accent-topped card of clickable doc links per group."""
    gap = T.inches(0.3)
    pad = T.inches(0.32)
    width = (w - gap * (len(groups) - 1)) // len(groups)
    for index, group in enumerate(groups):
        left = x + index * (width + gap)
        accent = LINK_ACCENTS[index % len(LINK_ACCENTS)]
        T.panel(slide, x=left, y=y, w=width, h=h, fill=T.P.card)
        T.bar(slide, x=left, y=y, w=width, h=T.inches(0.075), colour=accent)
        T.box(
            slide,
            str(group["title"]),
            x=left + pad,
            y=y + T.inches(0.28),
            w=width - 2 * pad,
            h=T.inches(0.45),
            size=16,
            bold=True,
            colour=T.P.ink,
            name="links-title",
        )
        T.rich(
            slide,
            [
                [
                    (f"{LINK_MARK}  ", {"colour": accent, "bold": True}),
                    (str(item["label"]), {"link": str(item["url"])}),
                ]
                for item in group["items"]
            ],
            x=left + pad,
            y=y + T.inches(0.88),
            w=width - 2 * pad,
            h=h - T.inches(1.08),
            size=15,
            colour=T.P.blue,
            name="links",
            space_after=7,
        )


def metrics(
    slide: Slide,
    items: list[dict[str, Any]],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw the sibling's large-number cards with readable labels."""
    gap = T.inches(0.18)
    width = (w - gap * (len(items) - 1)) // len(items)
    for index, item in enumerate(items):
        left = x + index * (width + gap)
        accent = T.ACCENTS[index % len(T.ACCENTS)]
        T.panel(slide, x=left, y=y, w=width, h=h, fill=T.P.card)
        T.bar(slide, x=left, y=y, w=width, h=T.inches(0.075), colour=accent)
        T.box(
            slide,
            str(item["value"]),
            x=left + gap,
            y=y + gap,
            w=width - 2 * gap,
            h=h * 45 // 100,
            size=32,
            colour=accent,
            bold=True,
        )
        T.box(
            slide,
            str(item["label"]),
            x=left + gap,
            y=y + h * 60 // 100,
            w=width - 2 * gap,
            h=h * 30 // 100,
            size=14,
        )


def lineage(
    slide: Slide,
    rows: list[dict[str, Any]],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw a name, its successor when it has one, and a status chip."""
    row_h = h // len(rows)
    for index, row in enumerate(rows):
        top = y + index * row_h
        status = str(row["status"])
        colour = T.STATUS_COLOURS[status]
        T.panel(
            slide,
            x=x,
            y=top,
            w=w,
            h=row_h - T.inches(0.12),
            fill=T.P.card,
        )
        T.bar(
            slide,
            x=x,
            y=top,
            w=T.inches(0.08),
            h=row_h - T.inches(0.12),
            colour=colour,
        )
        T.box(
            slide,
            str(row["before"]),
            x=x + T.inches(0.2),
            y=top + T.inches(0.1),
            w=w * 32 // 100 - T.inches(0.05),
            h=row_h - T.inches(0.3),
            size=15,
            colour=T.P.ink_soft,
            bold=True,
        )
        after = str(row["after"]).strip()
        if after not in {"", "-", "none"}:
            T.rich(
                slide,
                [
                    [
                        (
                            f"→ {row['relation']} →",
                            {"colour": T.P.blue, "size": 12},
                        )
                    ],
                    [(after, {})],
                ],
                x=x + w * 35 // 100,
                y=top + T.inches(0.1),
                w=w * 37 // 100,
                h=row_h - T.inches(0.3),
                size=14,
                colour=T.P.ink_soft,
                bold=True,
            )
        T.rich(
            slide,
            [
                [(status, {"colour": colour})],
                [(str(row["dates"]), {"bold": False})],
            ],
            x=x + w * 75 // 100,
            y=top + T.inches(0.1),
            w=w * 23 // 100,
            h=row_h - T.inches(0.3),
            size=12,
            colour=T.P.muted,
            bold=True,
        )


def timeline(
    slide: Slide,
    events: list[dict[str, Any]],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Lay out dated milestones in two chronological reading columns."""
    count = (len(events) + 1) // 2
    row_h = h // max(count, 1)
    for index, event in enumerate(events):
        column, row = divmod(index, count)
        left = x + column * w // 2
        top = y + row * row_h
        accent = (T.P.blue, T.P.teal)[column % 2]
        T.bar(
            slide,
            x=left,
            y=top,
            w=T.inches(0.08),
            h=row_h - T.inches(0.1),
            colour=accent,
        )
        T.box(
            slide,
            str(event["date"]),
            x=left + T.inches(0.22),
            y=top,
            w=T.inches(1.35),
            h=row_h,
            size=13,
            colour=accent,
            bold=True,
        )
        T.box(
            slide,
            str(event["label"]),
            x=left + T.inches(1.65),
            y=top,
            w=w // 2 - T.inches(1.85),
            h=row_h,
            size=15,
            colour=T.P.ink_soft,
        )


def layer_map(
    slide: Slide,
    layers: list[dict[str, Any]],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Show platform layers as labelled rows of status-filled tiles."""
    row_h = h // len(layers)
    for index, layer in enumerate(layers):
        top = y + index * row_h
        label_w = w // 5
        T.box(
            slide,
            str(layer["name"]),
            x=x,
            y=top,
            w=label_w - T.inches(0.2),
            h=row_h - T.inches(0.13),
            size=16,
            colour=T.P.blue,
            bold=True,
            anchor=MSO_ANCHOR.MIDDLE,
        )
        items = layer["items"]
        item_w = (w - label_w) // len(items)
        for item_index, item in enumerate(items):
            left = x + label_w + item_index * item_w
            colour = T.STATUS_COLOURS.get(item.get("status"), T.P.blue)
            T.panel(
                slide,
                x=left,
                y=top,
                w=item_w - T.inches(0.1),
                h=row_h - T.inches(0.13),
                fill=colour,
                line=None,
            )
            T.box(
                slide,
                str(item["label"]),
                x=left + T.inches(0.1),
                y=top + T.inches(0.13),
                w=item_w - T.inches(0.3),
                h=row_h - T.inches(0.4),
                size=13,
                colour=T.on_fill(colour),
                align=PP_ALIGN.CENTER,
                anchor=MSO_ANCHOR.MIDDLE,
            )


def prose(
    slide: Slide,
    block: dict[str, Any],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    dark: bool,
    accent: str,
) -> None:
    """Draw text, notes, callouts, bullets and steps, light or on navy.

    In the rail they sit on the sibling's navy panel; on the light
    body a callout is the sibling's accent bar beside bold text.
    """
    kind = str(block["type"])
    size = block.get("size") or (11 if kind == "source_note" else 14)
    tone = TONES.get(str(block.get("tone")), accent)
    colour, marker = T.P.body, accent
    bold = bool(block.get("bold"))
    anchor = MSO_ANCHOR.TOP
    if dark and kind in RAIL_PROSE:
        T.panel(slide, x=x, y=y, w=w, h=h, fill=T.P.panel_deep, line=None)
        if kind == "callout":
            T.bar(slide, x=x, y=y, w=w, h=T.inches(0.075), colour=tone)
        x, y, w, h = (
            x + T.inches(0.15),
            y + T.inches(0.13),
            w - T.inches(0.3),
            h - T.inches(0.26),
        )
        colour, marker = T.P.on_dark_soft, T.P.teal
    elif kind == "callout":
        # The bar grows with the sentence, as in the siblings, not with
        # the space the author allocated below it.
        lines = wrapped(str(block["text"]), w - T.inches(0.25), size)
        h = min(h, round(lines * size * 1.15 * 12700) + T.inches(0.2))
        T.bar(slide, x=x, y=y, w=T.inches(0.075), h=h, colour=tone)
        x, w = x + T.inches(0.25), w - T.inches(0.25)
        colour, bold, anchor = T.P.ink_soft, True, MSO_ANCHOR.MIDDLE
    elif kind == "source_note":
        colour = T.P.muted
    paragraphs = (
        item_runs(block["items"], numbered=kind == "steps", marker=marker)
        if kind in {"bullets", "steps"}
        else [[(line, {})] for line in str(block["text"]).split("\n")]
    )
    T.rich(
        slide,
        paragraphs,
        x=x,
        y=y,
        w=w,
        h=h,
        size=size,
        colour=colour,
        bold=bold,
        anchor=anchor,
    )


def draw(
    slide: Slide,
    block: dict[str, Any],
    *,
    x: int,
    y: int,
    w: int,
    root: Path,
    dark: bool = False,
    accent: str = T.P.coral,
    schedule: dict[str, str] | None = None,
) -> int:
    """Render one validated block and return its allocated height.

    ``dark`` puts prose on the rail's navy panel, ``accent`` is the
    section colour for markers and ``schedule`` colours agenda rows.
    """
    kind = str(block["type"])
    h = T.inches(block.get("h") or DEFAULT_HEIGHTS[kind])
    area: Area = {"x": x, "y": y, "w": w, "h": h}
    if kind in {"text", "source_note", "callout", "bullets", "steps"}:
        prose(slide, block, **area, dark=dark, accent=accent)
    elif kind in {"table", "matrix"}:
        columns, rows = list(block["columns"]), list(block["rows"])
        if is_schedule(columns, rows):
            agenda(
                slide,
                columns,
                rows,
                **area,
                widths=block.get("widths"),
                schedule=schedule,
            )
        else:
            table(
                slide,
                columns,
                rows,
                **area,
                widths=block.get("widths"),
                chips=kind == "matrix",
            )
    elif kind == "compare":
        compare(slide, block, **area)
    elif kind == "metrics":
        metrics(slide, block["items"], **area)
    elif kind == "links":
        links(slide, block["groups"], **area)
    elif kind == "lineage":
        lineage(slide, block["rows"], **area)
    elif kind == "timeline":
        timeline(slide, block["events"], **area)
    elif kind == "layer_map":
        layer_map(slide, block["layers"], **area)
    elif kind == "code":
        caption = str(block.get("caption") or "")
        caption_h = T.inches(0.36) if caption.strip() else 0
        panel_h = h - caption_h
        panel = T.panel(
            slide, x=x, y=y, w=w, h=panel_h, fill=T.P.panel, line=T.P.panel
        )
        panel.name = "code-panel"
        pad_x, pad_y = T.inches(0.2), T.inches(0.12)
        T.rich(
            slide,
            code_lines(block["code"]),
            x=x + pad_x,
            y=y + pad_y,
            w=w - 2 * pad_x,
            h=panel_h - 2 * pad_y,
            size=13.5,
            colour=T.P.code_fg,
            font=T.FONT_CODE,
            name="code",
        )
        if caption_h:
            T.box(
                slide,
                caption,
                x=x + pad_x,
                y=y + panel_h,
                w=w - 2 * pad_x,
                h=caption_h,
                size=10,
                colour=T.P.muted,
                name="code-caption",
            )
    elif kind == "image":
        image(slide, root / block["path"], block.get("caption") or "", **area)
    else:
        from deck.diagrams import diagram

        diagram(slide, block, **area)
    return h


def image(
    slide: Slide,
    path: Path,
    caption: str,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Contain an image in its panel, preserving its aspect ratio."""
    from pptx.parts.image import Image
    from pptx.util import Emu

    source = Image.from_file(str(path))
    width, height = source.size
    scale = min(w / width, (h - T.inches(0.4)) / height)
    rendered_w, rendered_h = round(width * scale), round(height * scale)
    slide.shapes.add_picture(
        str(path),
        Emu(x + (w - rendered_w) // 2),
        Emu(y),
        Emu(rendered_w),
        Emu(rendered_h),
    )
    T.box(
        slide,
        caption,
        x=x,
        y=y + h - T.inches(0.3),
        w=w,
        h=T.inches(0.3),
        size=11,
        colour=T.P.muted,
    )
