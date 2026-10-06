"""Editable diagrams adapted from the sibling's shape vocabulary.

Nodes are coloured by role, as in the sibling decks: the step where a
request enters is blue, the step where it ends violet, steps between
them teal and guards amber. Stores are outlined cylinders, external
systems outlined cards, and notes plain text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Pt

from deck import theme as T

if TYPE_CHECKING:
    from pptx.slide import Slide

# kind: (fill, outline, shape, lid depth) when role sets no colour.
OUTLINED = {
    "store": (T.P.wash, T.P.teal, MSO_SHAPE.CAN, 0.15),
    "external": (T.P.card, T.P.coral, MSO_SHAPE.ROUNDED_RECTANGLE, 0.0),
}


def role(name: str, sources: set[str], targets: set[str]) -> str:
    """Colour a step by where it sits in the flow."""
    if name not in targets:
        return T.P.blue
    if name not in sources:
        return T.P.violet
    return T.P.teal


def diagram(
    slide: Slide,
    block: dict[str, Any],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Render boundaries, connectors and typed nodes on their grid."""
    del w, h  # Schema validates coordinates against the allocated box.
    nodes = {node["id"]: node for node in block["nodes"]}
    sources = {edge["from"] for edge in block["edges"]}
    targets = {edge["to"] for edge in block["edges"]}
    for boundary in block.get("boundaries", ()):
        members = [nodes[name] for name in boundary["nodes"]]
        left = min(node["x"] for node in members)
        top = min(node["y"] for node in members)
        right = max(node["x"] + node["w"] for node in members)
        bottom = max(node["y"] + node["h"] for node in members)
        shape = T.panel(
            slide,
            x=x + T.inches(left),
            y=y + T.inches(top),
            w=T.inches(right - left),
            h=T.inches(bottom - top),
            fill=T.P.wash,
            line=T.P.faint,
        )
        shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        T.box(
            slide,
            boundary["label"],
            x=x + T.inches(left + 0.1),
            y=y + T.inches(top + 0.02),
            w=T.inches(right - left - 0.2),
            h=T.inches(0.25),
            size=10,
            colour=T.P.muted,
            bold=True,
        )
    for edge in block["edges"]:
        origin, target = nodes[edge["from"]], nodes[edge["to"]]
        start_x = origin["x"] + origin["w"]
        end_x = target["x"]
        start_y = origin["y"] + origin["h"] / 2
        end_y = target["y"] + target["h"] / 2
        connector = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT,
            Emu(x + T.inches(start_x)),
            Emu(y + T.inches(start_y)),
            Emu(x + T.inches(end_x)),
            Emu(y + T.inches(end_y)),
        )
        connector.line.color.rgb = T.rgb(T.P.faint)
        connector.line.width = Pt(1.5)
        if edge.get("style") == "dashed":
            connector.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        arrow = OxmlElement("a:tailEnd")
        arrow.set("type", "triangle")
        connector.line._get_or_add_ln().append(arrow)
        if edge.get("label"):
            T.box(
                slide,
                edge["label"],
                x=x + T.inches(min(start_x, end_x)),
                y=y + T.inches(min(start_y, end_y) - 0.35),
                w=T.inches(max(abs(end_x - start_x), 1.2)),
                h=T.inches(0.3),
                size=10,
                colour=T.P.muted,
            )
    for name, node in nodes.items():
        kind = node.get("kind", "default")
        area = {
            "x": x + T.inches(node["x"]),
            "y": y + T.inches(node["y"]),
            "w": T.inches(node["w"]),
            "h": T.inches(node["h"]),
        }
        colour, lid = T.P.muted, 0
        if kind in OUTLINED:
            fill, line, form, depth = OUTLINED[kind]
            T.panel(
                slide,
                **area,
                fill=fill,
                line=line,
                kind=form,
                radius=depth or T.CARD,
                line_w=2,
            )
            colour = T.P.ink_soft
            # A store's label sits below the lid of its cylinder.
            lid = round(min(area["w"], area["h"]) * depth)
        elif kind != "note":
            fill = (
                T.P.amber if kind == "guard" else role(name, sources, targets)
            )
            T.panel(
                slide,
                **area,
                fill=fill,
                line=None,
                kind=MSO_SHAPE.ROUNDED_RECTANGLE,
            )
            colour = T.on_fill(fill)
        T.box(
            slide,
            node["label"],
            x=area["x"] + T.inches(0.12),
            y=area["y"] + lid + T.inches(0.04 if lid else 0.14),
            w=area["w"] - T.inches(0.24),
            h=area["h"] - lid - T.inches(0.08 if lid else 0.28),
            size=14,
            colour=colour,
            bold=kind != "note",
            align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE,
        )
