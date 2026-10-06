"""Load keyed presenter scripts and attach them to PowerPoint slides."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pptx.slide import Slide

_MARKER = re.compile(
    r"^\s*<!-- slide: ([a-z][\w-]*(?:\.[\w-]+)+) -->[ \t]*$",
    re.MULTILINE,
)


def load_notes(path: Path) -> dict[str, str]:
    """Read keyed presenter text, retaining each slide heading."""
    source = path.read_text(encoding="utf-8")
    markers = list(_MARKER.finditer(source))
    notes: dict[str, str] = {}
    for index, marker in enumerate(markers):
        key = marker.group(1)
        if key in notes:
            message = f"Duplicate notes section: {key}"
            raise ValueError(message)
        end = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(source)
        )
        body = source[marker.end() : end]
        body = re.sub(r"(?m)^[ \t]*#{1,6}(?:[ \t]+|$)", "", body)
        body = body.replace("**", "").strip()
        if not body:
            message = f"Empty notes section: {key}"
            raise ValueError(message)
        notes[key] = body
    return notes


def notes_time_window(script: str) -> str:
    """Extract the time window from the heading, or mark an appendix."""
    heading = script.split("\n", 1)[0]
    match = re.search(r"\b\d+:\d{2}\s*[-â€“]\s*\d+:\d{2}\b", heading)
    return match.group() if match else "Not presented (appendix)"


def attach_notes(slide: Slide, script: str) -> None:
    """Prepend the presenter script without losing generated notes."""
    frame = slide.notes_slide.notes_text_frame
    if frame is None:
        message = "Slide has no notes text frame"
        raise ValueError(message)
    existing = frame.text
    frame.text = f"{script}\n\n{existing}" if existing else script
