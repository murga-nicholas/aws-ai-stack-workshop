"""Extract bounded, balanced code viewports from tested source."""

from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_MARKER = re.compile(r"^\s*# (slide|end-slide): ([\w-]+)\s*$")


def extract_snippet(ref: str, root: Path) -> str:
    """Read a source marker confined to the checkout."""
    relative, separator, wanted = ref.partition("#")
    path = (root / relative).resolve()
    if (
        not separator
        or not wanted
        or path.suffix != ".py"
        or not path.is_relative_to((root / "src" / "awsai_demo").resolve())
    ):
        message = f"Invalid source marker reference: {ref}"
        raise ValueError(message)
    snippets: dict[str, str] = {}
    opened: str | None = None
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _MARKER.fullmatch(line)
        if match is None:
            if opened is not None:
                lines.append(line)
            continue
        kind, name = match.group(1), match.group(2)
        if kind == "slide":
            if opened is not None or name in snippets:
                message = f"Nested or duplicate code marker: {name}"
                raise ValueError(message)
            opened, lines = name, []
        else:
            if name != opened:
                message = f"Unbalanced code marker: {name}"
                raise ValueError(message)
            snippets[match.group(2)] = textwrap.dedent("\n".join(lines))
            opened = None
    if opened is not None:
        message = f"Unclosed code marker: {opened}"
        raise ValueError(message)
    if wanted not in snippets:
        message = f"Missing code marker: {ref}"
        raise ValueError(message)
    snippet = snippets[wanted]
    if not snippet.strip() or len(snippet.splitlines()) > 14:
        message = f"Code marker must contain 1-14 lines: {ref}"
        raise ValueError(message)
    if any(len(line) > 72 for line in snippet.splitlines()):
        message = f"Code marker exceeds 72 columns: {ref}"
        raise ValueError(message)
    return snippet
