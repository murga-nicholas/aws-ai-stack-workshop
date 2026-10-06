"""Write the plain-text companion for the workshop deck."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from awsai_demo.cli import CommandParser  # noqa: E402
from deck.facts import load_facts  # noqa: E402
from deck.notes import notes_time_window  # noqa: E402
from deck.schema import (  # noqa: E402
    Deck,
    Section,
    Slide,
    TableBlock,
    load_deck,
    visible_text,
)

DEFAULT_DATE = "2026-10-05"
DEFAULT_AUTHOR = "Mykola Murha"
DEFAULT_ROLE = "Data and AI Engineer  ·  DataArt"
DEFAULT_OUTPUT = ROOT / "llms-full.txt"
_PLACEHOLDER = (
    "PLACEHOLDER: the dated header belongs in deck/llms_header.md "
    "and is read when that file is present."
)


def header_text(root: Path) -> str:
    """Read the dated header, or name the file that should hold it."""
    path = root / "deck" / "llms_header.md"
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return _PLACEHOLDER


def _whole(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _clock(minutes: float) -> str:
    whole = int(minutes)
    seconds = round((minutes - whole) * 60)
    return f"{whole}:{seconds:02d}"


def _total_minutes(deck: Deck) -> float:
    ends = [
        section.minutes[1]
        for section in deck.sections
        if section.minutes is not None
    ]
    return max(ends, default=0)


def _title_slide(deck: Deck) -> Slide:
    return next(slide for slide in deck.slides if slide.kind == "title")


def presentation_glance(
    deck: Deck,
    *,
    author: str,
    role: str,
    date: str,
) -> str:
    """Summarize length, audience and how to rebuild the deck."""
    title = _title_slide(deck)
    presented = [
        slide for slide in deck.slides if slide.section.minutes is not None
    ]
    appendix = [
        slide for slide in deck.slides if slide.section.minutes is None
    ]
    length = (
        f"{_whole(_total_minutes(deck))} minutes, "
        f"{len(presented)} presented slides"
    )
    if len(appendix) == 1:
        number = deck.slides.index(appendix[0]) + 1
        length += (
            f"; slide {number} is a take-home appendix that is not presented"
        )
    elif appendix:
        start = deck.slides.index(appendix[0]) + 1
        end = deck.slides.index(appendix[-1]) + 1
        length += (
            f"; slides {start}-{end} are a take-home appendix "
            "that is not presented"
        )
    audience = ", ".join(title.audience) or "everyone"
    role_parts = ", ".join(part.strip() for part in role.split("\u00b7"))
    presenter = f"{author}, {role_parts}"
    return "\n".join(
        (
            "## Presentation at a glance",
            "",
            f"- Title: {title.title} (workshop)",
            f"- Presenter: {presenter}",
            f"- Date: {date}",
            f"- Length: {length}",
            f"- Technologies on the cover: {title.deck}",
            f"- Audience: {audience}",
            "- Demos: every command runs offline on a laptop by default "
            "and reports its execution mode",
            "- Rebuild the deck: "
            "`uv run --group deck python deck/build_deck.py`",
        )
    )


def _agenda_table(deck: Deck) -> tuple[Slide | None, TableBlock | None]:
    slide = next(
        (item for item in deck.slides if item.id.endswith(".agenda")),
        None,
    )
    if slide is None:
        return None, None
    for block in (*slide.main, *slide.rail):
        if isinstance(block, TableBlock):
            return slide, block
    return slide, None


def _agenda_pair(table: TableBlock | None, index: int) -> tuple[str, str]:
    if table is None or index >= len(table.rows):
        return "", "-"
    row = table.rows[index]
    seen = row[2] if len(row) > 2 else ""
    command = row[3] if len(row) > 3 else "-"
    return seen, command or "-"


def agenda_section(deck: Deck) -> str:
    """List presented sections beside the agenda table, in order."""
    slide, table = _agenda_table(deck)
    number = ""
    if slide is not None:
        number = f" (slide {deck.slides.index(slide) + 1})"
    lines = [
        f"## Agenda{number}",
        "",
        "| When | Segment | What you will see | Command |",
        "|---|---|---|---|",
    ]
    presented = [
        section for section in deck.sections if section.minutes is not None
    ]
    for index, section in enumerate(presented):
        lines.append(_agenda_row(section, *_agenda_pair(table, index)))
    return "\n".join(lines)


def _agenda_row(section: Section, seen: str, command: str) -> str:
    start, end = cast("tuple[float, float]", section.minutes)
    shown = "-" if command == "-" else f"`{command}`"
    return (
        f"| {_whole(start)}-{_whole(end)} min | {section.title} "
        f"| {seen} | {shown} |"
    )


def _segment_line(slide: Slide, total: float) -> str:
    minutes = slide.section.minutes
    if minutes is None:
        return (
            f"Agenda segment: {slide.section.title} (not presented). "
            "Not presented (appendix)."
        )
    start, end = minutes
    window = notes_time_window(slide.notes)
    return (
        f"Agenda segment: {slide.section.title} "
        f"({_whole(start)}-{_whole(end)} min). "
        f"On screen {window} of {_clock(total)}."
    )


def _slide_lines(slide: Slide) -> list[str]:
    lines: list[str] = []
    for item in visible_text(slide):
        for line in item.splitlines():
            text = line.strip()
            if text:
                lines.append(f"- {text}")
    return lines


def slide_sections(deck: Deck) -> str:
    """Render every slide, its window and its speaker notes."""
    total = _total_minutes(deck)
    parts: list[str] = []
    for number, slide in enumerate(deck.slides, 1):
        parts.append(
            "\n".join(
                (
                    f"### Slide {number}. {slide.title}",
                    "",
                    _segment_line(slide, total),
                    "",
                    "On the slide:",
                    "",
                    *_slide_lines(slide),
                    "",
                    "Speaker notes:",
                    "",
                    slide.notes.strip(),
                )
            )
        )
    return "\n\n".join(parts)


def readme_section(root: Path) -> str:
    """Append the repository README, or say that it is absent."""
    path = root / "README.md"
    if path.is_file():
        body = path.read_text(encoding="utf-8").strip()
    else:
        body = "README.md is not present."
    return f"## Repository README\n\n{body}"


def render_llms(
    root: Path,
    deck: Deck,
    *,
    author: str,
    role: str,
    date: str,
) -> str:
    """Assemble the header, glance, agenda, slides and README."""
    parts = (
        header_text(root),
        presentation_glance(deck, author=author, role=role, date=date),
        agenda_section(deck),
        "## Slides",
        slide_sections(deck),
        readme_section(root),
    )
    return "\n\n".join(parts) + "\n"


def build_llms_text(
    root: Path,
    *,
    draft: bool,
    author: str = DEFAULT_AUTHOR,
    role: str = DEFAULT_ROLE,
    date: str = DEFAULT_DATE,
) -> str:
    """Load the deck and render text. Draft keeps missing facts."""
    facts = load_facts(root / "deck" / "facts.json", root=root)
    content = load_deck(
        root / "deck" / "content",
        root / "deck" / "speaker_notes.md",
        root=root,
        resolve=lambda path: facts.resolve(path, draft=draft),
        draft=draft,
        problems=facts.problems,
    )
    return render_llms(root, content, author=author, role=role, date=date)


def write_llms(text: str, output: Path) -> None:
    """Write the companion using UTF-8 and a trailing newline."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    """Write llms-full.txt from the deck sources."""
    parser = CommandParser(description=__doc__)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    try:
        args = parser.parse_args(argv)
        text = build_llms_text(
            root,
            draft=args.draft,
            author=args.author,
            role=args.role,
            date=args.date,
        )
        write_llms(text, args.output)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"Wrote {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
