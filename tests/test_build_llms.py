"""Plain-text deck companion, including draft and error paths."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from deck.build_llms import (
    DEFAULT_ROLE,
    agenda_section,
    header_text,
    main,
    presentation_glance,
    render_llms,
    slide_sections,
)
from deck.schema import (
    AppendixSlide,
    ContentSlide,
    Deck,
    Section,
    TableBlock,
    TextBlock,
    TitleSlide,
)

ROOT = Path(__file__).resolve().parents[1]


def _missing_fact_root(root: Path) -> Path:
    """Deck whose only measured value is still uncollected."""
    deck = root / "deck"
    (deck / "content").mkdir(parents=True)
    (root / "data").mkdir()
    (deck / "facts_contract.yaml").write_text(
        "schema_version: 1\n"
        "offline:\n"
        "  demo_count: {source: list, how: count, type: integer}\n"
        "live: {}\n",
        encoding="utf-8",
    )
    lane = {
        "fingerprint": None,
        "timestamp": None,
        "sources": {},
        "missing": {},
    }
    offline = dict(lane)
    offline["missing"] = {"demo_count": "not collected"}
    (deck / "facts.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "offline": {"demo_count": None},
                "live": {},
                "metadata": {"offline": offline, "live": lane},
            }
        ),
        encoding="utf-8",
    )
    (deck / "content" / "00.yaml").write_text(
        "schema_version: 1\n"
        "section: {id: open, index: '00', title: Opening, minutes: [0, 1]}\n"
        "slides:\n"
        "  - id: open.title\n"
        "    kind: title\n"
        "    eyebrow: WORKSHOP\n"
        '    title: "Count {facts.offline.demo_count}"\n'
        "    deck: Deck\n",
        encoding="utf-8",
    )
    (deck / "speaker_notes.md").write_text(
        "<!-- slide: open.title -->\n1. Cover - 0:00-0:45\nHello\n",
        encoding="utf-8",
    )
    (root / "data" / "lineage.yaml").write_text(
        "schema_version: 1\n"
        "verified: '2026-10-05'\n"
        "records:\n"
        "  - id: bedrock\n"
        "    name: Amazon Bedrock\n"
        "    former_names: []\n"
        "    entity_type: service\n"
        "    scope: Models\n"
        "    official_status: Available\n"
        "    normalized_status: active\n"
        "    launched: '2023-09-28'\n"
        "    announced: null\n"
        "    effective: null\n"
        "    end_of_support: null\n"
        "    successors: []\n"
        "    relation: none\n"
        "    sources: [https://docs.aws.amazon.com/bedrock/]\n"
        "    note: null\n",
        encoding="utf-8",
    )
    return root


def _section(
    identity: str,
    title: str,
    minutes: tuple[float, float] | None,
) -> Section:
    return Section(identity, "00", title, minutes)


def _title(
    section: Section,
    *,
    audience: tuple[str, ...] = ("everyone",),
) -> TitleSlide:
    return TitleSlide(
        id=f"{section.id}.title",
        kind="title",
        eyebrow="WORKSHOP",
        title="AWS AI stack",
        deck="Bedrock | AgentCore",
        section=section,
        notes="1. Cover - 0:00-0:45\nSay hello",
        audience=audience,
    )


def _content(
    section: Section,
    identity: str,
    blocks: tuple[object, ...] = (),
    *,
    kind: str = "content",
    notes: str = "2. Next - 0:45-1:30\nMore",
) -> ContentSlide:
    cls = AppendixSlide if kind == "appendix" else ContentSlide
    return cls(
        id=identity,
        kind=kind,
        eyebrow="",
        title="Useful title",
        deck="Deck",
        section=section,
        notes=notes,
        main=blocks,  # type: ignore[arg-type]
    )


def _deck(*slides: object, sections: tuple[Section, ...]) -> Deck:
    return Deck(sections, slides)  # type: ignore[arg-type]


def test_header_reads_a_file_and_names_the_placeholder(tmp_path: Path) -> None:
    assert "deck/llms_header.md" in header_text(tmp_path)
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "llms_header.md").write_text("   \n", encoding="utf-8")
    assert header_text(tmp_path) == header_text(tmp_path / "missing")
    (deck / "llms_header.md").write_text("# Dated\n", encoding="utf-8")
    assert header_text(tmp_path) == "# Dated"


def test_glance_agenda_and_slides_cover_partial_decks(tmp_path: Path) -> None:
    opening = _section("open", "Opening", (0, 0.5))
    close = _section("close", "Close", (0.5, 1.5))
    extra = _section("late", "Late", (1.5, 2))
    appendix = _section("appendix", "Appendix", None)
    title = _title(opening, audience=())
    table = TableBlock(
        type="table",
        columns=("When", "Segment", "What you will see", "Command"),
        rows=(
            ("0-1", "Opening", "Promise", "awsai-demo list"),
            ("short", "row"),
            ("a", "b", "seen", ""),
        ),
    )
    agenda = _content(
        opening,
        "open.agenda",
        (table, TextBlock(type="text", text="line one\nline two")),
    )
    bare = _content(
        close,
        "close.agenda",
        (TextBlock(type="text", text="not a table"),),
    )
    alone = _title(opening)
    deck = _deck(
        title,
        agenda,
        _content(close, "close.body"),
        _content(
            appendix,
            "appendix.one",
            kind="appendix",
            notes="Reference only",
        ),
        _content(
            appendix,
            "appendix.two",
            kind="appendix",
            notes="More reference",
        ),
        sections=(opening, close, extra, appendix),
    )
    glance = presentation_glance(
        deck, author="Mykola Murha", role=DEFAULT_ROLE, date="2026-10-05"
    )
    assert "Mykola Murha, Data and AI Engineer, DataArt" in glance
    assert "slides 4-5 are a take-home appendix" in glance
    assert "2 minutes, 3 presented slides" in glance
    text = agenda_section(deck)
    assert "0-0.5 min" in text
    assert "`awsai-demo list`" in text
    assert "| 0.5-1.5 min | Close |  | - |" in text
    assert "| 1.5-2 min | Late | seen | - |" in text
    rendered = render_llms(
        tmp_path,
        deck,
        author="Mykola Murha",
        role="Engineer",
        date="2026-10-05",
    )
    assert "README.md is not present." in rendered
    assert "Not presented (appendix)." in rendered
    assert "On screen 0:00-0:45 of 2:00." in rendered
    assert "- line one" in rendered
    assert "- line two" in rendered
    single = _deck(
        alone,
        _content(
            appendix, "appendix.only", kind="appendix", notes="Reference only"
        ),
        sections=(opening, appendix),
    )
    assert "slide 2 is a take-home appendix" in presentation_glance(
        single, author="A", role="B", date="2026-10-05"
    )
    presented_only = presentation_glance(
        _deck(alone, sections=(opening,)),
        author="A",
        role="B",
        date="2026-10-05",
    )
    assert "take-home appendix" not in presented_only
    assert agenda_section(_deck(alone, sections=(opening,))) == (
        "## Agenda\n\n"
        "| When | Segment | What you will see | Command |\n"
        "|---|---|---|---|\n"
        "| 0-0.5 min | Opening |  | - |"
    )
    no_table = agenda_section(_deck(alone, bare, sections=(opening, close)))
    assert "(slide 2)" in no_table
    assert slide_sections(_deck(alone, sections=(opening,))).startswith(
        "### Slide 1."
    )


def test_draft_build_writes_and_failures_return_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "nested" / "llms-full.txt"
    assert main(["--draft", "--output", str(output)]) == 0
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# AWS AI stack workshop: full text")
    assert "## Repository README" in text
    assert "Mykola Murha, Data and AI Engineer, DataArt" in text
    strict = tmp_path / "strict.txt"
    fixture = _missing_fact_root(tmp_path / "fixture")
    assert main(["--output", str(strict)], root=fixture) == 1
    assert "offline.demo_count: not collected" in capsys.readouterr().err
    assert not strict.exists()
    assert main(["--nope"]) == 1
    assert main(["--draft"], root=tmp_path) == 1
    monkeypatch.setattr(sys, "argv", ["build_llms.py", "--nope"])
    monkeypatch.delitem(sys.modules, "deck.build_llms", raising=False)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("deck.build_llms", run_name="__main__")
    assert stopped.value.code == 1
    assert "unrecognized" in capsys.readouterr().err
