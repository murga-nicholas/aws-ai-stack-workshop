"""Glossary section 6 must match data/lineage.yaml."""

from __future__ import annotations

from pathlib import Path

from awsai_demo.lineage import data_text, parse_lineage

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "active": "ACTIVE",
    "preview": "PREVIEW",
    "maintenance": "MAINTENANCE",
    "sunset": "SUNSET",
    "legacy": "LEGACY",
    "end_of_life": "END OF LIFE",
    "renamed": "RENAMED",
    "moved": "MOVED",
}


def _glossary_date(record: object) -> str:
    """Use the status date, then launch, then end of support."""
    effective = getattr(record, "effective", None)
    launched = getattr(record, "launched", None)
    end = getattr(record, "end_of_support", None)
    found = effective or launched or end or "-"
    return str(found)


def _rows() -> list[tuple[str, str, str, str, str]]:
    text = (ROOT / "docs" / "glossary.md").read_text(encoding="utf-8")
    section = text.split("## 6. Every name, by layer", 1)[1]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] in {"Name", ""}:
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue
        name = cells[0].replace("**", "").strip()
        rows.append(
            (
                name,
                cells[2],
                cells[3],
                cells[4],
                cells[5].strip("`").strip(),
            )
        )
    return rows


def test_section_six_matches_lineage_records() -> None:
    records = {
        record.id: record
        for record in parse_lineage(data_text("lineage.yaml", ROOT / "data"))
    }
    seen: set[str] = set()
    for name, status, date, successor, record_id in _rows():
        record = records[record_id]
        seen.add(record_id)
        names = (
            ", ".join(records[item].name for item in record.successors) or "-"
        )
        assert name == record.name
        assert status == LABELS[record.normalized_status]
        assert date == _glossary_date(record)
        assert successor == names
        assert record_id == record.id
    assert seen
