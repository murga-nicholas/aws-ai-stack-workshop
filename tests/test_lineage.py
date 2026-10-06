from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest
import yaml

from awsai_demo.lineage import (
    data_text,
    load_lineage,
    load_yaml,
    parse_components,
    parse_lineage,
)
from awsai_demo.registry import DEMOS


def document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "verified": "2026-10-05",
        "records": [
            {
                "id": "one",
                "name": "One",
                "former_names": [],
                "entity_type": "service",
                "scope": "all",
                "official_status": "Active",
                "normalized_status": "active",
                "launched": "2020",
                "announced": "2020-01",
                "effective": "2020-01-01",
                "end_of_support": None,
                "successors": [],
                "relation": "none",
                "sources": ["https://example.org/one"],
                "note": "A note",
            }
        ],
    }


def test_curated_records_and_component_references() -> None:
    records = load_lineage()
    assert {ref for demo in DEMOS for ref in demo.lifecycle_refs} <= {
        record.id for record in records
    }
    components = parse_components(
        data_text("components.yaml"),
        lineage_ids={r.id for r in records},
        demo_ids={demo.name for demo in DEMOS},
    )
    assert components
    assert any(not component.covered for component in components)
    assert parse_lineage(json.dumps(document()))[0].id == "one"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entity_type", "bogus"),
        ("normalized_status", "bogus"),
        ("relation", "bogus"),
        ("name", ""),
        ("name", 3),
        ("sources", []),
        ("sources", ["http://example.org"]),
        ("sources", ["https:///missing-host"]),
        ("successors", ["missing"]),
        ("former_names", "wrong"),
        ("launched", "2020-13"),
        ("launched", "20201301"),
        ("launched", "20200101"),
        ("extra", "unknown"),
    ],
)
def test_invalid_record(field: str, value: object) -> None:
    raw = document()
    rows = raw["records"]
    assert isinstance(rows, list)
    rows[0][field] = value
    with pytest.raises((ValueError, TypeError)):
        parse_lineage(json.dumps(raw))


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "{1: wrong}",
        "schema_version: 1\nschema_version: 2",
        "{}",
        '{"schema_version":2,"records":[],"verified":"2026-01-01"}',
        '{"schema_version":true,"records":[],"verified":"2026-01-01"}',
        '{"schema_version":1,"records":{},"verified":"2026-01-01"}',
    ],
)
def test_invalid_document(text: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_lineage(text)


def test_duplicate_and_missing_fields() -> None:
    raw = document()
    rows = raw["records"]
    assert isinstance(rows, list)
    rows.append(rows[0])
    with pytest.raises(ValueError, match="Duplicate record"):
        parse_lineage(json.dumps(raw))
    rows.pop()
    del rows[0]["id"]
    with pytest.raises(ValueError, match="Missing or unknown"):
        parse_lineage(json.dumps(raw))


def component_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "components": [
            {
                "id": "one",
                "name": "One",
                "layer": "models",
                "lineage": ["one"],
                "demos": ["demo"],
                "slides": ["slide"],
                "covered": True,
            }
        ],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lineage", ["missing"]),
        ("demos", ["missing"]),
        ("slides", ["missing"]),
        ("covered", "yes"),
        ("covered", False),
        ("demos", []),
        ("slides", []),
    ],
)
def test_invalid_component(field: str, value: object) -> None:
    raw = component_document()
    rows = raw["components"]
    assert isinstance(rows, list)
    rows[0][field] = value
    with pytest.raises((ValueError, TypeError)):
        parse_components(
            json.dumps(raw),
            lineage_ids={"one"},
            demo_ids={"demo"},
            slide_ids={"slide"},
        )


def test_components_complete_and_excluded() -> None:
    text = json.dumps(component_document())
    assert parse_components(
        text, lineage_ids={"one"}, demo_ids={"demo"}, slide_ids={"slide"}
    )[0].covered
    with pytest.raises(ValueError, match="Demos missing"):
        parse_components(text, lineage_ids={"one"}, demo_ids={"demo", "extra"})
    raw = component_document()
    rows = raw["components"]
    assert isinstance(rows, list)
    rows[0].update(covered=False, reason="Outside scope")
    assert not parse_components(
        json.dumps(raw), lineage_ids={"one"}, demo_ids={"demo"}
    )[0].covered


def test_yaml_safety_and_data_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(yaml.constructor.ConstructorError):
        load_yaml('!!python/object/apply:os.system ["never run"]')
    path = tmp_path / "lineage.yaml"
    path.write_text(json.dumps(document()), encoding="utf-8")
    assert load_lineage(tmp_path)[0].id == "one"
    with pytest.raises(ValueError, match="Unknown curated"):
        data_text("../secret")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "lineage.yaml").write_text(
        "packaged", encoding="utf-8"
    )
    monkeypatch.setattr(Path, "is_file", lambda _: False)

    def resource_root(_: str) -> Path:
        return tmp_path

    monkeypatch.setattr(resources, "files", resource_root)
    assert data_text("lineage.yaml") == "packaged"
