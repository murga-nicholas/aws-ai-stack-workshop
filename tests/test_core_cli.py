from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

from awsai_demo.cli import main, render
from awsai_demo.lineage import load_lineage
from awsai_demo.registry import DEMOS
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_offline_catalogue_runs_core_without_implicit_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lineage_ids = {row.id for row in load_lineage()}
    monkeypatch.chdir(tmp_path)
    output, errors = io.StringIO(), io.StringIO()
    code = main(
        ["all", "--format", "json"],
        settings_loader=Settings,
        stdout=output,
        stderr=errors,
    )
    assert code == 0, errors.getvalue()
    assert not errors.getvalue()
    report = json.loads(output.getvalue())
    assert report["status"] == "paused"
    children = {child["demo"]: child for child in report["children"]}
    assert len(children) == 30
    assert set(children) == {spec.name for spec in DEMOS}
    for name, child in children.items():
        assert set(child["lifecycle_refs"]) <= lineage_ids
        assert child["status"] == ("paused" if name == "decision" else "ok"), (
            name,
            child,
        )
    decision = children["decision"]
    assert decision["data"]["approval_source"] == "none"
    assert decision["data"]["run_id"] in render(decision, "pretty")
    assert decision["data"]["scenario"] == {
        "budget": 25000,
        "staffing_total": 18600,
        "platform_total": 1080,
        "total": 19680,
        "headroom": 5320,
    }
    assert len(decision["data"]["lanes"]) == 3
    assert len(decision["data"]["responsibility_matrix"]) == 9
    retrieval = children["knowledge-bases"]["data"]
    assert retrieval["injection_retrieved"] is True
    assert retrieval["injection_obeyed"] is False
    assert retrieval["answerable_citations"]
    gateway = children["agentcore-gateway"]["data"]
    assert "price_pilot" in gateway["tool_names"]
    assert gateway["denied_reason"]
    assert children["agentcore-runtime"]["data"]["runtime"]["ping_status"]
    assert children["guardrails"]["data"]["input_action"]
    assert children["guardrails"]["data"]["output_action"]
