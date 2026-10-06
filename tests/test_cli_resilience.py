from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from awsai_demo.cli import main, render
from awsai_demo.contracts import DemoResult, operation, result
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from awsai_demo.registry import DemoSpec


@pytest.mark.parametrize("failure", ["execute", "serialize", "nan"])
@pytest.mark.parametrize("execution", ["offline", "live"])
def test_all_preserves_other_children_after_a_bad_demo(
    failure: str,
    execution: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    seen = []

    def executor(
        spec: DemoSpec, args: argparse.Namespace, _: Settings
    ) -> DemoResult:
        seen.append(spec.name)
        if spec.name == "model-lifecycle" and failure == "execute":
            message = "bad account 123456789012 Bearer hidden-token"
            raise RuntimeError(message)
        data: Any = {"time": datetime(2026, 10, 6, tzinfo=UTC)}
        if spec.name == "model-lifecycle":
            data = object() if failure == "serialize" else float("nan")
        return result(
            demo=spec.name,
            technology=spec.technology,
            lane=spec.lane,
            lifecycle_refs=spec.lifecycle_refs,
            requested_execution=args.execution,
            headline="Injected local result; no AWS calls.",
            operations=[answered_operation()] if execution == "live" else [],
            data=data,
        )

    out, err = io.StringIO(), io.StringIO()
    code = main(
        [
            "all",
            "--lane",
            "models",
            "--format",
            "json",
            "--execution",
            execution,
        ],
        settings_loader=Settings,
        executor=executor,
        stdout=out,
        stderr=err,
    )
    assert code == 1
    assert not err.getvalue()
    assert "123456789012" not in out.getvalue()
    assert "hidden-token" not in out.getvalue()
    batch = json.loads(out.getvalue())
    assert len(seen) == len(batch["children"]) == 7
    assert batch["status"] == "error"
    bad = next(c for c in batch["children"] if c["demo"] == "model-lifecycle")
    assert bad["status"] == "error"
    assert bad["error"]["code"] == "validation_failed"
    assert bad["data"][
        "operation_evidence_unavailable"
        if failure == "execute"
        else "data_unavailable"
    ]
    if execution == "live" and failure != "execute":
        assert bad["operations"][0] == answered_operation()
        assert bad["mode"] == "live_service"
        assert bad["evidence"]["aws_executed"]
    good = [c for c in batch["children"] if c is not bad]
    assert all(c["status"] == "ok" for c in good)
    assert all(c["data"]["time"] == "2026-10-06T00:00:00+00:00" for c in good)


def test_renderer_accepts_sdk_timestamps_without_unknown_object_fallback() -> (
    None
):
    payload = result(
        demo="example",
        technology="SDK response",
        lane="models",
        lifecycle_refs=["bedrock"],
        requested_execution="offline",
        headline="Boundary regression.",
        data={"timestamp": datetime(2026, 10, 6, tzinfo=UTC)},
    )
    assert json.loads(render(payload, "json"))["data"]["timestamp"] == (
        "2026-10-06T00:00:00+00:00"
    )
    payload["data"] = object()
    with pytest.raises(TypeError, match="JSON serializable"):
        render(payload, "json")


def answered_operation() -> Any:
    return operation(
        service="bedrock",
        operation="ListFoundationModels",
        mode="live_service",
        execution_target="aws",
        transport="aws",
        endpoint_url="https://bedrock.us-east-1.amazonaws.com",
    )


@pytest.mark.parametrize("corruption", ["fields", "version", "child", "ops"])
def test_malformed_envelope_stays_inside_its_batch_child(
    corruption: str,
) -> None:
    def executor(spec: DemoSpec, _: argparse.Namespace, __: Settings) -> Any:
        payload = result(
            demo=spec.name,
            technology=spec.technology,
            lane=spec.lane,
            lifecycle_refs=spec.lifecycle_refs,
            requested_execution="offline",
            headline="No real network.",
        )
        if spec.name != "localstack":
            return payload
        if corruption == "fields":
            return {}
        if corruption == "version":
            payload["schema_version"] = 2
        elif corruption == "child":
            payload["mode"] = "batch"
            payload["children"] = [{}]
        else:
            payload["operations"] = [{}]
        return payload

    out, err = io.StringIO(), io.StringIO()
    assert (
        main(
            ["all", "--lane", "local", "--format", "json"],
            settings_loader=Settings,
            executor=executor,
            stdout=out,
            stderr=err,
        )
        == 1
    )
    assert not err.getvalue()
    children = json.loads(out.getvalue())["children"]
    assert [child["status"] for child in children] == ["error", "ok"]
    assert children[0]["error"]["code"] == "validation_failed"
