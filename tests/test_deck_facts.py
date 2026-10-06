from __future__ import annotations

import copy
import io
import json
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from awsai_demo.contracts import error, evidence, operation, result
from awsai_demo.runtime import Settings
from deck import collect_facts as collector
from deck.facts import (
    FactDefinition,
    FactError,
    empty_document,
    load_contract,
    load_facts,
    measurement_fingerprint,
    valid_value,
)

ROOT = Path(__file__).resolve().parents[1]


def contract_root(tmp_path: Path, contract: object | None = None) -> Path:
    (tmp_path / "deck").mkdir(exist_ok=True)
    raw = (
        contract
        if contract is not None
        else {
            "schema_version": 1,
            "offline": {
                "demo_count": {
                    "source": "list",
                    "how": "count",
                    "type": "integer",
                }
            },
            "live": {},
        }
    )
    (tmp_path / "deck/facts_contract.yaml").write_text(
        yaml.safe_dump(raw), encoding="utf-8"
    )
    return tmp_path


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"schema_version": 2, "offline": {}, "live": {}},
        {"schema_version": 1, "offline": [], "live": {}},
        {"schema_version": 1, "offline": {"x": {}}, "live": {}},
        {"schema_version": 1, "offline": {4: {}}, "live": {}},
        {"schema_version": 1, "offline": {"x": "bad"}, "live": {}},
        {
            "schema_version": 1,
            "offline": {"x": {"source": 3, "how": "x", "type": "integer"}},
            "live": {},
        },
        {
            "schema_version": 1,
            "offline": {"x": {"source": "list", "how": "x", "type": "object"}},
            "live": {},
        },
    ],
)
def test_contract_rejects_invalid(tmp_path: Path, raw: object) -> None:
    with pytest.raises(FactError):
        load_contract(contract_root(tmp_path, raw))


def test_contract_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "facts_contract.yaml").write_text(
        "schema_version: 1\n"
        "offline:\n"
        "  demo_count: {source: list, how: count, type: integer}\n"
        "  demo_count: {source: list, how: again, type: integer}\n"
        "live: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_contract(tmp_path)


def test_fingerprint_ignores_editorial_and_bytecode(tmp_path: Path) -> None:
    root = contract_root(tmp_path)
    (root / "src/__pycache__").mkdir(parents=True)
    (root / "src/example.py").write_text("x = 1", encoding="utf-8")
    first = measurement_fingerprint(root)
    (root / "src/__pycache__/x.pyc").write_bytes(b"compiled")
    (root / "src/ignored.pyc").write_bytes(b"compiled")
    (root / "deck/speaker_notes.md").write_text("new text", encoding="utf-8")
    assert measurement_fingerprint(root) == first
    (root / "src/example.py").write_text("x = 2", encoding="utf-8")
    assert measurement_fingerprint(root) != first


@pytest.mark.parametrize(
    ("value", "kind", "valid"),
    [
        (True, "integer", False),
        (3, "integer", True),
        (True, "boolean", True),
        (3, "boolean", False),
        ("x", "string", True),
        (None, "string", False),
        ([[1, 2.3, True, "x"]], "table", True),
        ([[None]], "table", False),
        ([1], "table", False),
        (None, "table", False),
        (25740, "identifier", True),
        ("25740", "identifier", True),
        (True, "identifier", False),
        (1.5, "identifier", False),
        (None, "identifier", False),
    ],
)
def test_value_types(value: object, kind: str, *, valid: bool) -> None:
    assert valid_value(value, kind) is valid


def test_identifier_contract_loads(tmp_path: Path) -> None:
    root = contract_root(
        tmp_path,
        {
            "schema_version": 1,
            "offline": {
                "decision.pause_pid": {
                    "source": "decision --simulate-approval",
                    "how": "pid of the process that paused",
                    "type": "identifier",
                }
            },
            "live": {},
        },
    )
    loaded = load_contract(root)["offline"]["decision.pause_pid"]
    assert loaded.type == "identifier"


def test_fact_store_missing_stale_type_and_unknown(tmp_path: Path) -> None:
    root = contract_root(tmp_path)
    path = root / "deck/facts.json"
    store = load_facts(path, root=root)
    with pytest.raises(FactError, match="not been collected"):
        store.resolve("offline.demo_count")
    assert "MISSING" in str(store.resolve("offline.demo_count", draft=True))
    store.resolve("offline.demo_count", draft=True)
    assert len(store.problems) == 1
    for key in ("unknown.x", "offline.unknown"):
        with pytest.raises(FactError, match="outside"):
            store.resolve(key)
    store.document["metadata"]["offline"]["missing"] = {}
    with pytest.raises(FactError, match="Stale"):
        store.resolve("offline.demo_count")
    store.document["metadata"]["offline"]["fingerprint"] = store.fingerprint
    with pytest.raises(FactError, match="wrong type"):
        store.resolve("offline.demo_count")
    store.document["offline"]["demo_count"] = 30
    assert store.resolve("facts.offline.demo_count") == 30
    path.write_text(json.dumps(store.document), encoding="utf-8")
    assert load_facts(path, root=root).resolve("offline.demo_count") == 30


@pytest.mark.parametrize(
    "change",
    [
        "list",
        "version",
        "extra",
        "lane_list",
        "lane_extra",
        "metadata_list",
        "metadata_extra",
        "metadata_lane_list",
        "metadata_lane_extra",
        "missing_list",
        "sources_list",
    ],
)
def test_reject_bad_snapshot(tmp_path: Path, change: str) -> None:
    root = contract_root(tmp_path)
    document: Any = empty_document(load_contract(root))
    if change == "list":
        document = []
    elif change == "version":
        document["schema_version"] = 3
    elif change == "extra":
        document["extra"] = 1
    elif change == "lane_list":
        document["offline"] = []
    elif change == "lane_extra":
        document["offline"]["extra"] = 1
    elif change == "metadata_list":
        document["metadata"] = []
    elif change == "metadata_extra":
        document["metadata"]["extra"] = 1
    elif change == "metadata_lane_list":
        document["metadata"]["offline"] = []
    elif change == "metadata_lane_extra":
        document["metadata"]["offline"]["extra"] = 1
    elif change == "missing_list":
        document["metadata"]["offline"]["missing"] = []
    else:
        document["metadata"]["offline"]["sources"] = []
    path = root / "facts.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FactError):
        load_facts(path, root=root)


def fake_run(args: Any, settings: Settings) -> Any:
    command = args.command
    data: Any = {
        "scenario": dict.fromkeys(
            (
                "budget",
                "staffing_total",
                "platform_total",
                "total",
                "headroom",
            ),
            123,
        ),
        "pause_pid": 1,
        "resume_pid": 2,
        "files_read": ["a", "b"],
        "effects_after_resume": 1,
        "effects_after_replay": 1,
        "approval_source": "simulated",
        "responsibility_matrix": [["a", "b"]],
        "lanes": [["A", "contract", "local_contract", "ok"]],
        "injection_retrieved": True,
        "injection_obeyed": False,
        "answerable_citations": "section-a",
        "total": 3,
        "passed": 2,
        "failed_case": "accept-in-pricing-turn",
        "span_names": ["a", "b"],
        "denied_reason": "approval required",
        "tool_names": ["a", "b"],
        "runtime": {"ping_status": "Healthy"},
        "tiers": [
            {
                "tier": "standard",
                "routing": "in-region",
                "usd_per_successful_run": "0.01",
            }
        ],
        "read_at": "2026-10-05",
        "advertised": [
            {
                "service": "Bedrock",
                "plan": "Ultimate",
                "operations": ["Converse"],
            }
        ],
        "packages": dict.fromkeys(
            ("boto3", "strands_agents", "bedrock_agentcore", "a2a_sdk", "mcp"),
            "1.0",
        ),
        "region": "us-east-1",
        "model_count": 120,
        "active_count": 100,
        "legacy_count": 20,
        "provider_count": 5,
        "global_profiles": 12,
        "model": "tested-model",
        "authenticated": True,
        "principal_type": "user",
        "bedrock_probe": "blocked"
        if settings.aws_profile == "aila-sandbox"
        else "ok",
        "bedrock_probe_code": "authorization_denied",
        "guardrail": {"input_action": "NONE", "output_action": "INTERVENED"},
        "list_indices_code": "SubscriptionRequiredException",
        "agents": {"classic_listed": 0},
    }
    if command == "lineage":
        data = [
            {"normalized_status": status}
            for status in ("maintenance", "sunset", "active")
        ]
    if command == "list":
        data = [1] * 30
    execution = getattr(args, "execution", "offline")
    operations = []
    if execution == "live":
        operations = [
            operation(
                service="aws",
                operation=name,
                mode="live_service",
                execution_target="aws",
                transport="aws",
                endpoint_url="https://bedrock.us-east-1.amazonaws.com",
                http_status=200,
                usage={"input_tokens": 30, "output_tokens": 20},
            )
            for name in (
                "ListFoundationModels",
                "ListInferenceProfiles",
                "Converse",
                "GetCallerIdentity",
                "ApplyGuardrail",
                "ListIndices",
                "ListAgents",
            )
        ]
    return result(
        demo=command,
        technology="test",
        lane="operations",
        lifecycle_refs=["bedrock"],
        requested_execution=execution,
        headline="test",
        operations=operations,
        data=data,
        evidence=evidence(
            sdk_invoked=bool(operations),
            network_attempted=bool(operations),
            aws_executed=bool(operations),
            estimated_cost_usd=0.03,
            cost_basis="estimated",
        ),
    )


def test_collect_both_lanes_preserves_other_lane(tmp_path: Path) -> None:
    output = tmp_path / "facts.json"
    calls: list[tuple[str, str | None, bool]] = []

    def run(args: Any, settings: Settings) -> Any:
        calls.append(
            (args.command, settings.aws_profile, settings.allow_create)
        )
        return fake_run(args, settings)

    offline = collector.collect_facts(root=ROOT, output=output, runner=run)
    assert offline["metadata"]["offline"]["missing"] == {}
    assert len(calls) == len(
        {d.source for d in load_contract(ROOT)["offline"].values()}
    )
    assert offline["offline"]["decision.files_read"] == 2
    assert offline["offline"]["scenario.total"] == 123
    before = copy.deepcopy(offline["offline"])
    calls.clear()
    live = collector.collect_facts(
        root=ROOT,
        output=output,
        execution="live",
        runner=run,
        profile="test-user",
        settings=Settings(),
        clock=lambda: datetime(2026, 10, 5, tzinfo=UTC),
    )
    assert live["offline"] == before
    assert live["metadata"]["live"]["missing"] == {}
    assert ("aws-identity", "aila-sandbox", False) in calls
    assert ("aws-identity", "test-user", False) in calls
    assert all(not create for _, _, create in calls)
    assert live["live"]["converse.estimated_usd"] == "0.030000"
    live_before = copy.deepcopy(live["live"])
    recollected = collector.collect_facts(root=ROOT, output=output, runner=run)
    assert recollected["live"] == live_before
    assert load_facts(output, root=ROOT).resolve("offline.demo_count") == 30


def test_missing_values_are_reasoned_and_sanitized(tmp_path: Path) -> None:
    def run(args: Any, settings: Settings) -> Any:
        if args.command == "decision":
            message = "account 123456789012 failed"
            raise RuntimeError(message)
        value = fake_run(args, settings)
        if args.command == "cost":
            value["data"]["tiers"][0]["usd_per_successful_run"] = None
        if args.command == "doctor":
            value["data"]["packages"]["mcp"] = None
        if args.command == "evaluation":
            value["status"] = "error"
            value["error"] = error("validation_failed", "case failure")
        return value

    value = collector.collect_facts(
        root=ROOT, output=tmp_path / "facts.json", runner=run
    )
    assert value["offline"]["scenario.total"] is None
    assert "123456789012" not in json.dumps(value)
    reasons = value["metadata"]["offline"]["missing"]
    assert "null" in reasons["cost.tiers"]
    assert "no value" in reasons["packages.mcp"]
    assert "case failure" in reasons["evaluation.total"]


def test_failed_live_call_is_not_zero_measurement() -> None:
    args = collector.source_arguments(
        "model-lifecycle --execution live",
        execution="live",
        allow_create=False,
    )
    value = fake_run(args, Settings())
    value["operations"] = []
    definition = FactDefinition(
        "model-lifecycle --execution live", "count", "integer"
    )
    with pytest.raises(FactError, match="No successful AWS"):
        collector.extract_fact(
            "live", "catalog.model_count", definition, value
        )
    value["status"] = "error"
    with pytest.raises(FactError, match="Source status"):
        collector.extract_fact(
            "live", "catalog.model_count", definition, value
        )
    value["status"] = "blocked"
    value["mode"] = "not_run"
    with pytest.raises(FactError, match="Source status"):
        collector.extract_fact(
            "live", "catalog.model_count", definition, value
        )
    value["status"] = "error"
    value["operations"] = [
        operation(
            service="fixture",
            operation="test",
            fixture_id="failure",
            status="blocked",
            error_code="missing_configuration",
        )
    ]
    with pytest.raises(FactError, match="missing_configuration"):
        collector.extract_fact(
            "live", "catalog.model_count", definition, value
        )


def test_child_operations_and_explicit_creation() -> None:
    args = collector.source_arguments(
        "guardrails --execution live --allow-create",
        execution="live",
        allow_create=True,
    )
    assert args.allow_create
    value = fake_run(args, Settings())
    parent = result(
        demo="batch",
        technology="test",
        lane="operations",
        lifecycle_refs=["bedrock"],
        requested_execution="live",
        headline="test",
        children=[value],
    )
    assert collector._answered(parent, "ApplyGuardrail")["status"] == "ok"
    with pytest.raises(FactError, match="does not match"):
        collector.source_arguments(
            "decision", execution="live", allow_create=False
        )


def test_cli_success_error_and_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = contract_root(tmp_path)
    monkeypatch.chdir(root)
    out, err = io.StringIO(), io.StringIO()
    assert (
        collector.main(
            ["--output", "facts.json"],
            runner=fake_run,
            settings_loader=Settings,
            stdout=out,
            stderr=err,
        )
        == 0
    )
    assert "0 missing" in out.getvalue()
    assert collector.main(["--unknown"], stdout=out, stderr=err) == 1
    assert "unrecognized" in err.getvalue()
    monkeypatch.setattr(sys, "argv", ["collect_facts.py", "--unknown"])
    monkeypatch.delitem(sys.modules, "deck.collect_facts")
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("deck.collect_facts", run_name="__main__")
    assert stopped.value.code == 1
    assert "unrecognized" in capsys.readouterr().err
