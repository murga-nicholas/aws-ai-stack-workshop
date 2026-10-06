from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import pytest
from botocore.exceptions import ClientError

from awsai_demo.agentcore_memory_demo import (
    _STRANDS_INTEGRATION,
    _mark_memory_not_active,
    _missing_live_port,
    _verify_for_child,
    create_event_params,
    create_memory_params,
    memory_event_charges,
    retrieve_memory_records_params,
    run_agentcore_memory_demo,
)
from awsai_demo.billing import open_budget_run, use_budget_run
from awsai_demo.contracts import operation
from awsai_demo.demo_support import AwsResponse
from awsai_demo.policy import ExecutionPolicy, ReservationUnavailable
from awsai_demo.runtime import Settings
from awsai_demo.stubs import validate_operation_request

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class Prices:
    def __init__(
        self,
        *,
        fail: bool = False,
        fail_after: int | None = None,
    ) -> None:
        self.fail = fail
        self.fail_after = fail_after
        self.calls: list[tuple[str, str]] = []

    def rate(
        self,
        feature: str,
        *,
        region: str,
        unit: str,
        tier: str,
        routing: str,
    ) -> Decimal:
        del tier, routing
        self.calls.append((feature, unit))
        if self.fail or (
            self.fail_after is not None and len(self.calls) > self.fail_after
        ):
            message = "missing price"
            raise ReservationUnavailable(message)
        assert region == "us-east-1"
        return Decimal("0.000001")


class MemoryPort:
    def __init__(
        self,
        statuses: list[str] | None = None,
        *,
        auth_ok: bool = True,
        create_ok: bool = True,
        verify_ok: bool = True,
        event_ids: bool = True,
        delete_ok: bool = True,
        get_ok: bool = True,
    ) -> None:
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []
        self.name = "awsai_abcdef123456_mem"
        self.memory_id = "awsai_abcdef123456_mem_123"
        self.run_id = "abcdef123456"
        self.statuses = list(statuses or ["ACTIVE"])
        self.events: list[Mapping[str, Any]] = []
        self.auth_ok = auth_ok
        self.create_ok = create_ok
        self.verify_ok = verify_ok
        self.event_ids = event_ids
        self.delete_ok = delete_ok
        self.get_ok = get_ok

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        self.calls.append((service, operation_name, params))
        endpoint = f"https://{service}.us-east-1.amazonaws.com"
        if operation_name == "GetCallerIdentity":
            payload = {"Account": "123456789012"} if self.auth_ok else {}
            return AwsResponse(payload, endpoint)
        if operation_name == "CreateMemory":
            self.name = str(params["name"])
            self.run_id = str(params["tags"]["run-id"])
            if not self.create_ok:
                return AwsResponse({"memory": {}}, endpoint)
            return AwsResponse(
                {"memory": self._memory_payload("CREATING")}, endpoint
            )
        if operation_name == "GetMemory":
            if not self.get_ok:
                raise ClientError(
                    {
                        "Error": {"Code": "AccessDeniedException"},
                        "ResponseMetadata": {"HTTPStatusCode": 403},
                    },
                    "GetMemory",
                )
            status = self.statuses.pop(0) if self.statuses else "ACTIVE"
            return AwsResponse(
                {"memory": self._memory_payload(status)}, endpoint
            )
        if operation_name == "ListMemories":
            return AwsResponse({"memories": []}, endpoint)
        if operation_name == "ListTagsForResource":
            run_id = self.run_id if self.verify_ok else "other"
            return AwsResponse({"tags": {"run-id": run_id}}, endpoint)
        if operation_name == "CreateEvent":
            event = self._event_payload(params, len(self.events) + 1)
            if not self.event_ids:
                event.pop("eventId", None)
            self.events.append(event)
            return AwsResponse({"event": event}, endpoint)
        if operation_name == "ListEvents":
            return AwsResponse({"events": list(self.events)}, endpoint)
        if operation_name == "DeleteMemory":
            if not self.delete_ok:
                raise ClientError(
                    {
                        "Error": {"Code": "AccessDeniedException"},
                        "ResponseMetadata": {"HTTPStatusCode": 403},
                    },
                    "DeleteMemory",
                )
            return AwsResponse(
                {"memoryId": self.memory_id, "status": "DELETING"},
                endpoint,
            )
        raise AssertionError(operation_name)

    def _arn(self) -> str:
        return (
            "arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/"
            + self.memory_id
        )

    def _memory_payload(self, status: str) -> dict[str, Any]:
        return {
            "arn": self._arn(),
            "id": self.memory_id,
            "name": self.name,
            "eventExpiryDuration": 7,
            "status": status,
            "createdAt": _NOW,
            "updatedAt": _NOW,
            "strategies": [],
        }

    def _event_payload(
        self,
        params: Mapping[str, Any],
        index: int,
    ) -> dict[str, Any]:
        payload = cast("list[Mapping[str, Any]]", params["payload"])
        return {
            "memoryId": params["memoryId"],
            "actorId": params["actorId"],
            "sessionId": params["sessionId"],
            "eventId": f"event-{index:02d}",
            "eventTimestamp": params["eventTimestamp"],
            "payload": payload,
            "metadata": params.get("metadata", {}),
        }


class EmptySession:
    def client(self, service_name: str, **kwargs: Any) -> object:
        raise AssertionError((service_name, kwargs))


def test_memory_offline_fixture_contract() -> None:
    result = run_agentcore_memory_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["status"] == "ok"
    assert [item["operation"] for item in result["operations"]] == [
        "CreateMemory",
        "GetMemory",
        "CreateEvent",
        "CreateEvent",
        "ListEvents",
        "RetrieveMemoryRecords",
        "NameStrandsIntegration",
        "DeleteMemory",
    ]
    data = cast("Mapping[str, Any]", result["data"])
    assert data["status_after_get"] == "ACTIVE"
    assert data["event_count"] == 2
    assert data["strands_integration"] == _STRANDS_INTEGRATION


def test_memory_request_helpers_validate() -> None:
    validate_operation_request(
        service_name="bedrock-agentcore-control",
        operation_name="CreateMemory",
        params=create_memory_params(),
        fixture_id="agentcore-memory-create-v1",
    )
    validate_operation_request(
        service_name="bedrock-agentcore",
        operation_name="CreateEvent",
        params=create_event_params(
            role="USER",
            text="remember finance review",
            client_token="agentcore-memory-demo-token-012345-user",  # noqa: S106
        ),
        fixture_id="agentcore-memory-create-user-event-v1",
    )
    validate_operation_request(
        service_name="bedrock-agentcore",
        operation_name="RetrieveMemoryRecords",
        params=retrieve_memory_records_params(),
        fixture_id="agentcore-memory-retrieve-contract-v1",
    )
    charge = memory_event_charges(2)[0]
    assert (charge.feature, charge.quantity, charge.unit) == (
        "memory.events",
        Decimal(2),
        "event",
    )
    assert _missing_live_port()[0]["error_code"] == "missing_configuration"


def test_memory_emulator_missing_and_configured() -> None:
    missing = run_agentcore_memory_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )
    calls: list[dict[str, str]] = []

    def factory(**kwargs: str) -> EmptySession:
        calls.append(kwargs)
        return EmptySession()

    configured = run_agentcore_memory_demo(
        execution="emulator",
        settings=Settings(localstack_auth_token="token"),  # noqa: S106
        policy=ExecutionPolicy(),
        session_factory=factory,
    )

    assert missing["status"] == "blocked"
    assert {item["error_code"] for item in missing["operations"][:5]} == {
        "missing_configuration"
    }
    assert {item["error_code"] for item in configured["operations"][:5]} == {
        "not_supported_by_emulator"
    }
    assert calls and calls[0]["region_name"] == "us-east-1"


def test_memory_live_without_allow_create_is_blocked() -> None:
    result = run_agentcore_memory_demo(
        execution="live",
        settings=Settings(allow_create=False),
        policy=ExecutionPolicy(),
    )

    assert result["status"] == "blocked"
    assert result["operations"][0]["error_code"] == "create_not_allowed"
    assert result["data"] == {"allow_create": False}


def test_memory_live_pricing_refuses_before_dispatch(tmp_path: Path) -> None:
    port = MemoryPort()
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(fail=True),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_agentcore_memory_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=port,
        )

    assert result["operations"][0]["error_code"] == "budget_exceeded"
    assert port.calls == []


def test_memory_live_create_events_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_now = datetime(2026, 10, 6, tzinfo=UTC)
    port = MemoryPort(statuses=["CREATING", "ACTIVE", "ACTIVE"])
    monkeypatch.setattr(
        "awsai_demo.agentcore_memory_demo.time.sleep",
        lambda _seconds: None,
    )
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_agentcore_memory_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=port,
            event_clock=lambda: live_now,
        )

    operations = [item["operation"] for item in result["operations"]]
    data = cast("Mapping[str, Any]", result["data"])
    assert operations[:2] == ["GetCallerIdentity", "CreateMemory"]
    assert operations.count("CreateEvent") == 2
    assert operations[-1] == "DeleteMemory"
    assert data["status_after_get"] == "ACTIVE"
    assert data["event_count"] == 2
    assert data["cleanup_incomplete"] is False
    event_calls = [
        params for _service, op, params in port.calls if op == "CreateEvent"
    ]
    assert {params["eventTimestamp"] for params in event_calls} == {live_now}
    assert live_now != _NOW
    event_ops = [
        item
        for item in result["operations"]
        if item["operation"] == "CreateEvent"
    ]
    assert all(item["reserved_usd"] for item in event_ops)


@pytest.mark.parametrize(
    ("port", "key"),
    [
        (MemoryPort(auth_ok=False), "identity"),
        (MemoryPort(create_ok=False), "created"),
        (MemoryPort(verify_ok=False), "ownership_verified"),
    ],
)
def test_memory_live_create_stops_on_setup_failures(
    tmp_path: Path,
    port: MemoryPort,
    key: str,
) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_agentcore_memory_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=port,
        )

    assert key in cast("Mapping[str, Any]", result["data"])
    if key in {"created", "ownership_verified"}:
        assert result["status"] == "error"
        assert (
            cast("Mapping[str, Any]", result["data"])["cleanup_incomplete"]
            is True
        )


def test_memory_live_catches_policy_error_after_create(
    tmp_path: Path,
) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(fail_after=1),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_agentcore_memory_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=MemoryPort(statuses=["ACTIVE", "ACTIVE", "ACTIVE"]),
        )

    data = cast("Mapping[str, Any]", result["data"])
    assert data["policy_error"] == "ReservationUnavailableError"
    assert any(
        item["operation"] == "CreateEvent"
        and item["error_code"] == "budget_exceeded"
        for item in result["operations"]
    )


def test_memory_live_delete_denied_reports_cleanup_error(
    tmp_path: Path,
) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_agentcore_memory_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=MemoryPort(
                statuses=["ACTIVE", "ACTIVE", "ACTIVE"],
                delete_ok=False,
            ),
        )

    assert result["status"] == "error"
    error = result["error"]
    assert error is not None
    assert error["code"] == "cleanup_incomplete"
    assert (
        cast("Mapping[str, Any]", result["data"])["cleanup_incomplete"] is True
    )


def test_memory_live_event_without_id_leaves_child_unknown(
    tmp_path: Path,
) -> None:
    port = MemoryPort(
        statuses=["ACTIVE", "ACTIVE", "ACTIVE"],
        event_ids=False,
    )
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_agentcore_memory_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=port,
        )

    assert result["status"] == "ok"
    assert (
        cast("Mapping[str, Any]", result["data"])["cleanup_incomplete"]
        is False
    )


def test_memory_live_create_missing_default_port(tmp_path: Path) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_agentcore_memory_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            boto_port_factory=lambda **_: None,
        )

    assert result["operations"][0]["error_code"] == "missing_configuration"


def test_memory_verify_child_uses_setup_phase() -> None:
    class PhaseVerifier:
        def __init__(self) -> None:
            self.phase = ""

        def verify(
            self,
            _entry: object,
            _kind: object,
            *,
            phase: str,
        ) -> bool:
            self.phase = phase
            return True

    verifier = PhaseVerifier()
    assert _verify_for_child(cast("Any", verifier), object(), object())
    assert verifier.phase == "setup"


def test_memory_mark_not_active_ignores_unrelated_operations() -> None:
    operations = [operation(service="local", operation="Other")]
    failed_get = operation(
        service="bedrock-agentcore-control",
        operation="GetMemory",
        phase="setup",
        execution_target="aws",
        mode="live_service",
        status="blocked",
        effect="read",
        transport="aws",
        endpoint_url="https://bedrock-agentcore-control.us-east-1.amazonaws.com",
        error_code="authorization_denied",
    )

    _mark_memory_not_active(operations, "FAILED")
    _mark_memory_not_active([failed_get], "GET_FAILED")
    _mark_memory_not_active([], "FAILED")

    assert operations[0]["status"] == "ok"
    assert failed_get["error_code"] == "authorization_denied"


def test_memory_live_waits_for_active_and_skips_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = MemoryPort(statuses=["CREATING", "CREATING", "ACTIVE"])
    monkeypatch.setattr(
        "awsai_demo.agentcore_memory_demo.time.sleep",
        lambda _seconds: None,
    )
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(max_wall_seconds=2),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_agentcore_memory_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=port,
        )

    operations = [item["operation"] for item in result["operations"]]
    data = cast("Mapping[str, Any]", result["data"])
    assert "CreateEvent" not in operations
    assert data["status_after_get"] == "CREATING"
    assert data["events_created"] is False
    get_ops = [
        item
        for item in result["operations"]
        if item["operation"] == "GetMemory" and item["phase"] == "setup"
    ]
    assert get_ops[-1]["status"] == "blocked"
    assert get_ops[-1]["error_code"] == "timeout"


def test_memory_live_get_failure_stops_poll(
    tmp_path: Path,
) -> None:
    port = MemoryPort(get_ok=False)
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_agentcore_memory_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=port,
        )

    operations = [item["operation"] for item in result["operations"]]
    data = cast("Mapping[str, Any]", result["data"])
    assert "CreateEvent" not in operations
    assert data["status_after_get"] == "GET_FAILED"
    assert data["events_created"] is False
    assert result["operations"][2]["error_code"] == "authorization_denied"


def test_memory_live_failed_status_is_error(
    tmp_path: Path,
) -> None:
    port = MemoryPort(statuses=["FAILED"])
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_agentcore_memory_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=port,
        )

    data = cast("Mapping[str, Any]", result["data"])
    get_ops = [
        item
        for item in result["operations"]
        if item["operation"] == "GetMemory" and item["phase"] == "setup"
    ]
    assert result["status"] == "error"
    assert data["status_after_get"] == "FAILED"
    assert data["events_created"] is False
    assert get_ops[-1]["status"] == "error"
    assert get_ops[-1]["error_code"] == "validation_failed"
