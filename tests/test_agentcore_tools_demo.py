from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from awsai_demo.agentcore_tools_demo import (
    INTERPRETER,
    SERVICE,
    SESSION_CHARGES,
    SESSION_ID,
    InterpreterSession,
    _read_stream,
    run_agentcore_tools_demo,
    start_request,
)
from awsai_demo.billing import open_budget_run, use_budget_run
from awsai_demo.demo_support import AwsResponse, client_method_name
from awsai_demo.owned_resources import OwnedResources
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.resource_cleanup import execute_cleanup
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from awsai_demo.billing import BudgetRun

RUN_ID = "abcdef123456"


class Port:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.bodies: list[tuple[str, dict[str, Any]]] = []
        self.responses: dict[str, Any] = {
            "GetCallerIdentity": {"Account": "123456789012"},
            "StartCodeInterpreterSession": {"sessionId": SESSION_ID},
            "GetCodeInterpreterSession": {
                "sessionId": SESSION_ID,
                "codeInterpreterIdentifier": INTERPRETER,
                "name": start_request(RUN_ID)["name"],
            },
            "ListCodeInterpreterSessions": {
                "items": [
                    {
                        "sessionId": SESSION_ID,
                        "name": start_request(RUN_ID)["name"],
                    }
                ]
            },
            "InvokeCodeInterpreter": {
                "stream": [
                    {"result": {"structuredContent": {"stdout": "19680.0\n"}}}
                ]
            },
            "StopCodeInterpreterSession": {},
        }

    def call(
        self, service: str, name: str, params: Mapping[str, Any]
    ) -> AwsResponse:
        assert service and isinstance(params, dict)
        self.calls.append(name)
        self.bodies.append((name, dict(params)))
        response = self.responses[name]
        if isinstance(response, list):
            response = response.pop(0)
        if isinstance(response, Exception):
            raise response
        return AwsResponse(
            response, f"https://{SERVICE}.us-east-1.amazonaws.com"
        )

    stream = call


def budget(path: Path, rate: str = "0.000001") -> BudgetRun:
    prices = Mock()
    prices.rate.return_value = Decimal(rate)
    return open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=path,
        run_id=RUN_ID,
        prices=prices,
    )


def context(path: Path, port: Port) -> OwnedResources:
    ctx = OwnedResources(
        run=budget(path), demo="agentcore-tools", port=port, operations=[]
    )
    assert ctx.authenticate()
    return ctx


def denied() -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "AccessDeniedException", "Message": "denied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "operation",
    )


def test_default_sdk_adapters_and_cleanup_other_entry(tmp_path: Path) -> None:
    port = Port()
    client = Mock()
    client.meta.endpoint_url = f"https://{SERVICE}.us-east-1.amazonaws.com"

    def sdk_method(operation: str) -> Any:
        def call(**params: Any) -> dict[str, Any]:
            return dict(port.call(SERVICE, operation, params).payload)

        return call

    for name in port.responses:
        getattr(client, client_method_name(name)).side_effect = sdk_method(
            name
        )
    factory = Mock(return_value=Mock(client=Mock(return_value=client)))
    with use_budget_run(budget(tmp_path)):
        report = run_agentcore_tools_demo(
            execution="live",
            settings=Settings(allow_create=True),
            session_factory=factory,
        )
    assert report["evidence"]["credential_source"] == "default_chain"
    another = tmp_path / "another"
    ctx = context(another, port)
    InterpreterSession(ctx).start()
    ctx.store.record_intent(
        run_id=RUN_ID,
        account_fingerprint=ctx.account_fingerprint,
        region="us-east-1",
        service="unknown",
        operation="Unknown",
        intended_name="awsai-unknown",
        client_token=RUN_ID,
        tags={},
        naming_scheme="unknown",
    )
    report = execute_cleanup(
        run_id=RUN_ID,
        root=another,
        settings=Settings(),
        session_factory=factory,
    )
    assert report["data"]["cleanup_incomplete"]


def test_offline_all_contracts_and_lane_refusals() -> None:
    offline = run_agentcore_tools_demo()
    assert offline["mode"] == "local_contract"
    assert offline["data"]["stdout"] == "19680.0\n"
    assert len(offline["operations"]) == 8
    assert run_agentcore_tools_demo(execution="emulator")["mode"] == "not_run"
    live = run_agentcore_tools_demo(execution="live")
    assert live["operations"][0]["error_code"] == "create_not_allowed"
    assert _read_stream([{}, {"result": {}}]) == ""
    assert _read_stream([]) is None
    for event in (
        {"result": {"isError": True}},
        {"validationException": {"message": "private diagnostic"}},
    ):
        with pytest.raises(ValueError, match="Code Interpreter"):
            _read_stream([event])


def test_live_session_cost_reserved_and_cleanup(tmp_path: Path) -> None:
    port = Port()
    run = budget(tmp_path)
    report = run_agentcore_tools_demo(
        execution="live",
        settings=Settings(allow_create=True),
        budget_run=run,
        port=port,
    )
    assert report["data"]["stdout"] == "19680.0\n"
    assert report["mode"] == "live_service"
    assert report["data"]["cleanup_incomplete"] is False
    assert port.calls == [
        "GetCallerIdentity",
        "StartCodeInterpreterSession",
        "InvokeCodeInterpreter",
        "GetCodeInterpreterSession",
        "StopCodeInterpreterSession",
    ]
    sent = dict(port.bodies)
    start_token = start_request(RUN_ID)["clientToken"]
    stop_token = f"awsai-{RUN_ID}-interpreter-stop"
    assert start_token != stop_token
    assert sent["StartCodeInterpreterSession"]["clientToken"] == start_token
    assert sent["StopCodeInterpreterSession"]["clientToken"] == stop_token
    assert run.command(
        "agentcore-tools"
    ).ledger.snapshot().reserved_usd == Decimal("0.006")
    assert [charge.quantity for charge in SESSION_CHARGES] == [600, 2400]


def test_policy_and_auth_refusals(tmp_path: Path) -> None:
    port = Port()
    run = budget(tmp_path, "1")
    report = run_agentcore_tools_demo(
        execution="live",
        settings=Settings(allow_create=True),
        budget_run=run,
        port=port,
    )
    assert report["operations"][0]["error_code"] == "budget_exceeded"
    assert not port.calls
    run = budget(tmp_path)
    port.responses["GetCallerIdentity"] = denied()
    report = run_agentcore_tools_demo(
        execution="live",
        settings=Settings(allow_create=True),
        budget_run=run,
        port=port,
    )
    assert port.calls == ["GetCallerIdentity"]
    assert report["status"] == "blocked"


def test_ambiguous_start_reconciles_and_cleans(tmp_path: Path) -> None:
    port = Port()
    port.responses["StartCodeInterpreterSession"] = denied()
    report = run_agentcore_tools_demo(
        execution="live",
        settings=Settings(allow_create=True),
        budget_run=budget(tmp_path),
        port=port,
    )
    assert report["data"]["cleanup_incomplete"] is False
    assert "InvokeCodeInterpreter" not in port.calls
    assert "ListCodeInterpreterSessions" in port.calls


def test_session_ownership_and_pagination_fail_closed(tmp_path: Path) -> None:
    port = Port()
    ctx = context(tmp_path, port)
    session = InterpreterSession(ctx)
    entry = session.start()
    assert entry is not None
    assert session.cleanup_entry(replace(entry, account_fingerprint="other"))
    assert session.cleanup_entry(replace(entry, region="eu-west-1"))
    original = port.responses["GetCodeInterpreterSession"]
    for key in ("name", "sessionId", "codeInterpreterIdentifier"):
        port.responses["GetCodeInterpreterSession"] = {
            **original,
            key: "wrong",
        }
        assert session.cleanup_entry(entry)
    port.responses["GetCodeInterpreterSession"] = original
    assert session.cleanup_entry(replace(entry, client_token=RUN_ID))
    port.responses["StopCodeInterpreterSession"] = denied()
    assert session.cleanup_entry(entry)
    port.responses["StopCodeInterpreterSession"] = {}
    assert not session.cleanup_entry(entry)
    assert not session.cleanup_entry(ctx.store.list_entries()[0])
    port.responses["GetCodeInterpreterSession"] = {
        **original,
        "status": "TERMINATED",
    }
    assert not session.cleanup_entry(entry)
    unknown = replace(entry, exact_id=None)
    for response in ({"items": []}, {"items": [], "nextToken": "repeat"}):
        port.responses["ListCodeInterpreterSessions"] = response
        assert session.cleanup_entry(unknown)


def test_stream_exception_still_stops_and_failed_stop_is_visible(
    tmp_path: Path,
) -> None:
    port = Port()
    port.responses["InvokeCodeInterpreter"] = RuntimeError("broken stream")
    with pytest.raises(RuntimeError, match="broken stream"):
        run_agentcore_tools_demo(
            execution="live",
            settings=Settings(allow_create=True),
            budget_run=budget(tmp_path),
            port=port,
        )
    assert port.calls[-1] == "StopCodeInterpreterSession"
    port.responses["InvokeCodeInterpreter"] = {"stream": []}
    port.responses["StopCodeInterpreterSession"] = denied()
    report = run_agentcore_tools_demo(
        execution="live",
        settings=Settings(allow_create=True),
        budget_run=budget(tmp_path / "another"),
        port=port,
    )
    assert report["error"]["code"] == "cleanup_incomplete"
    assert report["status"] == "error"


def test_cleanup_command_named_session_and_auth_failure(
    tmp_path: Path,
) -> None:
    port = Port()
    ctx = context(tmp_path, port)
    session = InterpreterSession(ctx)
    session.start()
    report = execute_cleanup(
        run_id=RUN_ID, root=tmp_path, settings=Settings(), port=port
    )
    assert not report["data"]["cleanup_incomplete"]
    assert report["data"]["entries"][0]["status"] == "deleted"
    another = tmp_path / "second"
    ctx = context(another, port)
    InterpreterSession(ctx).start()
    port.responses["GetCallerIdentity"] = denied()
    report = execute_cleanup(
        run_id=RUN_ID, root=another, settings=Settings(), port=port
    )
    assert report["data"]["cleanup_incomplete"]
