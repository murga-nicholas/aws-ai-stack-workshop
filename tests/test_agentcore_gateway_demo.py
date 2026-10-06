from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer

import awsai_demo.gateway_server as gateway_server
from awsai_demo.agentcore_gateway_demo import (
    CEDAR_POLICY,
    POLICY_DENIED_REASON,
    _first_json_value,
    _package_versions,
    build_stdio_server_parameters,
    evaluate_accept_proposal,
    run_agentcore_gateway_demo,
    run_mcp_round_trip,
    validate_agentcore_gateway_contracts,
)
from awsai_demo.credentials import SelectedSession
from awsai_demo.demo_support import AwsResponse
from awsai_demo.gateway_server import accept_proposal, price_pilot
from awsai_demo.runtime import Settings


def test_gateway_demo_runs_mcp_cedar_and_contracts() -> None:
    built = run_agentcore_gateway_demo(
        execution="offline",
        settings=Settings(),
    )

    assert built["mode"] == "local_contract"
    assert built["data"]["tool_names"] == (
        "accept_proposal",
        "price_pilot",
    )
    assert built["data"]["policy_denied"] is True
    assert built["data"]["policy_allowed"] is True
    assert built["data"]["denied_reason"] == POLICY_DENIED_REASON
    assert built["data"]["price_total"] == 19680
    assert len(built["data"]["request_contracts"]) == 6
    assert [op["mode"] for op in built["operations"]].count(
        "local_execution",
    ) == 2
    assert [op["mode"] for op in built["operations"]].count(
        "local_contract",
    ) == 6


def test_gateway_emulator_is_contract_only_without_fixtures() -> None:
    built = run_agentcore_gateway_demo(
        execution="emulator",
        settings=Settings(),
    )

    assert built["mode"] == "not_run"
    assert built["data"]["emulator"] == "contract_only"
    assert len(built["operations"]) == 6
    assert {op["mode"] for op in built["operations"]} == {"not_run"}
    assert {op["fixture_id"] for op in built["operations"]} == {None}
    assert all(op["request_validated"] for op in built["operations"])


def test_gateway_live_without_port_lists_missing_configuration() -> None:
    built = run_agentcore_gateway_demo(execution="live", settings=Settings())

    assert built["mode"] == "not_run"
    assert built["data"]["live_adapter"] == "missing"
    assert [op["error_code"] for op in built["operations"][-2:]] == [
        "missing_configuration",
        "missing_configuration",
    ]


def test_gateway_live_uses_boto_adapter_for_two_read_only_lists() -> None:
    class FakeClient:
        class Meta:
            endpoint_url = "https://agentcore.us-east-1.amazonaws.com"

        meta = Meta()

        def list_gateways(self, **params: object) -> dict[str, object]:
            assert params == {"maxResults": 5}
            return {"gateways": [{"name": "gw"}]}

        def list_policy_engines(self, **params: object) -> dict[str, object]:
            assert params == {"maxResults": 5}
            return {"policyEngines": [{"name": "pe"}, {"name": "pe2"}]}

    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def client(self, service_name: str, **_kwargs: object) -> FakeClient:
            self.calls.append(service_name)
            return FakeClient()

    session = FakeSession()
    built = run_agentcore_gateway_demo(
        execution="live",
        settings=Settings(),
        selected_session=SelectedSession(
            session=session,
            source="profile",
            region="us-east-1",
        ),
    )

    assert built["mode"] == "live_service"
    assert session.calls == [
        "bedrock-agentcore-control",
        "bedrock-agentcore-control",
    ]
    assert built["evidence"]["credential_source"] == "profile"
    assert built["data"]["live_gateways"] == 1
    assert built["data"]["live_policy_engines"] == 2
    assert [op["mode"] for op in built["operations"][-2:]] == [
        "live_service",
        "live_service",
    ]
    assert {op["fixture_id"] for op in built["operations"]} == {None}


def test_gateway_live_accepts_injected_fake_port() -> None:
    class FakePort:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(
            self,
            service: str,
            operation_name: str,
            params: dict[str, object],
        ) -> AwsResponse:
            assert service == "bedrock-agentcore-control"
            assert params == {"maxResults": 5}
            self.calls.append(operation_name)
            if operation_name == "ListGateways":
                return AwsResponse(
                    {"gateways": []},
                    "https://agentcore.us-east-1.amazonaws.com",
                )
            return {"policyEngines": []}

    port = FakePort()
    built = run_agentcore_gateway_demo(
        execution="live",
        settings=Settings(),
        port=port,
    )

    assert built["mode"] == "live_service"
    assert port.calls == ["ListGateways", "ListPolicyEngines"]


def test_mcp_round_trip_uses_stdio_server() -> None:
    result = run_mcp_round_trip()

    assert result["tool_names"] == ("accept_proposal", "price_pilot")
    assert result["price_total"] == 19680


def test_cedar_policy_denies_unapproved_and_allows_approved() -> None:
    denied = evaluate_accept_proposal(approved=False)
    allowed = evaluate_accept_proposal(approved=True)

    assert "resource.proposal_version" in CEDAR_POLICY
    assert denied["allowed"] is False
    assert denied["reason"] == POLICY_DENIED_REASON
    assert allowed["allowed"] is True
    assert allowed["reason"] == ""


def test_agentcore_gateway_request_contracts_are_validated() -> None:
    operations = validate_agentcore_gateway_contracts()

    assert [op["operation"] for op in operations] == [
        "CreateGateway",
        "CreateGatewayTarget",
        "CreatePolicyEngine",
        "CreatePolicy",
        "ListGateways",
        "ListPolicyEngines",
    ]
    assert {op["fixture_id"] for op in operations} == {
        "agentcore-gateway-request-CreateGateway-v1",
        "agentcore-gateway-request-CreateGatewayTarget-v1",
        "agentcore-gateway-request-CreatePolicyEngine-v1",
        "agentcore-gateway-request-CreatePolicy-v1",
        "agentcore-gateway-request-ListGateways-v1",
        "agentcore-gateway-request-ListPolicyEngines-v1",
    }


def test_gateway_server_tools_share_the_scenario() -> None:
    priced = price_pilot()
    accepted = accept_proposal(
        approved=True,
        proposal_version=str(priced["proposal_version"]),
        approver="Ada",
    )
    denied = accept_proposal(
        approved=False,
        proposal_version=str(priced["proposal_version"]),
        approver="Ada",
    )

    assert priced["total_usd"] == 19680
    assert accepted["accepted"] is True
    assert accepted["approver"] == "Ada"
    assert denied["accepted"] is False
    assert denied["approver"] is None


def test_stdio_parameters_allow_absent_network_guard() -> None:
    server = build_stdio_server_parameters(policy_lookup=lambda: None)

    assert server.args == [
        "-c",
        "from awsai_demo.gateway_server import main; main()",
    ]
    assert server.env is None


def test_mcp_payload_parsing_falls_back_to_content_json() -> None:
    assert (
        _first_json_value(
            {"content": [{"json": {"total_usd": 42}}]}, "total_usd"
        )
        == 42
    )
    with pytest.raises(ValueError, match="did not include"):
        _first_json_value({"content": []}, "total_usd")
    with pytest.raises(ValueError, match="did not include"):
        _first_json_value({"content": [{}]}, "total_usd")
    with pytest.raises(ValueError, match="did not include"):
        _first_json_value({"content": [{"json": {"other": 1}}]}, "total_usd")


def test_gateway_package_versions_handles_missing_package() -> None:
    assert _package_versions(("not-a-real-awsai-demo-package",)) == {
        "not-a-real-awsai-demo-package": "missing",
    }


def test_gateway_server_entrypoint_runs_stdio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(self: MCPServer, transport: str = "stdio") -> None:
        del self
        calls.append(transport)

    monkeypatch.setattr(MCPServer, "run", fake_run)

    gateway_server.main()
    runpy.run_path(str(Path(gateway_server.__file__)), run_name="__main__")

    assert calls == ["stdio", "stdio"]
