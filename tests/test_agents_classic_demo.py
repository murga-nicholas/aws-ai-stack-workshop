from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from botocore.exceptions import ClientError

from awsai_demo.agents_classic_demo import _run_live as run_live_rows
from awsai_demo.agents_classic_demo import (
    classify_access_denied,
    invoke_agent_params,
    list_agents_params,
    return_control_result_params,
    run_agents_classic_demo,
)
from awsai_demo.demo_support import AwsResponse
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping


def _data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", result["data"])


class ClassicPort:
    def __init__(
        self,
        endpoint_url: str,
        *,
        error_message: str | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.error_message = error_message
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        self.calls.append((service, operation_name, params))
        if self.error_message is not None:
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": self.error_message,
                    },
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                operation_name,
            )
        return AwsResponse(
            {"agentSummaries": [{"agentId": "one"}, {"agentId": "two"}]},
            self.endpoint_url,
        )


class MappingClassicPort:
    endpoint_url = "https://bedrock-agent.us-east-1.amazonaws.com"

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> Mapping[str, object]:
        del service, operation_name, params
        return {"agentSummaries": [{"agentId": "mapped"}]}


class ClassicSession:
    def __init__(self) -> None:
        self.client_instance = ClassicClient()
        self.clients: list[tuple[str, dict[str, object]]] = []

    def client(self, service_name: str, **kwargs: object) -> ClassicClient:
        self.clients.append((service_name, kwargs))
        self.client_instance.meta = ClassicMeta(
            str(kwargs.get("endpoint_url", "https://bedrock-agent.aws")),
        )
        return self.client_instance


class ClassicClient:
    def __init__(self) -> None:
        self.meta = ClassicMeta(
            "https://bedrock-agent.us-east-1.amazonaws.com"
        )
        self.port = ClassicPort(self.meta.endpoint_url)

    def list_agents(self, **kwargs: Any) -> dict[str, object]:
        return dict(
            self.port.call("bedrock-agent", "ListAgents", kwargs).payload
        )


class ClassicMeta:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url


def test_agents_classic_offline_return_control_contract() -> None:
    result = run_agents_classic_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["mode"] == "local_contract"
    assert result["status"] == "ok"
    assert [item["operation"] for item in result["operations"]] == [
        "InvokeAgent",
        "InvokeAgent",
        "ListAgents",
        "ClassifyAccessDeniedException",
    ]
    assert _data(result)["agents"] == {"classic_listed": 0}
    assert _data(result)["diagnostic"] == "authorization_denied"
    assert _data(result)["returnControl"]["invocationId"] == (
        "approval-contract"
    )


def test_agents_classic_emulator_is_unsupported() -> None:
    result = run_agents_classic_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["mode"] == "not_run"
    assert len(result["operations"]) == 3
    assert {item["error_code"] for item in result["operations"]} == {
        "not_supported_by_emulator",
    }


def test_agents_classic_live_lists_agents_and_sets_fact_path() -> None:
    port = ClassicPort("https://bedrock-agent.us-east-1.amazonaws.com")
    result = run_agents_classic_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert result["mode"] == "live_service"
    assert result["status"] == "ok"
    assert [call[1] for call in port.calls] == ["ListAgents"]
    assert _data(result)["agents"]["classic_listed"] == 2
    assert "historical allowlisting" in result["headline"]
    assert result["operations"][0]["error_code"] == "contract_only"
    assert result["operations"][1]["mode"] == "live_service"


def test_agents_classic_live_accepts_mapping_payload() -> None:
    result = run_agents_classic_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=MappingClassicPort(),
    )

    assert _data(result)["agents"] == {"classic_listed": 1}
    assert result["operations"][1]["endpoint_url"] == (
        "https://bedrock-agent.us-east-1.amazonaws.com"
    )


def test_agents_classic_live_maintenance_diagnostic_is_exact() -> None:
    port = ClassicPort(
        "https://bedrock-agent.us-east-1.amazonaws.com",
        error_message=(
            "Bedrock Agents is in Maintenance Mode. New agent creation is "
            "not available."
        ),
    )
    result = run_agents_classic_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert result["mode"] == "live_service"
    assert result["operations"][1]["error_code"] == "maintenance_mode"
    assert _data(result)["diagnostic"] == "maintenance_mode"


def test_agents_classic_live_missing_and_default_port() -> None:
    operations, data = run_live_rows(Settings(), None)
    session = ClassicSession()

    result = run_agents_classic_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        session_factory=lambda **_kwargs: session,
    )

    assert operations[1]["error_code"] == "missing_configuration"
    assert data["agents"] == {"classic_listed": 0}
    assert result["evidence"]["credential_source"] == "default_chain"
    assert session.clients[0][0] == "bedrock-agent"


def test_agents_classic_helpers_and_authorization_diagnostic() -> None:
    denied = ClientError(
        {
            "Error": {"Code": "AccessDeniedException", "Message": "Denied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "ListAgents",
    )

    assert classify_access_denied(denied) == "authorization_denied"
    assert invoke_agent_params()["agentId"] == "ABCDEFGHIJ"
    resume = return_control_result_params()
    assert resume["sessionState"]["returnControlInvocationResults"]
    assert list_agents_params() == {"maxResults": 10}
