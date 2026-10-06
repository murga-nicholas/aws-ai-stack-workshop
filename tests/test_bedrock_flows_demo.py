from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from awsai_demo.bedrock_flows_demo import _run_live as run_live_rows
from awsai_demo.bedrock_flows_demo import (
    create_flow_alias_params,
    create_flow_params,
    create_flow_version_params,
    create_prompt_params,
    create_prompt_version_params,
    invoke_flow_params,
    list_flows_params,
    list_prompts_params,
    prepare_flow_params,
    run_bedrock_flows_demo,
)
from awsai_demo.demo_support import AwsResponse
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping


def _data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", result["data"])


class FlowPort:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        self.calls.append((service, operation_name, params))
        if operation_name == "ListPrompts":
            payload: Mapping[str, Any] = {"promptSummaries": [{"id": "p"}]}
        else:
            payload = {"flowSummaries": [{"id": "f"}, {"id": "g"}]}
        return AwsResponse(payload, self.endpoint_url)


class FlowSession:
    def __init__(self) -> None:
        self.client_instance = FlowClient()
        self.clients: list[tuple[str, dict[str, object]]] = []

    def client(self, service_name: str, **kwargs: object) -> FlowClient:
        self.clients.append((service_name, kwargs))
        self.client_instance.meta = FlowMeta(
            str(kwargs.get("endpoint_url", "https://bedrock-agent.aws")),
        )
        return self.client_instance


class FlowClient:
    def __init__(self) -> None:
        self.meta = FlowMeta("https://bedrock-agent.us-east-1.amazonaws.com")
        self.port = FlowPort(self.meta.endpoint_url)

    def list_prompts(self, **kwargs: Any) -> dict[str, object]:
        return dict(
            self.port.call("bedrock-agent", "ListPrompts", kwargs).payload
        )

    def list_flows(self, **kwargs: Any) -> dict[str, object]:
        return dict(
            self.port.call("bedrock-agent", "ListFlows", kwargs).payload
        )


class FlowMeta:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url


def test_bedrock_flows_offline_validates_all_fixture_rows() -> None:
    result = run_bedrock_flows_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["mode"] == "local_contract"
    assert result["status"] == "ok"
    assert [item["operation"] for item in result["operations"]] == [
        "CreatePrompt",
        "CreatePromptVersion",
        "CreateFlow",
        "PrepareFlow",
        "CreateFlowVersion",
        "CreateFlowAlias",
        "InvokeFlow",
        "ListPrompts",
        "ListFlows",
    ]
    assert result["operations"][0]["phase"] == "setup"
    assert _data(result)["stream_event_count"] == 2
    assert _data(result)["prompt_count"] == 1


def test_bedrock_flows_emulator_marks_all_rows_unsupported() -> None:
    result = run_bedrock_flows_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["mode"] == "not_run"
    assert len(result["operations"]) == 9
    assert {item["error_code"] for item in result["operations"]} == {
        "not_supported_by_emulator",
    }


def test_bedrock_flows_live_uses_injected_port_for_read_rows() -> None:
    port = FlowPort("https://bedrock-agent.us-east-1.amazonaws.com")
    result = run_bedrock_flows_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert result["mode"] == "live_service"
    assert result["status"] == "ok"
    assert [call[1] for call in port.calls] == ["ListPrompts", "ListFlows"]
    assert _data(result)["prompt_count"] == 1
    assert _data(result)["flow_count"] == 2
    assert result["operations"][0]["error_code"] == "contract_only"
    assert result["operations"][-1]["mode"] == "live_service"


def test_bedrock_flows_live_missing_and_default_port() -> None:
    operations, data = run_live_rows(Settings(), None)
    session = FlowSession()

    result = run_bedrock_flows_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        session_factory=lambda **_kwargs: session,
    )

    assert operations[-2]["error_code"] == "missing_configuration"
    assert data["prompt_count"] == 0
    assert result["evidence"]["credential_source"] == "default_chain"
    assert [call[0] for call in session.clients] == [
        "bedrock-agent",
        "bedrock-agent",
    ]


def test_bedrock_flows_request_helpers() -> None:
    assert create_prompt_params()["name"] == "awsaiDemoPrompt"
    assert create_prompt_version_params()["promptIdentifier"] == "PROMPT12345"
    assert create_flow_params()["executionRoleArn"].startswith("arn:aws:iam")
    assert prepare_flow_params()["flowIdentifier"] == "FLOW12345"
    assert create_flow_version_params()["flowIdentifier"] == "FLOW12345"
    assert create_flow_alias_params()["routingConfiguration"][0] == {
        "flowVersion": "1",
    }
    assert invoke_flow_params()["inputs"][0]["nodeName"] == "FlowInputNode"
    assert list_prompts_params() == {"maxResults": 10}
    assert list_flows_params() == {"maxResults": 10}
