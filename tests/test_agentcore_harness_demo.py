from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from awsai_demo.agentcore_harness_demo import (
    _pause_contract_from_events,
    build_harness_pause_contract,
    build_harness_resume_request,
    create_harness_params,
    delete_harness_params,
    get_harness_params,
    harness_model,
    harness_system_prompt,
    inline_function_tool,
    invoke_harness_params,
    list_harnesses_params,
    run_agentcore_harness_demo,
)
from awsai_demo.agentcore_harness_demo import (
    _run_live as _run_harness_live,
)
from awsai_demo.demo_support import AwsResponse, validate_request
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping


def _data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", result["data"])


class HarnessPort:
    def __init__(self, *, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        self.calls.append((service, operation_name, params))
        return AwsResponse(
            {"harnesses": [{"harnessId": "h"}]}, self.endpoint_url
        )


class HarnessSession:
    def __init__(self, *, endpoint_url: str) -> None:
        self.clients: list[tuple[str, dict[str, object]]] = []
        self.client_instance = HarnessClient(endpoint_url=endpoint_url)

    def client(self, service_name: str, **kwargs: object) -> HarnessClient:
        self.clients.append((service_name, kwargs))
        endpoint = kwargs.get(
            "endpoint_url",
            self.client_instance.meta.endpoint_url,
        )
        self.client_instance.meta = HarnessMeta(str(endpoint))
        return self.client_instance


class HarnessClient:
    def __init__(self, *, endpoint_url: str) -> None:
        self.meta = HarnessMeta(endpoint_url)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def list_harnesses(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("list_harnesses", kwargs))
        return {"harnesses": [{"harnessId": "h"}]}


class HarnessMeta:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url


def test_agentcore_harness_offline_returns_pending_pause_data() -> None:
    result = run_agentcore_harness_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["status"] == "ok"
    assert result["mode"] == "local_contract"
    assert result["lifecycle_refs"] == ["agentcore-harness", "agents-classic"]
    assert len(result["operations"]) == 5
    assert [item["operation"] for item in result["operations"]] == [
        "CreateHarness",
        "GetHarness",
        "InvokeHarness",
        "ListHarnesses",
        "DeleteHarness",
    ]
    assert _data(result)["status_after_get"] == "READY"
    assert _data(result)["tool_use_id"] == "tooluse-accept-proposal-1"
    assert _data(result)["pause"]["state"] == "pending"
    assert _data(result)["pause"]["tool_name"] == "accept_proposal"
    assert _data(result)["pause"]["stop_reason"] == "tool_use"
    assert (
        _data(result)["resume_request"]["messages"][0]["content"][0][
            "toolResult"
        ]["status"]
        == "error"
    )


def test_agentcore_harness_emulator_missing_token_blocks() -> None:
    result = run_agentcore_harness_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["status"] == "blocked"
    assert result["mode"] == "not_run"
    assert [item["operation"] for item in result["operations"]] == [
        "CreateHarness",
        "GetHarness",
        "InvokeHarness",
        "ListHarnesses",
        "DeleteHarness",
    ]
    assert {item["error_code"] for item in result["operations"]} == {
        "missing_configuration"
    }


def test_agentcore_harness_emulator_builds_default_port_then_blocks() -> None:
    session = HarnessSession(endpoint_url="http://localhost:4566")

    def factory(**kwargs: str) -> HarnessSession:
        assert kwargs["region_name"] == "us-east-1"
        return session

    result = run_agentcore_harness_demo(
        execution="emulator",
        settings=Settings(
            localstack_auth_token="token",  # noqa: S106
            default_bedrock_model="qwen-test",
        ),
        policy=ExecutionPolicy(),
        session_factory=factory,
    )

    assert result["status"] == "blocked"
    assert result["mode"] == "not_run"
    assert _data(result) == {
        "model": "ollama.qwen-test",
        "emulator": "not_supported",
    }
    assert session.clients == []
    assert {item["error_code"] for item in result["operations"]} == {
        "not_supported_by_emulator"
    }


def test_agentcore_harness_live_without_allow_create_lists_only() -> None:
    port = HarnessPort(
        endpoint_url="https://bedrock-agentcore-control.us-east-1.amazonaws.com"
    )
    result = run_agentcore_harness_demo(
        execution="live",
        settings=Settings(allow_create=False),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert result["mode"] == "live_service"
    assert result["status"] == "ok"
    assert [call[1] for call in port.calls] == ["ListHarnesses"]
    assert _data(result)["listed"] == 1
    assert result["operations"][0]["error_code"] == "create_not_allowed"


def test_agentcore_harness_live_allow_create_refuses_missing_prices() -> None:
    port = HarnessPort(
        endpoint_url="https://bedrock-agentcore-control.us-east-1.amazonaws.com"
    )
    result = run_agentcore_harness_demo(
        execution="live",
        settings=Settings(allow_create=True),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert result["status"] == "blocked"
    assert result["mode"] == "live_service"
    assert result["operations"][0]["error_code"] == "budget_exceeded"
    assert result["operations"][-1]["operation"] == "DeleteHarness"
    assert _data(result)["listed"] == 1
    assert _data(result)["pricing"] == "missing"


def test_agentcore_harness_live_builds_default_read_port() -> None:
    session = HarnessSession(
        endpoint_url="https://bedrock-agentcore-control.us-east-1.amazonaws.com"
    )

    def factory(**kwargs: str) -> HarnessSession:
        assert kwargs == {"region_name": "us-east-1"}
        return session

    result = run_agentcore_harness_demo(
        execution="live",
        settings=Settings(allow_create=False),
        policy=ExecutionPolicy(),
        session_factory=factory,
    )

    assert result["mode"] == "live_service"
    assert result["operations"][-2]["operation"] == "ListHarnesses"
    assert result["operations"][-1]["operation"] == "DeleteHarness"
    assert session.client_instance.calls == [
        ("list_harnesses", {"maxResults": 10})
    ]


def test_agentcore_harness_live_without_port_reports_missing_list() -> None:
    operations, data = _run_harness_live(
        Settings(allow_create=False), "model", None
    )

    assert operations[-2]["operation"] == "ListHarnesses"
    assert operations[-2]["error_code"] == "missing_configuration"
    assert operations[-1]["operation"] == "DeleteHarness"
    assert data["pricing"] == "missing"


def test_agentcore_harness_rejects_non_pause_stream() -> None:
    with pytest.raises(ValueError, match="did not pause"):
        _pause_contract_from_events(())


def test_agentcore_harness_model_id_fallbacks() -> None:
    custom = run_agentcore_harness_demo(
        execution="offline",
        settings=Settings(model="custom.model"),
        policy=ExecutionPolicy(),
    )
    prefixed = run_agentcore_harness_demo(
        execution="emulator",
        settings=Settings(default_bedrock_model="ollama.existing"),
        policy=ExecutionPolicy(),
        port=HarnessPort(endpoint_url="http://localhost:4566"),
    )
    configured_settings = Settings()
    object.__setattr__(
        configured_settings,
        "localstack_bedrock_model",
        "ollama.configured",
    )
    configured = run_agentcore_harness_demo(
        execution="emulator",
        settings=configured_settings,
        policy=ExecutionPolicy(),
        port=HarnessPort(endpoint_url="http://localhost:4566"),
    )

    assert custom["evidence"]["requested_model"] == "custom.model"
    assert _data(prefixed)["model"] == "ollama.existing"
    assert _data(configured)["model"] == "ollama.configured"


def test_agentcore_harness_resume_request_shapes() -> None:
    contract = build_harness_pause_contract(
        harness_arn=(
            "arn:aws:bedrock-agentcore:us-east-1:123456789012:"
            "harness/awsaiDemoHarness-1A2B3C4D5E"
        ),
        runtime_session_id="session-0123456789abcdef0123456789",
        tool_use_id="tooluse-accept-proposal-1",
    )
    denied = build_harness_resume_request(contract)
    approved = build_harness_resume_request(
        contract,
        approved=True,
        reason="Approved by isolated fixture.",
    )

    assert denied["messages"][0]["content"][0]["toolResult"]["status"] == (
        "error"
    )
    assert approved["messages"][0]["content"][0]["toolResult"]["status"] == (
        "success"
    )
    validate_request(
        service="bedrock-agentcore",
        operation_name="InvokeHarness",
        params=approved,
        fixture_id="agentcore-harness-resume-approved-v1",
    )


def test_agentcore_harness_request_helpers_match_sdk_shapes() -> None:
    model_id = "global.amazon.nova-2-lite-v1:0"
    role_arn = "arn:aws:iam::123456789012:role/awsai-demo-harness"
    create_params = create_harness_params(
        model_id=model_id,
        role_arn=role_arn,
    )

    assert create_params["tools"][0] == inline_function_tool("accept_proposal")
    assert harness_model(model_id)["bedrockModelConfig"]["modelId"] == model_id
    assert "topP" not in harness_model(model_id)["bedrockModelConfig"]
    assert harness_system_prompt()[0]["text"]
    validate_request(
        service="bedrock-agentcore-control",
        operation_name="CreateHarness",
        params=create_params,
        fixture_id="agentcore-harness-create-v1",
    )
    for operation_name, params in (
        ("GetHarness", get_harness_params()),
        ("DeleteHarness", delete_harness_params()),
        ("ListHarnesses", list_harnesses_params()),
    ):
        validate_request(
            service="bedrock-agentcore-control",
            operation_name=operation_name,
            params=params,
            fixture_id=f"agentcore-harness-{operation_name}",
        )
    validate_request(
        service="bedrock-agentcore",
        operation_name="InvokeHarness",
        params=invoke_harness_params(model_id),
        fixture_id="agentcore-harness-invoke-tool-use-v1",
    )


def test_agentcore_harness_marker_is_short_and_shows_inline_tool() -> None:
    lines = _marker_lines(
        "src/awsai_demo/agentcore_harness_demo.py",
        "create-harness",
    )

    assert len(lines) <= 14
    assert all(len(line) <= 72 for line in lines)
    joined = "\n".join(lines)
    assert "client.create_harness(" in joined
    assert "harnessName=_HARNESS_NAME" in joined
    assert "executionRoleArn=role_arn" in joined
    assert "tools=[inline_function_tool(_TOOL_NAME)]" in joined


def _marker_lines(path: str, marker: str) -> list[str]:
    start = f"# slide: {marker}"
    end = f"# end-slide: {marker}"
    in_marker = False
    lines: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.strip() == start:
                in_marker = True
                continue
            if line.strip() == end:
                break
            if in_marker:
                lines.append(line)
    return [line.removeprefix("    ") for line in lines]
