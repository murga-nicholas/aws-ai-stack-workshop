from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from awsai_demo.bedrock_runtime_demo import (
    _run_live as _run_bedrock_live,
)
from awsai_demo.bedrock_runtime_demo import (
    converse_stream_params,
    converse_text_params,
    converse_tool_params,
    count_tokens_params,
    invoke_model_params,
    run_bedrock_runtime_demo,
)
from awsai_demo.demo_support import AwsResponse
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Settings

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pytest


def _data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", result["data"])


class RuntimePort:
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
        if operation_name == "CountTokens":
            return AwsResponse({"inputTokens": 12}, self.endpoint_url)
        return AwsResponse({"ok": True}, self.endpoint_url)


class RuntimeSession:
    def __init__(self) -> None:
        self.clients: list[tuple[str, dict[str, object]]] = []
        self.client_instance = RuntimeClient()

    def client(self, service_name: str, **kwargs: object) -> RuntimeClient:
        self.clients.append((service_name, kwargs))
        endpoint = kwargs.get(
            "endpoint_url",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
        )
        self.client_instance.meta = RuntimeMeta(str(endpoint))
        return self.client_instance


class RuntimeClient:
    def __init__(self) -> None:
        self.meta = RuntimeMeta(
            "https://bedrock-runtime.us-east-1.amazonaws.com"
        )
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def converse(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("converse", kwargs))
        return {"ok": True}

    def invoke_model(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("invoke_model", kwargs))
        return {"ok": True}

    def count_tokens(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("count_tokens", kwargs))
        return {"inputTokens": 12}


class RuntimeMeta:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url


def runtime_session_factory(**kwargs: str) -> RuntimeSession:
    del kwargs
    return RuntimeSession()


def test_bedrock_runtime_offline_uses_strict_fixtures() -> None:
    result = run_bedrock_runtime_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["status"] == "ok"
    assert result["mode"] == "local_contract"
    assert len(result["operations"]) == 7
    assert _data(result)["text"] == "The risk is within budget."
    assert _data(result)["tool_result"] == {
        "approved": False,
        "amount_usd": 19680,
    }
    assert _data(result)["stream_event_count"] == 3


def test_bedrock_runtime_emulator_missing_port_blocks_supported_calls() -> (
    None
):
    result = run_bedrock_runtime_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    errors = [item["error_code"] for item in result["operations"]]
    assert errors[:3] == ["missing_configuration"] * 3
    assert "not_supported_by_emulator" in errors
    assert result["mode"] == "not_run"


def test_bedrock_runtime_emulator_uses_injected_loopback_port() -> None:
    port = RuntimePort(endpoint_url="http://localhost:4566")
    result = run_bedrock_runtime_demo(
        execution="emulator",
        settings=Settings(localstack_endpoint="http://localhost:4566"),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert [call[1] for call in port.calls] == [
        "Converse",
        "Converse",
        "InvokeModel",
    ]
    assert result["mode"] == "local_emulator"
    assert result["evidence"]["emulator"] == {
        "endpoint": "http://localhost:4566"
    }
    assert _data(result)["emulator_endpoint"] == "http://localhost:4566"


def test_bedrock_runtime_emulator_builds_default_port() -> None:
    session = RuntimeSession()

    def factory(**kwargs: str) -> RuntimeSession:
        assert kwargs["region_name"] == "us-east-1"
        return session

    result = run_bedrock_runtime_demo(
        execution="emulator",
        settings=Settings(
            localstack_auth_token="token",  # noqa: S106
            default_bedrock_model="qwen-test",
        ),
        policy=ExecutionPolicy(),
        session_factory=factory,
    )

    assert result["mode"] == "local_emulator"
    assert session.clients[0][1]["endpoint_url"] == "http://localhost:4566"
    assert session.client_instance.calls[0][1]["modelId"] == (
        "ollama.qwen-test"
    )


def test_bedrock_runtime_live_refuses_billable_without_prices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_bedrock_runtime_demo(
        execution="live",
        settings=Settings(model="custom.model"),
        policy=ExecutionPolicy(),
        session_factory=runtime_session_factory,
    )

    assert result["status"] == "blocked"
    assert result["mode"] == "live_service"
    assert result["evidence"]["requested_model"] == "custom.model"
    assert result["operations"][0]["error_code"] == "budget_exceeded"
    assert _data(result)["input_tokens"] == 12


def test_bedrock_runtime_live_without_port_skips_count_tokens() -> None:
    operations, data = _run_bedrock_live(Settings(), "model", None)

    assert operations[4]["operation"] == "CountTokens"
    assert operations[4]["error_code"] == "missing_configuration"
    assert data == {"model": "model", "pricing": "missing"}


def test_bedrock_runtime_emulator_model_id_fallbacks() -> None:
    prefixed_port = RuntimePort(endpoint_url="http://localhost:4566")
    prefixed = run_bedrock_runtime_demo(
        execution="emulator",
        settings=Settings(default_bedrock_model="ollama.existing"),
        policy=ExecutionPolicy(),
        port=prefixed_port,
    )
    configured_settings = Settings()
    object.__setattr__(
        configured_settings,
        "localstack_bedrock_model",
        "ollama.configured",
    )
    configured_port = RuntimePort(endpoint_url="http://localhost:4566")
    configured = run_bedrock_runtime_demo(
        execution="emulator",
        settings=configured_settings,
        policy=ExecutionPolicy(),
        port=configured_port,
    )

    assert prefixed["evidence"]["requested_model"] == "ollama.existing"
    assert prefixed_port.calls[0][2]["modelId"] == "ollama.existing"
    assert configured["evidence"]["requested_model"] == "ollama.configured"
    assert configured_port.calls[0][2]["modelId"] == "ollama.configured"


def test_bedrock_runtime_live_count_tokens_uses_read_port_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    port = RuntimePort(
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
    )
    result = run_bedrock_runtime_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert [call[1] for call in port.calls] == ["CountTokens"]
    assert result["mode"] == "live_service"
    assert result["status"] == "blocked"
    assert _data(result)["input_tokens"] == 12


def test_bedrock_runtime_request_helpers_are_sdk_shapes() -> None:
    model_id = "amazon.nova-2-lite-v1:0"
    tool_params = converse_tool_params(model_id)
    invoke_params = invoke_model_params(model_id)
    count_params = count_tokens_params(model_id)

    assert tool_params["toolConfig"]["tools"][0]["toolSpec"]["name"] == (
        "price_pilot"
    )
    assert invoke_params["contentType"] == "application/json"
    assert count_params["input"]["converse"]["messages"]


def test_haiku_requests_use_one_sampling_control() -> None:
    for builder in (
        converse_text_params,
        converse_tool_params,
        converse_stream_params,
    ):
        config = builder(TESTED_LIVE_BEDROCK_MODEL)["inferenceConfig"]
        assert config == {"maxTokens": 128, "temperature": 0.0}


def test_bedrock_runtime_marker_is_short_and_contains_required_call() -> None:
    lines = _marker_lines(
        "src/awsai_demo/bedrock_runtime_demo.py",
        "converse-request",
    )

    assert len(lines) <= 14
    assert all(len(line) <= 72 for line in lines)
    joined = "\n".join(lines)
    assert "client.converse(" in joined
    assert "modelId=model_id" in joined
    assert "messages=messages" in joined
    assert "system=system" in joined
    assert "inferenceConfig=inference_config" in joined


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
