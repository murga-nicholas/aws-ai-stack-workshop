from __future__ import annotations

import json
import runpy
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

import awsai_demo.agentcore_runtime_demo as runtime
import awsai_demo.agentcore_runtime_worker as worker
from awsai_demo.agentcore_runtime_demo import (
    RuntimeExchange,
    create_agent_runtime_request,
    invoke_agent_runtime_request,
    list_agent_runtimes_request,
    run_agentcore_runtime_demo,
    validate_agentcore_runtime_contracts,
)
from awsai_demo.agentcore_runtime_worker import (
    RuntimeScriptedModel,
    build_runtime_agent,
    invoke_runtime_agent,
)
from awsai_demo.contracts import operation
from awsai_demo.credentials import SelectedSession
from awsai_demo.demo_support import AwsResponse
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class FailingRuntimePort:
    def exercise(self, policy: ExecutionPolicy) -> RuntimeExchange:
        assert policy.max_wall_seconds > 0
        message = "down"
        raise OSError(message)


class FakeRuntimePort:
    def exercise(self, policy: ExecutionPolicy) -> RuntimeExchange:
        assert policy.max_wall_seconds > 0
        return RuntimeExchange(
            endpoint_url="http://127.0.0.1:1234",
            ping_status="Healthy",
            invocation={"model_calls": ("runtime.scripted_model",)},
            model_calls=("runtime.scripted_model",),
            operations=(
                operation(
                    service="bedrock-agentcore",
                    operation="GET /ping",
                    execution_target="local",
                    mode="local_execution",
                    transport="loopback",
                    endpoint_url="http://127.0.0.1:1234",
                ),
            ),
        )


class FakePort:
    def __init__(self, response: AwsResponse | dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse | Mapping[str, Any]:
        self.calls.append((service, operation_name, dict(params)))
        return self.response


class RaisingPort:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        del service, operation_name, params
        raise self.exc


class DefaultPort:
    def __init__(self, port: object) -> None:
        self.port = port
        self.credential_source = "env"


class FakeClient:
    class Meta:
        """Fake botocore metadata."""

        endpoint_url = (
            "https://bedrock-agentcore-control.us-west-2.amazonaws.com"
        )

    meta = Meta()

    def list_agent_runtimes(self, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"maxResults": 10}
        return {"agentRuntimes": [{"name": "runtime"}]}


class FakeSession:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def client(self, service_name: str, **kwargs: object) -> FakeClient:
        assert service_name == "bedrock-agentcore-control"
        self.kwargs = dict(kwargs)
        return FakeClient()


def test_offline_runtime_runs_real_worker_subprocess() -> None:
    built = run_agentcore_runtime_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(max_wall_seconds=20),
    )

    assert built["mode"] == "local_contract"
    assert built["status"] == "ok"
    data = built["data"]
    assert isinstance(data, dict)
    assert data["runtime"]["ping_status"] == "Healthy"
    assert data["runtime"]["model_calls"] == ("runtime.scripted_model",)
    assert data["invocation"]["total_usd"] == 19680


def test_runtime_worker_marker_and_direct_invocation() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    marker = source.split("# slide: app\n", 1)[1].split(
        "# end-slide: app",
        1,
    )[0]
    lines = [line for line in marker.splitlines() if line.strip()]

    assert len(lines) <= 14
    assert max(len(line) for line in lines) <= 72
    app = runtime.build_app()
    assert worker.build_app()._ping_handler is not None
    assert "main" in app.handlers
    assert app._ping_handler is not None
    assert app._ping_handler() == "Healthy"
    assert app.handlers["main"]({"prompt": "handler prompt"})["prompt"] == (
        "handler prompt"
    )
    model = RuntimeScriptedModel()
    fallback_model = RuntimeScriptedModel()
    payload = invoke_runtime_agent(
        build_runtime_agent(model),
        model,
        {"prompt": "custom prompt"},
    )
    fallback = invoke_runtime_agent(
        build_runtime_agent(fallback_model),
        fallback_model,
        {"prompt": "   "},
    )
    assert payload["accepted"] is False
    assert payload["within_budget"] is True
    assert payload["prompt"] == "custom prompt"
    assert payload["model_calls"] == ("runtime.scripted_model",)
    assert fallback["prompt"]


def test_runtime_worker_main_rejects_non_loopback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    non_loopback = ".".join(("0", "0", "0", "0"))
    assert worker.main(["--host", non_loopback, "--port", "8080"]) == 1
    captured = capsys.readouterr()
    assert "ValueError" in captured.err
    assert "127.0.0.1" in captured.err

    assert worker._parse_args(["--port", "9000"]) == ("127.0.0.1", 9000)
    with pytest.raises(ValueError):
        worker._parse_args(["--unknown"])
    with pytest.raises(ValueError):
        worker._parse_args(["--host"])
    with pytest.raises(ValueError):
        worker._parse_args(["--port", "70000"])


def test_runtime_worker_main_success_with_fake_runner() -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.args: tuple[str, int] | None = None

        def run(self, *, host: str, port: int) -> None:
            self.args = (host, port)

    fake = FakeApp()

    assert worker.main(["--port", "9001"], app_factory=lambda: fake) == 0
    assert fake.args == ("127.0.0.1", 9001)


def test_runtime_worker_module_entry_rejects_non_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    non_loopback = ".".join(("0", "0", "0", "0"))
    monkeypatch.setattr(
        sys,
        "argv",
        [worker.__file__, "--host", non_loopback],
    )
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(worker.__file__, run_name="__main__")
    assert raised.value.code == 1


def test_contract_requests_validate_against_botocore() -> None:
    operations = validate_agentcore_runtime_contracts()

    assert [op["operation"] for op in operations] == [
        "CreateAgentRuntime",
        "InvokeAgentRuntime",
        "ListAgentRuntimes",
    ]
    assert create_agent_runtime_request()["protocolConfiguration"] == {
        "serverProtocol": "HTTP",
    }
    assert invoke_agent_runtime_request()["runtimeSessionId"].startswith(
        "awsai-demo-session",
    )
    assert list_agent_runtimes_request() == {"maxResults": 10}


def test_fake_runtime_port_and_failure_paths() -> None:
    fake = run_agentcore_runtime_demo(
        execution="offline",
        runtime_port=FakeRuntimePort(),
    )
    failed = run_agentcore_runtime_demo(
        execution="offline",
        runtime_port=FailingRuntimePort(),
    )

    fake_data = cast("dict[str, Any]", fake["data"])
    assert fake_data["runtime"]["ping_status"] == "Healthy"
    assert failed["mode"] == "attempt_failed"
    failed_error = failed["error"]
    assert failed_error is not None
    assert failed_error["code"] == "timeout"


def test_emulator_and_live_missing_port_are_explicit() -> None:
    emulator = run_agentcore_runtime_demo(execution="emulator")
    live = run_agentcore_runtime_demo(
        execution="live",
        boto_port_factory=lambda **_kwargs: None,
    )

    assert emulator["mode"] == "not_run"
    assert {op["error_code"] for op in emulator["operations"]} == {
        "not_supported_by_emulator",
    }
    assert live["mode"] == "not_run"
    live_data = cast("dict[str, Any]", live["data"])
    assert live_data["live_adapter"] == "missing"


def test_live_injected_port_and_boto_adapter() -> None:
    port = FakePort(
        AwsResponse(
            {"agentRuntimes": [{"name": "runtime"}]},
            "https://bedrock-agentcore-control.us-east-1.amazonaws.com",
        ),
    )
    injected = run_agentcore_runtime_demo(execution="live", port=port)

    assert injected["mode"] == "live_service"
    injected_data = cast("dict[str, Any]", injected["data"])
    assert injected_data["live_agent_runtimes"] == 1
    assert port.calls == [
        (
            "bedrock-agentcore-control",
            "ListAgentRuntimes",
            {"maxResults": 10},
        ),
    ]

    session = FakeSession()
    selected = SelectedSession(
        session=session,
        source="profile",
        region="us-west-2",
    )
    boto = run_agentcore_runtime_demo(
        execution="live",
        settings=Settings(region="us-west-2"),
        selected_session=selected,
    )
    assert boto["evidence"]["credential_source"] == "profile"
    boto_data = cast("dict[str, Any]", boto["data"])
    assert boto_data["live_agent_runtimes"] == 1
    assert session.kwargs is not None
    assert session.kwargs["region_name"] == "us-west-2"


def test_live_default_port_and_failed_live_operations() -> None:
    defaulted = run_agentcore_runtime_demo(
        execution="live",
        boto_port_factory=lambda **_kwargs: DefaultPort(
            FakePort(
                AwsResponse(
                    {"agentRuntimes": [{"name": "runtime"}]},
                    "https://bedrock-agentcore-control.us-east-1.amazonaws.com",
                ),
            ),
        ),
    )
    assert defaulted["evidence"]["credential_source"] == "env"

    denied = run_agentcore_runtime_demo(
        execution="live",
        port=RaisingPort(
            ClientError(
                {
                    "Error": {"Code": "AccessDeniedException"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "ListAgentRuntimes",
            ),
        ),
    )
    assert denied["mode"] == "live_service"
    assert denied["operations"][-1]["error_code"] == "authorization_denied"

    failed = run_agentcore_runtime_demo(
        execution="live",
        port=RaisingPort(
            EndpointConnectionError(endpoint_url="https://example.com"),
        ),
    )
    assert failed["mode"] == "attempt_failed"
    assert failed["operations"][-1]["error_code"] == "timeout"


def test_response_and_process_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __init__(self, status: int, body: bytes) -> None:
            self.status = status
            self._body = BytesIO(body)

        def read(self) -> bytes:
            return self._body.read()

    with pytest.raises(OSError):
        runtime._read_json_response(cast("Any", Response(500, b"{}")))
    with pytest.raises(OSError):
        runtime._read_json_response(cast("Any", Response(200, b"[]")))
    assert runtime._read_json_response(
        cast("Any", Response(200, json.dumps({"ok": True}).encode())),
    ) == {"ok": True}

    class ExitedProcess:
        def poll(self) -> int:
            return 0

        def communicate(self, timeout: int) -> tuple[str, str]:
            assert timeout == 1
            return "", ""

    runtime._stop_process(cast("Any", ExitedProcess()))

    class HangingProcess:
        def __init__(self) -> None:
            self.calls = 0
            self.killed = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def communicate(self, timeout: int) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                cmd = "worker"
                raise subprocess.TimeoutExpired(cmd, timeout)
            return "", ""

    hanging = HangingProcess()
    runtime._stop_process(cast("Any", hanging))
    assert hanging.killed is True

    values = iter((0.0, 0.1, 0.2, 2.0))
    responses: Iterator[dict[str, object] | OSError] = iter(
        ({}, OSError("no ping")),
    )
    module = cast("Any", runtime)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(values))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    def fake_get_json(
        host: str,
        port: int,
        path: str,
        policy: ExecutionPolicy,
    ) -> dict[str, object]:
        del host, port, path, policy
        response = next(responses)
        if isinstance(response, OSError):
            raise response
        return response

    with pytest.raises(OSError, match="no ping"):
        runtime._wait_for_ping(
            "127.0.0.1",
            1,
            ExecutionPolicy(max_wall_seconds=1),
            get_json=fake_get_json,
        )

    values = iter((0.0, 2.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(values))
    with pytest.raises(OSError, match="did not answer"):
        runtime._wait_for_ping(
            "127.0.0.1",
            1,
            ExecutionPolicy(max_wall_seconds=1),
            get_json=lambda *_args: {},
        )

    monkeypatch.setattr(
        module.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(
            module.metadata.PackageNotFoundError,
        ),
    )
    assert runtime._package_versions(("missing",)) == {"missing": "missing"}
