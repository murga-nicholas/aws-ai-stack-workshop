"""AgentCore Runtime local app and live request-contract demo.

Lane: agents; lifecycle: agentcore-runtime.
Run: uv run awsai-demo agentcore-runtime --execution offline.
"""

from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from awsai_demo import scenario
from awsai_demo.agentcore_runtime_worker import (
    RuntimeScriptedModel,
    build_runtime_agent,
    invoke_runtime_agent,
)
from awsai_demo.contracts import (
    CredentialSource,
    DemoResult,
    Execution,
    OperationOutcome,
    error,
    evidence,
    operation,
    result,
)
from awsai_demo.demo_support import (
    AwsPort,
    BotoAwsPort,
    BotoSessionPort,
    bounded_boto_config,
    not_run_operation,
    port_operation,
    require_port_not_run,
    resolve_boto_port,
    unsupported_emulator,
    validate_request,
)
from awsai_demo.network import (
    NetworkPolicy,
    active_policy,
    environment_for_subprocess,
    network_guard,
    registered_loopback_endpoint,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings
from awsai_demo.stubs import validate_operation_request

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    from awsai_demo.credentials import SelectedSession
    from awsai_demo.demo_support import BotoPortFactory


FIXTURE_PREFIX = "agentcore-runtime-request"
_WORKER_FIXTURE = "strands-script-agentcore-runtime-worker-v1"
_DEMO = "agentcore-runtime"
_TECHNOLOGY = "AgentCore Runtime"
_REFS = ("agentcore-runtime",)
_CONTROL = "bedrock-agentcore-control"
_RUNTIME = "bedrock-agentcore"


@dataclass(frozen=True)
class RuntimeExchange:
    """Local runtime HTTP transcript."""

    endpoint_url: str
    ping_status: str
    invocation: Mapping[str, Any]
    model_calls: tuple[str, ...]
    operations: tuple[OperationOutcome, ...]


# fmt: off
# slide: app
def build_app() -> BedrockAgentCoreApp:
    """Return the local AgentCore Runtime contract app."""
    from bedrock_agentcore.runtime import BedrockAgentCoreApp
    app = BedrockAgentCoreApp()
    model = RuntimeScriptedModel()
    agent = build_runtime_agent(model)
    @app.ping
    def ping() -> str:
        return "Healthy"
    @app.entrypoint
    def invoke(payload: dict[str, Any]) -> dict[str, object]:
        return invoke_runtime_agent(agent, model, payload)
    return app
# end-slide: app
# fmt: on


class RuntimePort(Protocol):
    """Port that exercises the local runtime contract."""

    def exercise(self, policy: ExecutionPolicy) -> RuntimeExchange:
        """Return the runtime ping and invocation transcript."""


def run_agentcore_runtime_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    runtime_port: RuntimePort | None = None,
    port: AwsPort | None = None,
    selected_session: SelectedSession | None = None,
    boto_port_factory: BotoPortFactory | None = None,
) -> DemoResult:
    """Run local AgentCore Runtime and live read-only contracts."""
    active_settings = settings or Settings()
    chosen_policy = policy or ExecutionPolicy()
    if execution == "offline":
        active_port = runtime_port or LocalRuntimeSubprocessPort()
        return _run_offline(active_settings, chosen_policy, active_port)
    if execution == "emulator":
        return _run_emulator(active_settings)
    return _run_live_read_only(
        active_settings,
        chosen_policy,
        port=port,
        selected_session=selected_session,
        boto_port_factory=boto_port_factory,
    )


class LocalRuntimeSubprocessPort:
    """Exercise BedrockAgentCoreApp over guarded loopback HTTP."""

    def exercise(self, policy: ExecutionPolicy) -> RuntimeExchange:
        """Start worker, call /ping and /invocations, then stop it."""
        host = "127.0.0.1"
        port = _free_loopback_port()
        endpoint = f"http://{host}:{port}"
        guard = _runtime_network_policy(host, port)
        with tempfile.TemporaryDirectory(prefix="awsai-runtime-") as raw_tmp:
            env = environment_for_subprocess(
                guard,
                bootstrap_dir=Path(raw_tmp),
            )
            command = [
                sys.executable,
                "-m",
                "awsai_demo.agentcore_runtime_worker",
                "--host",
                host,
                "--port",
                str(port),
            ]
            with network_guard(guard):
                process = _start_worker(command, env)
                try:
                    ping = _wait_for_ping(host, port, policy)
                    invocation = _post_json(
                        host,
                        port,
                        "/invocations",
                        {"prompt": scenario.BRIEF},
                        policy,
                    )
                finally:
                    _stop_process(process)
        ping_status = str(ping.get("status", "unknown"))
        model_calls = tuple(
            str(item) for item in invocation.get("model_calls", ())
        )
        operations = (
            operation(
                service="bedrock-agentcore",
                operation="BedrockAgentCoreApp.run",
                phase="setup",
                execution_target="local",
                mode="local_execution",
            ),
            operation(
                service="bedrock-agentcore",
                operation="GET /ping",
                execution_target="local",
                mode="local_execution",
                transport="loopback",
                endpoint_url=endpoint,
                http_status=200,
            ),
            operation(
                service="bedrock-agentcore",
                operation="POST /invocations",
                execution_target="local",
                mode="local_execution",
                transport="loopback",
                endpoint_url=endpoint,
                http_status=200,
            ),
            *(
                operation(
                    service="bedrock-runtime",
                    operation=f"ScriptedModel.stream:{call}",
                    mode="local_contract",
                    effect="none",
                    fixture_id=_WORKER_FIXTURE,
                    usage={
                        "input_tokens": 70,
                        "output_tokens": 18,
                        "total_tokens": 88,
                    },
                )
                for call in model_calls
            ),
        )
        return RuntimeExchange(
            endpoint_url=endpoint,
            ping_status=ping_status,
            invocation=invocation,
            model_calls=model_calls,
            operations=operations,
        )


def validate_agentcore_runtime_contracts() -> list[OperationOutcome]:
    """Validate AgentCore Runtime control-plane request shapes."""
    operations: list[OperationOutcome] = []
    for service, name, params in _request_rows():
        validation = validate_operation_request(
            service_name=service,
            operation_name=name,
            params=params,
            fixture_id=_fixture_id(name),
        )
        operations.append(
            operation(
                service=service,
                operation=name,
                mode="local_contract",
                effect="write" if name == "CreateAgentRuntime" else "read",
                fixture_id=validation.fixture_id,
                request_validated=validation.request_validated,
            ),
        )
    return operations


def _run_offline(
    settings: Settings,
    policy: ExecutionPolicy,
    runtime_port: RuntimePort,
) -> DemoResult:
    try:
        exchange = runtime_port.exercise(policy)
    except OSError as exc:
        return _runtime_attempt_failed(settings, str(exc))
    operations = [
        *exchange.operations,
        *validate_agentcore_runtime_contracts(),
    ]
    data = {
        "runtime": {
            "endpoint": exchange.endpoint_url,
            "ping_status": exchange.ping_status,
            "model_calls": exchange.model_calls,
        },
        "invocation": dict(exchange.invocation),
        "request_contracts": [row[1] for row in _request_rows()],
    }
    return _runtime_result(
        execution="offline",
        settings=settings,
        operations=operations,
        headline="Local BedrockAgentCoreApp answered ping and invocation.",
        provider="local-agentcore-runtime",
        credential_source="none",
        data=data,
    )


def _run_emulator(settings: Settings) -> DemoResult:
    operations = [
        unsupported_emulator(
            service=_CONTROL,
            operation_name="CreateAgentRuntime",
        ),
        unsupported_emulator(
            service=_RUNTIME,
            operation_name="InvokeAgentRuntime",
        ),
    ]
    return _runtime_result(
        execution="emulator",
        settings=settings,
        operations=operations,
        headline="AgentCore Runtime has no LocalStack emulator path.",
        provider="localstack-contract-only",
        credential_source="none",
        data={"emulator": "not_supported"},
    )


def _run_live_read_only(
    settings: Settings,
    policy: ExecutionPolicy,
    *,
    port: AwsPort | None,
    selected_session: SelectedSession | None,
    boto_port_factory: BotoPortFactory | None = None,
) -> DemoResult:
    active_port = port
    credential_source = _credential_source(selected_session)
    if active_port is None:
        active_port, credential_source = _live_port_from_session(
            settings,
            policy,
            selected_session,
            boto_port_factory=boto_port_factory,
        )
    operations = [_validated_not_run(row) for row in _request_rows()[:2]]
    data: dict[str, object] = {
        "request_contracts": [row[1] for row in _request_rows()[:2]],
    }
    if active_port is None:
        operations.append(
            require_port_not_run(
                service=_CONTROL,
                operation_name="ListAgentRuntimes",
            ),
        )
        data["live_adapter"] = "missing"
    else:
        outcome, payload = _live_read_operation(
            port=active_port,
            settings=settings,
            operation_name="ListAgentRuntimes",
            params=list_agent_runtimes_request(),
        )
        operations.append(outcome)
        data["live_agent_runtimes"] = len(
            cast("Sequence[object]", payload.get("agentRuntimes", ())),
        )
    return _runtime_result(
        execution="live",
        settings=settings,
        operations=operations,
        headline="Live AgentCore Runtime path lists read-only runtimes.",
        provider="bedrock-agentcore-control",
        credential_source=credential_source,
        data=data,
    )


def _live_read_operation(
    *,
    port: AwsPort,
    settings: Settings,
    operation_name: str,
    params: Mapping[str, Any],
) -> tuple[OperationOutcome, Mapping[str, Any]]:
    demo_operation = port_operation(
        port=port,
        service=_CONTROL,
        operation_name=operation_name,
        params=params,
        execution="live",
        effect="read",
        fixture_id=_fixture_id(operation_name),
        endpoint_url=_control_endpoint(settings),
    )
    return demo_operation.outcome, demo_operation.payload


def create_agent_runtime_request() -> dict[str, Any]:
    """Return a valid CreateAgentRuntime request fixture."""
    return {
        "agentRuntimeName": "awsai_demo_runtime",
        "agentRuntimeArtifact": {
            "codeConfiguration": {
                "code": {
                    "s3": {
                        "bucket": "awsai-demo-runtime-bucket",
                        "prefix": "runtime/worker.zip",
                    },
                },
                "entryPoint": ["python", "-m", "awsai_demo.worker"],
                "runtime": "PYTHON_3_12",
            },
        },
        "clientToken": "awsai-runtime-client-token-000001",
        "networkConfiguration": {"networkMode": "PUBLIC"},
        "protocolConfiguration": {"serverProtocol": "HTTP"},
        "roleArn": "arn:aws:iam::123456789012:role/awsai-demo-runtime",
        "tags": {"project": "aws-ai-stack-workshop"},
    }


def invoke_agent_runtime_request() -> dict[str, Any]:
    """Return a valid InvokeAgentRuntime request fixture."""
    return {
        "agentRuntimeArn": (
            "arn:aws:bedrock-agentcore:us-east-1:123456789012:"
            "runtime/awsai_demo_runtime"
        ),
        "contentType": "application/json",
        "payload": json.dumps({"prompt": scenario.BRIEF}).encode(),
        "runtimeSessionId": "awsai-demo-session-000000000000000",
    }


def list_agent_runtimes_request() -> dict[str, Any]:
    """Return the read-only runtime list request."""
    return {"maxResults": 10}


def _runtime_result(
    *,
    execution: Execution,
    settings: Settings,
    operations: Sequence[OperationOutcome],
    headline: str,
    provider: str,
    credential_source: CredentialSource,
    data: Mapping[str, object],
) -> DemoResult:
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution=execution,
        headline=headline,
        operations=operations,
        evidence=evidence(
            sdk_invoked=any(
                item["mode"] != "not_run" or item["request_validated"]
                for item in operations
            ),
            network_attempted=any(
                item["transport"] in {"aws", "external", "loopback"}
                for item in operations
            ),
            aws_executed=any(
                item["execution_target"] == "aws" and item["response_received"]
                for item in operations
            ),
            provider=provider,
            region=settings.region,
            credential_source=credential_source,
            fixture_id=next(
                (
                    item["fixture_id"]
                    for item in operations
                    if item["fixture_id"]
                ),
                None,
            )
            if execution != "live"
            else None,
            packages=_package_versions(
                ("bedrock-agentcore", "strands-agents", "botocore"),
            ),
        ),
        data=dict(data),
    )


def _runtime_attempt_failed(settings: Settings, message: str) -> DemoResult:
    operations = [
        operation(
            service="bedrock-agentcore",
            operation="GET /ping",
            execution_target="local",
            mode="attempt_failed",
            status="blocked",
            effect="none",
            transport="loopback",
            endpoint_url="http://127.0.0.1:0",
            response_received=False,
            request_validated=True,
            error_code="timeout",
        ),
    ]
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution="offline",
        headline="Local AgentCore Runtime loopback contract did not answer.",
        operations=operations,
        evidence=evidence(
            sdk_invoked=True,
            network_attempted=True,
            provider="local-agentcore-runtime",
            region=settings.region,
            packages=_package_versions(
                ("bedrock-agentcore", "strands-agents", "botocore"),
            ),
        ),
        data={"runtime": {"ping_status": "unavailable"}},
        error=error("timeout", message),
    )


def _validated_not_run(
    row: tuple[str, str, Mapping[str, Any]],
) -> OperationOutcome:
    service, name, params = row
    validate_request(
        service=service,
        operation_name=name,
        params=params,
        fixture_id=_fixture_id(name),
    )
    return not_run_operation(
        service=service,
        operation_name=name,
        error_code="contract_only",
        request_validated=True,
    )


def _request_rows() -> tuple[tuple[str, str, dict[str, Any]], ...]:
    return (
        (_CONTROL, "CreateAgentRuntime", create_agent_runtime_request()),
        (_RUNTIME, "InvokeAgentRuntime", invoke_agent_runtime_request()),
        (_CONTROL, "ListAgentRuntimes", list_agent_runtimes_request()),
    )


def _fixture_id(operation_name: str) -> str:
    return f"{FIXTURE_PREFIX}-{operation_name}-v1"


def _runtime_network_policy(host: str, port: int) -> NetworkPolicy:
    base = active_policy() or NetworkPolicy()
    return base.with_endpoint(
        registered_loopback_endpoint("agentcore-runtime", host, port),
    )


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_ping(
    host: str,
    port: int,
    policy: ExecutionPolicy,
    *,
    get_json: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Poll /ping until the worker answers or the deadline passes."""
    fetch = _get_json if get_json is None else get_json
    deadline = time.monotonic() + min(10, policy.max_wall_seconds)
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            payload = fetch(host, port, "/ping", policy)
            if payload.get("status"):
                return payload
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    message = "runtime worker did not answer /ping"
    if last_error is not None:
        message = f"{message}: {last_error}"
    raise OSError(message)


def _get_json(
    host: str,
    port: int,
    path: str,
    policy: ExecutionPolicy,
) -> Mapping[str, Any]:
    conn = http.client.HTTPConnection(
        host,
        port,
        timeout=max(1, min(policy.max_wall_seconds, 10)),
    )
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return _read_json_response(response)
    finally:
        conn.close()


def _post_json(
    host: str,
    port: int,
    path: str,
    payload: Mapping[str, Any],
    policy: ExecutionPolicy,
) -> Mapping[str, Any]:
    body = json.dumps(payload, sort_keys=True).encode()
    conn = http.client.HTTPConnection(
        host,
        port,
        timeout=max(1, min(policy.max_wall_seconds, 10)),
    )
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        return _read_json_response(response)
    finally:
        conn.close()


def _read_json_response(
    response: http.client.HTTPResponse,
) -> Mapping[str, Any]:
    body = response.read()
    if response.status >= 400:
        message = f"runtime returned HTTP {response.status}"
        raise OSError(message)
    payload = json.loads(body.decode())
    if not isinstance(payload, dict):
        message = "runtime response was not a JSON object"
        raise OSError(message)
    return cast("Mapping[str, Any]", payload)


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        process.communicate(timeout=1)
        return
    process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def _start_worker(
    command: Sequence[str],
    env: Mapping[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(  # noqa: S603
        command,
        cwd=str(Path.cwd()),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _live_port_from_session(
    settings: Settings,
    policy: ExecutionPolicy,
    selected_session: SelectedSession | None,
    *,
    boto_port_factory: BotoPortFactory | None = None,
) -> tuple[AwsPort | None, CredentialSource]:
    if selected_session is None:
        default_port = resolve_boto_port(
            execution="live",
            settings=settings,
            policy=policy,
            factory=boto_port_factory,
        )
        if default_port is None:
            return None, "none"
        return default_port.port, default_port.credential_source
    return (
        BotoAwsPort(
            session=cast("BotoSessionPort", selected_session.session),
            region_name=selected_session.region,
            config=bounded_boto_config(policy),
        ),
        selected_session.source,
    )


def _control_endpoint(settings: Settings) -> str:
    return f"https://bedrock-agentcore-control.{settings.region}.amazonaws.com"


def _credential_source(
    selected_session: SelectedSession | None,
) -> CredentialSource:
    if selected_session is None:
        return "none"
    return selected_session.source


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions
