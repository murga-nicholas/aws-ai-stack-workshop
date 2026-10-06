"""Amazon Bedrock Agents Classic return-control demo.

Technology: Amazon Bedrock Agents Classic.
Lane: agents.
Lifecycle refs: agents-classic, agents-classic-multi-agent.
Run:
    uv run awsai-demo agents-classic
    uv run awsai-demo agents-classic --execution live
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from botocore.exceptions import ClientError

from awsai_demo.contracts import ErrorCode, operation
from awsai_demo.demo_support import (
    AwsPort,
    AwsResponse,
    build_default_boto_port,
    build_result,
    contract_only,
    fixture_operation,
    require_port_not_run,
    unsupported_emulator,
    validate_request,
)
from awsai_demo.stubs import named_fixture_stream, stubbed_client

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from awsai_demo.contracts import (
        CredentialSource,
        DemoResult,
        OperationOutcome,
    )
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.runtime import Execution, Settings

_DEMO = "agents-classic"
_TECHNOLOGY = "Bedrock Agents Classic"
_REFS = ["agents-classic", "agents-classic-multi-agent"]
_SERVICE = "bedrock-agent"
_RUNTIME_SERVICE = "bedrock-agent-runtime"
_AGENT_ID = "ABCDEFGHIJ"
_ALIAS_ID = "TSTALIASID"
_SESSION_ID = "classic-session-123"
_INVOCATION_ID = "approval-contract"
_MAINTENANCE_TEXT = "Bedrock Agents is in Maintenance Mode"


def run_agents_classic_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Run Agents Classic return-control and list diagnostics."""
    credential_source: CredentialSource | None = None
    if port is None:
        default_port = build_default_boto_port(
            execution=execution,
            settings=settings,
            policy=policy,
            session_factory=session_factory,
            csv_loader=csv_loader,
        )
        if default_port is not None:
            port = default_port.port
            credential_source = default_port.credential_source
    if execution == "offline":
        operations, data = _run_offline()
    elif execution == "emulator":
        operations, data = _run_emulator()
    else:
        operations, data = _run_live(settings, port)
    return build_result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        execution=execution,
        headline=_headline(data),
        operations=operations,
        settings=settings,
        data=data,
        credential_source=credential_source,
    )


def _run_offline() -> tuple[list[OperationOutcome], dict[str, object]]:
    validate_request(
        service=_RUNTIME_SERVICE,
        operation_name="InvokeAgent",
        params=invoke_agent_params(),
        fixture_id=_fixture_id("InvokeAgentReturnControl"),
    )
    stream = named_fixture_stream(
        fixture_id=_fixture_id("InvokeAgentReturnControl"),
        events=_return_control_events(),
    )
    return_control = next(iter(stream))["returnControl"]
    validate_request(
        service=_RUNTIME_SERVICE,
        operation_name="InvokeAgent",
        params=return_control_result_params(),
        fixture_id=_fixture_id("InvokeAgentReturnResult"),
    )
    with stubbed_client(_SERVICE) as stubber:
        client = stubber.client
        stubber.add_response(
            "list_agents",
            {"agentSummaries": []},
            expected_params=list_agents_params(),
        )
        listed = client.list_agents(**list_agents_params())
    operations = [
        fixture_operation(
            service=_RUNTIME_SERVICE,
            operation_name="InvokeAgent",
            fixture_id=_fixture_id("InvokeAgentReturnControl"),
            effect="none",
        ),
        fixture_operation(
            service=_RUNTIME_SERVICE,
            operation_name="InvokeAgent",
            fixture_id=_fixture_id("InvokeAgentReturnResult"),
            effect="none",
        ),
        fixture_operation(
            service=_SERVICE,
            operation_name="ListAgents",
            fixture_id=_fixture_id("ListAgents"),
            effect="read",
        ),
        _diagnostic_operation(),
    ]
    return operations, _classic_data(
        listed,
        diagnostic="authorization_denied",
        return_control=cast("Mapping[str, Any]", return_control),
    )


def _run_emulator() -> tuple[list[OperationOutcome], dict[str, object]]:
    return (
        [
            unsupported_emulator(
                service=_RUNTIME_SERVICE,
                operation_name="InvokeAgent",
            ),
            unsupported_emulator(
                service=_SERVICE, operation_name="ListAgents"
            ),
            unsupported_emulator(
                service=_SERVICE,
                operation_name="ClassifyAccessDeniedException",
            ),
        ],
        {
            "agents": {"classic_listed": 0},
            "diagnostic": "not_observed",
            "emulator": "not_supported",
        },
    )


def _run_live(
    settings: Settings,
    port: AwsPort | None,
) -> tuple[list[OperationOutcome], dict[str, object]]:
    operations = [
        _validated_contract_only(
            service=_RUNTIME_SERVICE,
            operation_name="InvokeAgent",
            params=invoke_agent_params(),
        ),
    ]
    payload: Mapping[str, Any] = {"agentSummaries": []}
    diagnostic = "not_observed"
    if port is None:
        operations.append(
            require_port_not_run(
                service=_SERVICE, operation_name="ListAgents"
            ),
        )
    else:
        outcome, payload, diagnostic = _live_list_agents(
            settings=settings,
            port=port,
        )
        operations.append(outcome)
    operations.append(_diagnostic_operation())
    data = _classic_data(payload, diagnostic=diagnostic, return_control={})
    data["region"] = settings.region
    return operations, data


def invoke_agent_params() -> dict[str, str]:
    """Return a first-turn InvokeAgent request."""
    return {
        "agentId": _AGENT_ID,
        "agentAliasId": _ALIAS_ID,
        "sessionId": _SESSION_ID,
        "inputText": "Should the support pilot proceed?",
    }


def return_control_result_params() -> dict[str, Any]:
    """Return the return-control answer InvokeAgent request."""
    return {
        "agentId": _AGENT_ID,
        "agentAliasId": _ALIAS_ID,
        "sessionId": _SESSION_ID,
        "sessionState": {
            "invocationId": _INVOCATION_ID,
            "returnControlInvocationResults": [
                {
                    "functionResult": {
                        "actionGroup": "pilot",
                        "function": "accept_proposal",
                        "responseBody": {
                            "TEXT": {"body": "approved=False"},
                        },
                    },
                },
            ],
        },
    }


def list_agents_params() -> dict[str, int]:
    """Return bounded ListAgents params."""
    return {"maxResults": 10}


def classify_access_denied(exc: ClientError) -> ErrorCode:
    """Classify Agents Classic maintenance messages exactly."""
    error_payload = exc.response.get("Error", {})
    code = str(error_payload.get("Code", ""))
    message = str(error_payload.get("Message", ""))
    if code == "AccessDeniedException" and _MAINTENANCE_TEXT in message:
        return "maintenance_mode"
    return "authorization_denied"


def _live_list_agents(
    *,
    settings: Settings,
    port: AwsPort,
) -> tuple[OperationOutcome, Mapping[str, Any], str]:
    params = list_agents_params()
    validate_request(
        service=_SERVICE,
        operation_name="ListAgents",
        params=params,
        fixture_id=_fixture_id("ListAgents"),
    )
    endpoint = _port_endpoint(port) or _default_endpoint(settings)
    try:
        raw = port.call(_SERVICE, "ListAgents", params)
    except ClientError as exc:
        diagnostic = classify_access_denied(exc)
        return (
            operation(
                service=_SERVICE,
                operation="ListAgents",
                execution_target="aws",
                mode="live_service",
                status="blocked",
                effect="read",
                transport="aws",
                endpoint_url=endpoint,
                response_received=True,
                request_validated=True,
                http_status=_http_status(exc),
                error_code=diagnostic,
            ),
            {},
            diagnostic,
        )
    payload, endpoint = _normalize_response(raw, endpoint)
    return (
        operation(
            service=_SERVICE,
            operation="ListAgents",
            execution_target="aws",
            mode="live_service",
            effect="read",
            transport="aws",
            endpoint_url=endpoint,
            response_received=True,
            request_validated=True,
        ),
        payload,
        "not_observed",
    )


def _validated_contract_only(
    *,
    service: str,
    operation_name: str,
    params: Mapping[str, Any],
) -> OperationOutcome:
    validate_request(
        service=service,
        operation_name=operation_name,
        params=params,
        fixture_id=_fixture_id(operation_name),
    )
    return contract_only(service=service, operation_name=operation_name)


def _diagnostic_operation() -> OperationOutcome:
    return operation(
        service=_SERVICE,
        operation="ClassifyAccessDeniedException",
        execution_target="local",
        mode="local_execution",
        effect="none",
        transport="none",
    )


def _classic_data(
    payload: Mapping[str, Any],
    *,
    diagnostic: str,
    return_control: Mapping[str, Any],
) -> dict[str, object]:
    listed = len(cast("Sequence[Any]", payload.get("agentSummaries", ())))
    return {
        "agents": {"classic_listed": listed},
        "diagnostic": diagnostic,
        "returnControl": dict(return_control),
        "observation": (
            f"{listed} agents returned in this region; historical "
            "allowlisting is not observable from a list call."
        ),
    }


def _headline(data: Mapping[str, object]) -> str:
    observation = data.get("observation")
    if isinstance(observation, str):
        return observation
    return "Agents Classic return control was validated."


def _return_control_events() -> tuple[dict[str, object], ...]:
    return (
        {
            "returnControl": {
                "invocationId": _INVOCATION_ID,
                "invocationInputs": [
                    {
                        "functionInvocationInput": {
                            "actionGroup": "pilot",
                            "function": "accept_proposal",
                        },
                    },
                ],
            },
        },
    )


def _normalize_response(
    response: AwsResponse | Mapping[str, Any],
    fallback_endpoint: str,
) -> tuple[Mapping[str, Any], str]:
    if isinstance(response, AwsResponse):
        return response.payload, response.endpoint_url or fallback_endpoint
    return response, fallback_endpoint


def _http_status(exc: ClientError) -> int | None:
    metadata = exc.response.get("ResponseMetadata", {})
    status = metadata.get("HTTPStatusCode")
    return int(status) if isinstance(status, int) else None


def _port_endpoint(port: AwsPort) -> str | None:
    endpoint = getattr(port, "endpoint_url", None)
    return endpoint if isinstance(endpoint, str) else None


def _default_endpoint(settings: Settings) -> str:
    return f"https://bedrock-agent.{settings.region}.amazonaws.com"


def _fixture_id(operation_name: str) -> str:
    return f"agents-classic-{operation_name}-v1"
