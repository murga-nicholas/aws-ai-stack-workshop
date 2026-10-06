"""AgentCore Gateway and Cedar policy demo.

Lane: agents; lifecycle: agentcore-gateway, agentcore-policy.
Run: uv run awsai-demo agentcore-gateway --execution offline.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cedarpy
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

from awsai_demo import scenario
from awsai_demo.contracts import (
    CredentialSource,
    DemoResult,
    Effect,
    Execution,
    OperationOutcome,
    evidence,
    operation,
    result,
)
from awsai_demo.demo_support import (
    AwsPort,
    AwsResponse,
    BotoAwsPort,
    BotoSessionPort,
    bounded_boto_config,
    not_run_operation,
    require_port_not_run,
    validate_request,
)
from awsai_demo.network import (
    active_policy,
    bootstrap_python_args,
    environment_for_subprocess,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings
from awsai_demo.stubs import validate_operation_request

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from awsai_demo.credentials import SelectedSession
    from awsai_demo.network import NetworkPolicy

FIXTURE_PREFIX = "agentcore-gateway-request"
POLICY_DENIED_REASON = (
    "context.approved must be true and proposal_version must match"
)
_DEMO = "agentcore-gateway"
_TECHNOLOGY = "AgentCore Gateway and Policy"
_REFS = ("agentcore-gateway", "agentcore-policy")
_SERVICE = "bedrock-agentcore-control"

# slide: cedar-policy
CEDAR_POLICY = """
@id("approve-matching-proposal")
permit(
  principal == User::"pilot-operator",
  action == Action::"accept_proposal",
  resource == Proposal::"support-pilot"
)
when {
  context.approved == true &&
  context.proposal_version == resource.proposal_version
};
""".strip()
# end-slide: cedar-policy


def run_agentcore_gateway_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    port: AwsPort | None = None,
    selected_session: SelectedSession | None = None,
) -> DemoResult:
    """Run local MCP/Cedar proof and AgentCore request contracts."""
    active_settings = settings or Settings()
    chosen_policy = policy or ExecutionPolicy()
    if execution == "offline":
        return _run_offline(active_settings)
    if execution == "emulator":
        return _run_emulator_contract_only(active_settings)
    return _run_live_read_only(
        active_settings,
        chosen_policy,
        port=port,
        selected_session=selected_session,
    )


def _run_offline(settings: Settings) -> DemoResult:
    mcp_result = run_mcp_round_trip()
    denied = evaluate_accept_proposal(approved=False)
    allowed = evaluate_accept_proposal(approved=True)
    request_operations = validate_agentcore_gateway_contracts()
    operations = [
        operation(
            service="mcp",
            operation="stdio list/call price_pilot",
            execution_target="local",
            mode="local_execution",
        ),
        operation(
            service="cedarpy",
            operation="is_authorized accept_proposal",
            execution_target="local",
            mode="local_execution",
        ),
        *request_operations,
    ]
    return _gateway_result(
        execution="offline",
        settings=settings,
        operations=operations,
        headline="MCP tools listed locally; Cedar denied unapproved accept.",
        provider="offline",
        credential_source="none",
        data=_offline_data(mcp_result, denied, allowed, request_operations),
    )


def _run_emulator_contract_only(settings: Settings) -> DemoResult:
    operations = [_validated_not_run(row) for row in _request_rows()]
    return _gateway_result(
        execution="emulator",
        settings=settings,
        operations=operations,
        headline="AgentCore Gateway has no LocalStack emulator path.",
        provider="localstack-contract-only",
        credential_source="none",
        data={
            "emulator": "contract_only",
            "request_contracts": [row[0] for row in _request_rows()],
        },
    )


def _run_live_read_only(
    settings: Settings,
    policy: ExecutionPolicy,
    *,
    port: AwsPort | None,
    selected_session: SelectedSession | None,
) -> DemoResult:
    active_port = port or _live_port_from_session(
        settings,
        policy,
        selected_session,
    )
    operations = [_validated_not_run(row) for row in _request_rows()[:4]]
    data: dict[str, object] = {
        "request_contracts": [row[0] for row in _request_rows()[:4]],
    }
    if active_port is None:
        operations.extend(
            require_port_not_run(service=_SERVICE, operation_name=name)
            for name in ("ListGateways", "ListPolicyEngines")
        )
        data["live_adapter"] = "missing"
    else:
        gateway_outcome, gateway_payload = _live_read_operation(
            port=active_port,
            settings=settings,
            operation_name="ListGateways",
            params=_list_gateways_params(),
        )
        policy_outcome, policy_payload = _live_read_operation(
            port=active_port,
            settings=settings,
            operation_name="ListPolicyEngines",
            params=_list_policy_engines_params(),
        )
        operations.extend((gateway_outcome, policy_outcome))
        data.update(_live_data(gateway_payload, policy_payload))
    return _gateway_result(
        execution="live",
        settings=settings,
        operations=operations,
        headline="Live AgentCore Gateway path lists read-only resources.",
        provider="bedrock-agentcore-control",
        credential_source=_credential_source(selected_session),
        data=data,
    )


def _live_read_operation(
    *,
    port: AwsPort,
    settings: Settings,
    operation_name: str,
    params: Mapping[str, Any],
) -> tuple[OperationOutcome, Mapping[str, Any]]:
    validate_request(
        service=_SERVICE,
        operation_name=operation_name,
        params=params,
        fixture_id=_fixture_id(operation_name),
    )
    raw_response = port.call(_SERVICE, operation_name, params)
    payload, endpoint_url = _normalize_port_response(
        raw_response,
        fallback_endpoint=_default_live_endpoint(settings),
    )
    return (
        operation(
            service=_SERVICE,
            operation=operation_name,
            execution_target="aws",
            mode="live_service",
            effect="read",
            transport="aws",
            endpoint_url=endpoint_url,
            response_received=True,
            request_validated=True,
        ),
        payload,
    )


def _normalize_port_response(
    response: AwsResponse | Mapping[str, Any],
    *,
    fallback_endpoint: str,
) -> tuple[Mapping[str, Any], str]:
    if isinstance(response, AwsResponse):
        return response.payload, response.endpoint_url or fallback_endpoint
    return response, fallback_endpoint


def _default_live_endpoint(settings: Settings) -> str:
    return f"https://bedrock-agentcore-control.{settings.region}.amazonaws.com"


def _live_port_from_session(
    settings: Settings,
    policy: ExecutionPolicy,
    selected_session: SelectedSession | None,
) -> AwsPort | None:
    del settings
    if selected_session is None:
        return None
    return BotoAwsPort(
        session=cast("BotoSessionPort", selected_session.session),
        region_name=selected_session.region,
        config=bounded_boto_config(policy),
    )


def run_mcp_round_trip() -> dict[str, object]:
    """List and call the local MCP server through Strands MCPClient."""
    server = build_stdio_server_parameters()
    transport = cast("Any", lambda: stdio_client(server))
    client = MCPClient(transport, startup_timeout=10)
    with client:
        tools = client.list_tools_sync()
        tool_names = sorted(tool.tool_name for tool in tools)
        price = client.call_tool_sync(
            "call-price",
            "price_pilot",
            {},
            read_timeout_seconds=timedelta(seconds=10),
        )
    return {
        "tool_names": tuple(tool_names),
        "price_total": _first_json_value(price, "total_usd"),
    }


def build_stdio_server_parameters(
    *,
    policy_lookup: Callable[[], NetworkPolicy | None] | None = None,
) -> StdioServerParameters:
    """Build subprocess parameters for the local stdio MCP server."""
    args = [
        "-c",
        "from awsai_demo.gateway_server import main; main()",
    ]
    env = None
    lookup = active_policy if policy_lookup is None else policy_lookup
    if (guard_policy := lookup()) is not None:
        args = bootstrap_python_args(args)
        env = environment_for_subprocess(guard_policy)
    return StdioServerParameters(
        command=sys.executable,
        args=args,
        env=env,
        cwd=str(Path.cwd()),
    )


def evaluate_accept_proposal(*, approved: bool) -> dict[str, object]:
    """Evaluate accept_proposal against Cedar."""
    cost = scenario.price_pilot()
    version = scenario.proposal_version(cost)
    decision = cedarpy.is_authorized(
        {
            "principal": {"type": "User", "id": "pilot-operator"},
            "action": {"type": "Action", "id": "accept_proposal"},
            "resource": {"type": "Proposal", "id": "support-pilot"},
            "context": {
                "approved": approved,
                "proposal_version": version,
            },
        },
        CEDAR_POLICY,
        [
            {
                "uid": {"type": "Proposal", "id": "support-pilot"},
                "attrs": {"proposal_version": version},
                "parents": [],
            },
        ],
    )
    return {
        "allowed": bool(decision.allowed),
        "reason": "" if decision.allowed else POLICY_DENIED_REASON,
        "proposal_version": version,
    }


def validate_agentcore_gateway_contracts() -> list[OperationOutcome]:
    """Validate AgentCore Gateway request shapes."""
    operations: list[OperationOutcome] = []
    for name, params, effect in _request_rows():
        fixture_id = _fixture_id(name)
        validation = validate_operation_request(
            service_name=_SERVICE,
            operation_name=name,
            params=params,
            fixture_id=fixture_id,
        )
        operations.append(
            operation(
                service=_SERVICE,
                operation=name,
                mode="local_contract",
                effect=effect,
                fixture_id=validation.fixture_id,
                request_validated=validation.request_validated,
            ),
        )
    return operations


def _gateway_result(
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
            packages=_package_versions(
                ("strands-agents", "mcp", "cedarpy", "botocore"),
            ),
        ),
        data=dict(data),
    )


def _offline_data(
    mcp_result: Mapping[str, object],
    denied: Mapping[str, object],
    allowed: Mapping[str, object],
    request_operations: Sequence[OperationOutcome],
) -> dict[str, object]:
    return {
        "tool_names": mcp_result["tool_names"],
        "price_total": mcp_result["price_total"],
        "policy_denied": not denied["allowed"],
        "policy_allowed": allowed["allowed"],
        "denied_reason": denied["reason"],
        "proposal_version": denied["proposal_version"],
        "request_contracts": [op["operation"] for op in request_operations],
    }


def _live_data(
    gateways: Mapping[str, Any],
    policy_engines: Mapping[str, Any],
) -> dict[str, object]:
    gateway_rows = cast("Sequence[object]", gateways.get("gateways", ()))
    policy_rows = cast(
        "Sequence[object]",
        policy_engines.get("policyEngines", ()),
    )
    return {
        "live_gateways": len(gateway_rows),
        "live_policy_engines": len(policy_rows),
    }


def _validated_not_run(
    row: tuple[str, dict[str, Any], Effect],
) -> OperationOutcome:
    name, params, _effect = row
    validate_request(
        service=_SERVICE,
        operation_name=name,
        params=params,
        fixture_id=_fixture_id(name),
    )
    return not_run_operation(
        service=_SERVICE,
        operation_name=name,
        error_code="contract_only",
        request_validated=True,
    )


def _request_rows() -> tuple[tuple[str, dict[str, Any], Effect], ...]:
    return (
        ("CreateGateway", _create_gateway_params(), "write"),
        ("CreateGatewayTarget", _create_gateway_target_params(), "write"),
        ("CreatePolicyEngine", _create_policy_engine_params(), "write"),
        ("CreatePolicy", _create_policy_params(), "write"),
        ("ListGateways", _list_gateways_params(), "read"),
        ("ListPolicyEngines", _list_policy_engines_params(), "read"),
    )


def _fixture_id(operation_name: str) -> str:
    return f"{FIXTURE_PREFIX}-{operation_name}-v1"


def _create_gateway_params() -> dict[str, Any]:
    return {
        "name": "awsai_demo_gateway",
        "description": "Offline request contract for a workshop gateway.",
        "clientToken": "awsai-demo-gateway-token-000000001",
        "roleArn": "arn:aws:iam::123456789012:role/awsai-demo-gateway",
        "protocolType": "MCP",
        "authorizerType": "NONE",
        "tags": {"project": "aws-ai-stack-workshop"},
    }


def _create_gateway_target_params() -> dict[str, Any]:
    return {
        "gatewayIdentifier": "gw-1234567890",
        "name": "awsai_demo_mcp_target",
        "clientToken": "awsai-demo-gateway-target-token-0001",
        "targetConfiguration": {
            "mcp": {
                "mcpServer": {
                    "endpoint": "http://127.0.0.1:7777/mcp",
                    "listingMode": "DYNAMIC",
                },
            },
        },
    }


def _create_policy_engine_params() -> dict[str, Any]:
    return {
        "name": "awsai_demo_policy_engine",
        "description": "Offline request contract for Cedar policy engine.",
        "clientToken": "awsai-demo-policy-engine-token-00001",
        "tags": {"project": "aws-ai-stack-workshop"},
    }


def _create_policy_params() -> dict[str, Any]:
    return {
        "name": "awsai_demo_accept_policy",
        "policyEngineId": "pe-1234567890",
        "definition": {"cedar": {"statement": CEDAR_POLICY}},
        "validationMode": "FAIL_ON_ANY_FINDINGS",
        "enforcementMode": "ACTIVE",
        "clientToken": "awsai-demo-policy-token-00000000001",
    }


def _list_gateways_params() -> dict[str, Any]:
    return {"maxResults": 5}


def _list_policy_engines_params() -> dict[str, Any]:
    return {"maxResults": 5}


def _credential_source(
    selected_session: SelectedSession | None,
) -> CredentialSource:
    if selected_session is None:
        return "none"
    return selected_session.source


def _first_json_value(payload: Mapping[str, Any], key: str) -> object:
    structured = payload.get("structuredContent")
    if isinstance(structured, dict) and key in structured:
        return structured[key]
    for item in payload.get("content", []):
        if isinstance(item, dict) and "json" in item:
            data = item["json"]
            if isinstance(data, dict) and key in data:
                return data[key]
    message = f"MCP result did not include JSON key {key}"
    raise ValueError(message)


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions
